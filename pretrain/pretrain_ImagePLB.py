# coding=UTF-8
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.data_utils import transforms_for_pretrain
from dataloader.dataset import PocketWithLigandImageTrajectoryDataset
from model.base.predictor import FramePredictor
from model.egnn_pytorch import EGNN_Network
from model.fusion_model import CrossModalityMultiHeadAttention
from model.model_utils import write_result_dict_to_tb, save_checkpoint, load_checkpoint
from model.predictor import NextComplexPredictor, NextPredictorWithCondition
from pretrain.pretrain_utils import train_one_epoch_vfds_add_diff_loss as train_one_epoch
from utils.public_utils import fix_train_random_seed, cal_torch_model_params


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of ImagePLB-P')

    # basic
    parser.add_argument('--dataroot', type=str, default="../datasets/pre-training/MISATO-10-complex/processed/")
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # model params
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')

    # pocket params
    parser.add_argument('--max_len_pocket', type=int, default=100, help='')
    parser.add_argument('--center', action='store_true', default=False, help='')
    parser.add_argument('--n_dim_graph', type=int, default=100, help='')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--weighted_loss', action='store_true', help='add regularization for multi-task loss')

    # train
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument('--n_ckpt_save', default=1, type=int,
                        help='save a checkpoint every n epochs, n_ckpt_save=0: no save')
    parser.add_argument('--n_batch_step_optim', default=1, type=int, help='update model parameters every n batches')

    # labmda
    parser.add_argument('--lambda_next_mol', default=1, type=float, help='')
    parser.add_argument('--lambda_next_pocket', default=1, type=float, help='')
    parser.add_argument('--lambda_next_complex', default=1, type=float, help='')

    # log
    parser.add_argument('--log_dir', default='./experiments/pretrain_ImagePLB', help='path to log')
    parser.add_argument('--tb_step_num', default=100, type=int,
                        help='The training results of every n steps are recorded in tb')

    # Parse arguments
    return parser.parse_args()


def print_only_rank0(text, logger=None):
    log = print if logger is None else logger.info
    log(text)


def is_rank0():
    return 0 == 0


def main(args):
    # [*] initialize the distributed process group and device
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # initializing logger
    args.log_dir = args.log_dir / Path(args.model_name)
    args.tb_dir = os.path.join(args.log_dir, "tb")
    args.tb_step_dir = os.path.join(args.log_dir, "tb_step")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    print_only_rank0("run command: " + " ".join(sys.argv))
    print_only_rank0("args: {}".format(args))
    print_only_rank0("log_dir: {}".format(args.log_dir))

    ########################## load dataset
    # transforms
    train_transforms = transforms_for_pretrain(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    # Load dataset
    print_only_rank0("load dataset")
    train_dataset = PocketWithLigandImageTrajectoryDataset(args.dataroot, center=args.center,
                                                           img_transform=train_transforms,
                                                           max_len_pocket=args.max_len_pocket)

    # initialize data loader
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=True)

    ########################## load model
    # Load model
    vision_model = FramePredictor(model_name=args.model_name, head_arch="none", num_tasks=None, pretrained=True).to(
        device)
    n_dim_vision = vision_model.in_features
    gnn_model = EGNN_Network(num_tokens=100, dim=args.n_dim_graph, depth=6, num_nearest_neighbors=32, dropout=0.15,
                             global_linear_attn_every=1, coor_weights_clamp_value=2, norm_coors=False,
                             update_coors=False).to(device)
    fusion_model = CrossModalityMultiHeadAttention(n_dim1=n_dim_vision, n_dim2=args.n_dim_graph,
                                                   num_of_attention_heads=2, hidden_size=args.n_dim_graph).to(device)

    nextMolPredictor = NextPredictorWithCondition(in_features=n_dim_vision, condition_dim=args.n_dim_graph).to(device)
    nextPocketPredictor = NextPredictorWithCondition(in_features=args.n_dim_graph, condition_dim=n_dim_vision).to(
        device)
    nextComplexPredictor = NextComplexPredictor(in_features=args.n_dim_graph, out_features=args.n_dim_graph).to(device)

    if len(device_ids) > 1:
        vision_model = torch.nn.DataParallel(vision_model, device_ids=device_ids)
        gnn_model = torch.nn.DataParallel(gnn_model, device_ids=device_ids)
        fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
        nextMolPredictor = torch.nn.DataParallel(nextMolPredictor, device_ids=device_ids)
        nextPocketPredictor = torch.nn.DataParallel(nextPocketPredictor, device_ids=device_ids)
        nextComplexPredictor = torch.nn.DataParallel(nextComplexPredictor, device_ids=device_ids)

    vision_model_params_num = cal_torch_model_params(vision_model, unit="M")
    gnn_model_params_num = cal_torch_model_params(gnn_model, unit="M")
    fusion_model_params_num = cal_torch_model_params(fusion_model, unit="M")
    nextMolPredictor_params_num = cal_torch_model_params(nextMolPredictor, unit="M")
    nextPocketPredictor_params_num = cal_torch_model_params(nextPocketPredictor, unit="M")
    nextComplexPredictor_params_num = cal_torch_model_params(nextComplexPredictor, unit="M")

    print_only_rank0("vision model: {}".format(vision_model_params_num))
    print_only_rank0("gnn model: {}".format(gnn_model_params_num))
    print_only_rank0("fusion model: {}".format(fusion_model_params_num))
    print_only_rank0("nextMolPredictor: {}".format(nextMolPredictor_params_num))
    print_only_rank0("nextPocketPredictor: {}".format(nextPocketPredictor_params_num))
    print_only_rank0("nextComplexPredictor: {}".format(nextComplexPredictor_params_num))

    # Loss and optimizer
    optim_params = [{"params": vision_model.parameters()},
                    {"params": gnn_model.parameters()},
                    {"params": fusion_model.parameters()},
                    {"params": nextMolPredictor.parameters()},
                    {"params": nextPocketPredictor.parameters()},
                    {"params": nextComplexPredictor.parameters()}]
    optimizer = SGD(optim_params, momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    criterionL1 = nn.L1Loss()
    criterionL1_none = nn.L1Loss(reduction="none")

    # lr scheduler
    lr_scheduler = None

    # Resume weights
    model_dict = {
        "vision_model": vision_model, "gnn_model": gnn_model, "fusion_model": fusion_model,
        "nextMolPredictor": nextMolPredictor, "nextPocketPredictor": nextPocketPredictor,
        "nextComplexPredictor": nextComplexPredictor
    }
    if args.resume is not None:
        flag, resume_desc = load_checkpoint(args.resume, model_dict)
        args.start_epoch = int(resume_desc['epoch'])
        assert flag, "error in loading pretrained model {}.".format(args.resume)
        print_only_rank0("[resume description] {}".format(resume_desc))

    # Tensorboard
    tb_writer = SummaryWriter(log_dir=args.tb_dir)
    tb_step_writer = SummaryWriter(log_dir=args.tb_step_dir)

    ########################## train
    for epoch in range(args.start_epoch, args.epochs):
        train_dict = train_one_epoch(vision_model, gnn_model, fusion_model,
                                     nextMolPredictor=nextMolPredictor,
                                     nextPocketPredictor=nextPocketPredictor,
                                     nextComplexPredictor=nextComplexPredictor,
                                     optimizer=optimizer, data_loader=train_loader,
                                     criterionReg=(criterionL1, criterionL1_none),
                                     device=device, epoch=epoch, lr_scheduler=lr_scheduler, tb_writer=tb_step_writer,
                                     args=args, weighted_loss=args.weighted_loss,
                                     lambda_next_mol=args.lambda_next_mol,
                                     lambda_next_pocket=args.lambda_next_pocket,
                                     lambda_next_complex=args.lambda_next_complex)

        print_only_rank0(str(train_dict))

        # save model
        optimizer_dict = {"optimizer": optimizer}
        lr_scheduler_dict = {"lr_scheduler": lr_scheduler} if lr_scheduler is not None else None

        cur_loss = train_dict["total_loss"]
        if is_rank0() and args.n_ckpt_save > 0 and epoch % args.n_ckpt_save == 0:
            ckpt_pre = "ckpt_epoch={}_loss={:.2f}".format(epoch, cur_loss)
            save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                            train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                            name_pre=ckpt_pre, name_post="")

        write_result_dict_to_tb(tb_writer, train_dict, optimizer_dict={"optimizer": optimizer})


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许数据副本，提高数据加载的效率
    # [*] initialize some arguments
    args = parse_args()
    # [*] run with dp
    main(args)
