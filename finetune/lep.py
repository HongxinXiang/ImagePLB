#coding=UTF-8
import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as opt
import torch_scatter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader.data_utils import transforms_for_finetune
from dataloader.pdbbind_dataset import PDBBindPocketWithLigandImageDataset as PocketWithLigandImageDataset
from finetune.finetune_utils import train_one_epoch, run_eval_cls, save_finetune_ckpt
from model.base.predictor import FramePredictor
from model.egnn_pytorch import EGNN_Network
from model.fusion_model import CrossModalityMultiHeadAttention
from model.model_utils import load_checkpoint
from utils.public_utils import fix_train_random_seed, cal_torch_model_params, is_left_better_right


# [*] Initialize the distributed process group and distributed device
def setup_DDP_mp(init_method, local_rank, rank, world_size, backend="nccl", verbose=False):
    if sys.platform == "win32":
        backend = "gloo"
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return device


def get_y_pred_func(vision_model, graph_model, fusion_model, predictor, video_3d, pocket_feats, pocket_coords, mask, device):
    n_samples, n_views, n_chanel, h, w = video_3d.shape
    feats_3d = vision_model(video_3d.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
    feats_3d_mean = feats_3d.mean(1)

    feats_graph, coors_out = graph_model(pocket_feats, pocket_coords, mask=mask)

    # 交互
    batch_feat_3d = torch.arange(len(feats_3d_mean)).to(device)
    batch_feat_graph = torch.arange(len(feats_graph)).repeat(feats_graph.shape[1], 1).T.flatten().to(device)[
        mask.flatten()]

    feat_3d_v_fusion_g, feat_3d_g_fusion_v = fusion_model(feats_3d_mean, batch_feat_3d, feats_graph[mask],
                                                          batch_feat_graph)
    feat_3d_g_fusion_v_mean = torch_scatter.scatter_mean(feat_3d_g_fusion_v, batch_feat_graph, dim=0)

    y_pred = predictor(torch.concat([feat_3d_v_fusion_g, feat_3d_g_fusion_v_mean], dim=1))

    return y_pred


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of ImagePLB')

    # basic
    parser.add_argument('--dataroot', type=str, default="../datasets/fine-tuning/lep", help='data root')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # ddp
    parser.add_argument("--nodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--ngpus_per_node", default=1, type=int, help="number of GPUs per node for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12355", type=str, help="url used to set up distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")

    # model params
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')

    # pocket params
    parser.add_argument('--max_len_pocket', type=int, default=100, help='')
    parser.add_argument('--center', action='store_true', default=False, help='')
    parser.add_argument('--n_dim_graph', type=int, default=100, help='')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--split_type', default="protein", type=str, help='', choices=["protein"])
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument('--n_batch_step_optim', default=1, type=int, help='update model parameters every n batches')

    parser.add_argument('--egnn_dropout', type=float, default=0.15, help='Dropout rate of egnn.')
    parser.add_argument('--predictor_dropout', type=float, default=0.1, help='Dropout rate of classifier.')

    # log
    parser.add_argument('--log_dir', default='./experiments/finetune_lep', help='path to log')
    parser.add_argument('--tb_step_num', default=100, type=int, help='The training results of every n steps are recorded in tb')

    # Parse arguments
    return parser.parse_args()


def print_only_rank0(text, logger=None):
    log = print if logger is None else logger.info
    if dist.get_rank() == 0:
        log(text)


def is_rank0():
    return dist.get_rank() == 0


def get_tqdm_desc(dataset, epoch):
    tqdm_train_desc = "[train] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_train_desc = "[eval on train set] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_val_desc = "[eval on valid set] dataset: {}; epoch: {}".format(dataset, epoch)
    tqdm_eval_test_desc = "[eval on test set] dataset: {}; epoch: {}".format(dataset, epoch)
    return tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc


def main(local_rank, ngpus_per_node, args):

    args.local_rank = local_rank
    args.rank = args.node_rank * ngpus_per_node + local_rank

    # [*] initialize the distributed process group and device
    device = setup_DDP_mp(init_method=args.dist_url, local_rank=args.local_rank, rank=args.rank,
                          world_size=args.world_size, verbose=True)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # initializing logger
    args.tb_dir = os.path.join(args.log_dir, "tb")
    args.tb_step_dir = os.path.join(args.log_dir, "tb_step")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    print_only_rank0("run command: " + " ".join(sys.argv))
    print_only_rank0("log_dir: {}".format(args.log_dir))
    print_only_rank0("args: {}".format(args))

    ########################## load dataset
    # transforms
    transforms = transforms_for_finetune(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    # Load dataset
    print_only_rank0("load dataset")
    train_dataset = PocketWithLigandImageDataset(args.dataroot, split_type=args.split_type, split="train", center=args.center, img_transform=transforms, max_len_pocket=args.max_len_pocket)
    valid_dataset = PocketWithLigandImageDataset(args.dataroot, split_type=args.split_type, split="val", center=args.center, img_transform=transforms, max_len_pocket=args.max_len_pocket)
    test_dataset = PocketWithLigandImageDataset(args.dataroot, split_type=args.split_type, split="test", center=args.center, img_transform=transforms, max_len_pocket=args.max_len_pocket)

    # initialize data loader
    # [*] using DistributedSampler
    batch_size = args.batch // args.world_size  # [*] // world_size
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=batch_size, sampler=DistributedSampler(train_dataset, shuffle=True), num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=valid_dataset.collate_fn, batch_size=batch_size, sampler=DistributedSampler(valid_dataset, shuffle=True), num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=batch_size, sampler=DistributedSampler(test_dataset, shuffle=True), num_workers=args.workers, pin_memory=True)

    ########################## load model
    # Load model
    vision_model = FramePredictor(model_name=args.model_name, head_arch="none", num_tasks=None, pretrained=True)
    n_dim_vision = vision_model.in_features
    gnn_model = EGNN_Network(num_tokens=100, dim=args.n_dim_graph, depth=6, num_nearest_neighbors=32, dropout=args.egnn_dropout,
                             global_linear_attn_every=1, coor_weights_clamp_value=2, norm_coors=False, update_coors=False)
    fusion_model = CrossModalityMultiHeadAttention(n_dim1=n_dim_vision, n_dim2=args.n_dim_graph, num_of_attention_heads=2, hidden_size=args.n_dim_graph)
    predictor = nn.Sequential(nn.Linear(args.n_dim_graph*2, args.n_dim_graph), nn.ReLU(), nn.Dropout(p=args.predictor_dropout), nn.Linear(args.n_dim_graph, 1))

    vision_model_params_num = cal_torch_model_params(vision_model, unit="M")
    gnn_model_params_num = cal_torch_model_params(gnn_model, unit="M")
    fusion_model_params_num = cal_torch_model_params(fusion_model, unit="M")
    predictor_params_num = cal_torch_model_params(predictor, unit="M")

    print_only_rank0("vision model: {}".format(vision_model_params_num))
    print_only_rank0("gnn model: {}".format(gnn_model_params_num))
    print_only_rank0("fusion model: {}".format(fusion_model_params_num))
    print_only_rank0("predictor model: {}".format(predictor_params_num))

    # Loss and optimizer
    optim_params = [{"params": vision_model.parameters()},
                    {"params": gnn_model.parameters()},
                    {"params": fusion_model.parameters()},
                    {"params": predictor.parameters()}]
    optimizer = Adam(optim_params, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # lr scheduler
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-6)

    # Resume pretrained weights
    model_dict = {"vision_model": vision_model, "gnn_model": gnn_model, "fusion_model": fusion_model}
    if args.resume is not None and args.resume != "None":
        flag, resume_desc = load_checkpoint(args.resume, model_dict)
        assert flag, "error in loading pretrained model {}.".format(args.resume)
        print_only_rank0("[resume description] {}".format(resume_desc))

    # model with DDP
    print_only_rank0("starting DDP.")
    # [*] using DistributedDataParallel
    vision_model = DDP(vision_model.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    gnn_model = DDP(gnn_model.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    fusion_model = DDP(fusion_model.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    predictor = DDP(predictor.to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    ########################## train
    results = {
        'highest_valid': {"auc": -np.inf},
        'final_train': None,
        'final_test': None,
        'highest_valid_desc': None,
        "final_train_desc": None,
        "final_test_desc": None,
    }
    early_stop = 0
    patience = 100
    for epoch in range(args.start_epoch, args.epochs):
        tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(f"pdbbind-{args.split_type}", epoch)
        # [*] set sampler
        train_loader.sampler.set_epoch(epoch)

        loss = train_one_epoch(vision_model, gnn_model, fusion_model, predictor, optimizer=optimizer,
                               data_loader=train_loader, criterion=criterion, device=device,
                               epoch=epoch, lr_scheduler=None, args=args, is_ddp=True, get_y_pred=get_y_pred_func,
                               tqdm_desc=tqdm_train_desc)

        train_loss, train_results, train_desc_results = run_eval_cls(vision_model, gnn_model, fusion_model, predictor, train_loader, criterion, device, epoch, get_y_pred=get_y_pred_func, tqdm_desc=tqdm_eval_train_desc)
        valid_loss, valid_results, valid_desc_results = run_eval_cls(vision_model, gnn_model, fusion_model, predictor, valid_loader, criterion, device, epoch, get_y_pred=get_y_pred_func, tqdm_desc=tqdm_eval_val_desc)
        test_loss, test_results, test_desc_results = run_eval_cls(vision_model, gnn_model, fusion_model, predictor, test_loader, criterion, device, epoch, get_y_pred=get_y_pred_func, tqdm_desc=tqdm_eval_test_desc)

        if lr_scheduler is not None:
            lr_scheduler.step(valid_results["auc"])

        print_only_rank0(f"[result info] dataset: pdbbind-{args.split_type} epoch: {epoch} early_stop: {early_stop} | loss: {train_loss}; train: {train_results}; valid: {valid_results}; test: {test_results}")

        if is_left_better_right(valid_results["auc"], results["highest_valid"]["auc"], standard="max"):
            results['epoch'] = epoch
            results['loss'] = train_loss

            results['highest_valid'] = valid_results
            results['final_train'] = train_results
            results['final_test'] = test_results

            results['highest_valid_desc'] = valid_desc_results
            results['final_train_desc'] = train_desc_results
            results['final_test_desc'] = test_desc_results

            save_finetune_ckpt(vision_model, gnn_model, fusion_model, predictor, optimizer, round(train_loss, 4), epoch,
                               args.log_dir, "valid_best", lr_scheduler=lr_scheduler, result_dict=results)

            with open(f"{args.log_dir}/prediction_dict.pkl", "wb") as f:
                pickle.dump(results, f)

            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    del results["highest_valid_desc"], results["final_train_desc"], results["final_test_desc"]
    print_only_rank0(f"final results: {results}\n")


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许数据副本，提高数据加载的效率
    # [*] initialize some arguments
    args = parse_args()
    args.world_size = args.ngpus_per_node * args.nodes
    # [*] run with torch.multiprocessing
    mp.spawn(main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
