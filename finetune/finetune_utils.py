import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch_scatter
from scipy import stats
from torch.nn.functional import mse_loss
from tqdm import tqdm

from utils.metrics import metric


def log_all(log_loss_dict, loss_dict):
    for key_1 in loss_dict.keys():
        for key_2 in loss_dict[key_1].keys():
            for key_3 in loss_dict[key_1][key_2].keys():
                if key_1 not in log_loss_dict.keys():
                    log_loss_dict[key_1] = {}
                if key_2 not in log_loss_dict[key_1].keys():
                    log_loss_dict[key_1][key_2] = {}

                if isinstance(loss_dict[key_1][key_2][key_3], torch.Tensor):
                    if key_3 not in log_loss_dict[key_1][key_2].keys():
                        log_loss_dict[key_1][key_2][key_3] = 0
                    log_loss_dict[key_1][key_2][key_3] += loss_dict[key_1][key_2][key_3].item()
                else:
                    if key_3 not in log_loss_dict[key_1][key_2].keys():
                        log_loss_dict[key_1][key_2][key_3] = defaultdict(int)
                    for key_4 in loss_dict[key_1][key_2][key_3].keys():
                        log_loss_dict[key_1][key_2][key_3][key_4] += loss_dict[key_1][key_2][key_3][key_4].item()
    log_loss_dict["mol"]["all"] = np.sum([log_loss_dict["mol"][key]["total"] for key in log_loss_dict["mol"].keys() if isinstance(log_loss_dict["mol"][key], dict)])
    log_loss_dict["ligand"]["all"] = np.sum([log_loss_dict["ligand"][key]["total"] for key in log_loss_dict["ligand"].keys() if isinstance(log_loss_dict["ligand"][key], dict)])
    log_loss_dict["receptor"]["all"] = np.sum([log_loss_dict["receptor"][key]["total"] for key in log_loss_dict["receptor"].keys() if isinstance(log_loss_dict["receptor"][key], dict)])


def get_y_pred_func(vision_model, graph_model, fusion_model, predictor, video_3d, pocket_feats, pocket_coords, mask, device):
    n_samples, n_views, n_chanel, h, w = video_3d.shape
    feats_3d = vision_model(video_3d.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
    feats_3d_mean = feats_3d.mean(1)

    feats_graph, coors_out = graph_model(pocket_feats, pocket_coords, mask=mask)
    global_feats = (feats_graph * mask.reshape(mask.shape[0], mask.shape[1], 1)).sum(1) / torch.unsqueeze(mask.sum(1),
                                                                                                         dim=1)
    # 交互
    batch_feat_3d = torch.arange(len(feats_3d_mean)).to(device)
    batch_feat_graph = torch.arange(len(feats_graph)).repeat(feats_graph.shape[1], 1).T.flatten().to(device)[
        mask.flatten()]

    feat_3d_v_fusion_g, feat_3d_g_fusion_v = fusion_model(feats_3d_mean, batch_feat_3d, feats_graph[mask],
                                                          batch_feat_graph)
    feat_3d_g_fusion_v_mean = torch_scatter.scatter_mean(feat_3d_g_fusion_v, batch_feat_graph, dim=0)

    # prediction: 预测可以使用 (feat_3d_v_fusion_g, feat_3d_g_fusion_v_mean) 中的任意组合
    y_pred = predictor(feat_3d_v_fusion_g + feat_3d_g_fusion_v_mean)
    return y_pred


def train_one_epoch(vision_model, graph_model, fusion_model, predictor,
                    optimizer, data_loader, criterion, device, epoch, get_y_pred=get_y_pred_func,
                    lr_scheduler=None, args=None, logger=None, is_rank0=True, is_ddp=False, tqdm_desc="train"):
    for model in [vision_model, graph_model, fusion_model, predictor]:
        model.train()

    accu_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, total=len(data_loader))
    for step, data in enumerate(data_loader):
        video_3d = data["3d_video"].to(device)
        pocket_feats, pocket_coords, mask = data["pocket_feats"].to(device), data["pocket_coords"].to(device), data["mask"].to(device)
        affinities = data["affinities"].to(device)

        y_pred = get_y_pred(vision_model, graph_model, fusion_model, predictor, video_3d, pocket_feats, pocket_coords, mask, device)

        loss = criterion(y_pred, affinities.view(y_pred.shape).float())
        loss.backward()

        # # logger
        accu_loss += loss.detach()
        data_loader.desc = "{}; total loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if step % args.n_batch_step_optim == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update learning rates
        if lr_scheduler is not None:
            lr_scheduler.step()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def run_eval_reg(vision_model, graph_model, fusion_model, predictor, data_loader, criterion, device, epoch, get_y_pred=get_y_pred_func, tqdm_desc="evaluation"):

    for model in [vision_model, graph_model, fusion_model, predictor]:
        model.eval()

    accu_loss = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, total=len(data_loader))

    y_true_list, y_pred_list = [], []
    for step, data in enumerate(data_loader):
        video_3d = data["3d_video"].to(device)
        pocket_feats, pocket_coords, mask = data["pocket_feats"].to(device), data["pocket_coords"].to(device), data["mask"].to(device)
        affinities = data["affinities"].to(device)

        y_pred = get_y_pred(vision_model, graph_model, fusion_model, predictor, video_3d, pocket_feats, pocket_coords, mask, device)
        loss = criterion(y_pred, affinities.view(y_pred.shape).float())

        # # logger
        accu_loss += loss.detach()
        data_loader.desc = "{}; total loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

        y_true_list.append(affinities.view(y_pred.shape).cpu())
        y_pred_list.append(y_pred.cpu())

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    RMSE = mse_loss(y_pred, y_true).item() ** 0.5
    spearman = stats.spearmanr(y_pred.cpu().numpy(), y_true.numpy())[0]
    pearson = stats.pearsonr(y_pred.cpu().numpy().flatten(), y_true.numpy().flatten())[0]

    result_dict = {
        "spearman": spearman, "pearson": pearson, "RMSE": RMSE
    }
    result_desc_dict = {
        "loss": accu_loss.item() / (step + 1), "spearman": spearman, "pearson": pearson, "RMSE": RMSE,
        "desc": {"y_true": y_true, "y_pred": y_pred}
    }

    return accu_loss.item() / (step + 1), result_dict, result_desc_dict


@torch.no_grad()
def run_eval_cls(vision_model, graph_model, fusion_model, predictor, data_loader, criterion, device, epoch, get_y_pred=get_y_pred_func, tqdm_desc="evaluation"):

    for model in [vision_model, graph_model, fusion_model, predictor]:
        model.eval()

    accu_loss = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, total=len(data_loader))

    y_true_list, y_pred_list = [], []
    for step, data in enumerate(data_loader):
        video_3d = data["3d_video"].to(device)
        pocket_feats, pocket_coords, mask = data["pocket_feats"].to(device), data["pocket_coords"].to(device), data["mask"].to(device)
        affinities = data["affinities"].to(device)

        y_pred = get_y_pred(vision_model, graph_model, fusion_model, predictor, video_3d, pocket_feats, pocket_coords, mask, device)
        loss = criterion(y_pred, affinities.view(y_pred.shape).float())

        # # logger
        accu_loss += loss.detach()
        data_loader.desc = "{}; total loss: {:.3f}".format(tqdm_desc, accu_loss.item() / (step + 1))

        y_true_list.append(affinities.view(y_pred.shape).cpu())
        y_pred_list.append(y_pred.cpu())

    y_true = torch.cat(y_true_list, dim=0)
    y_logit = torch.cat(y_pred_list, dim=0)
    y_prob = torch.sigmoid(y_logit)

    y_pred = torch.ones_like(y_true)
    y_pred[y_logit < 0.5] = 0

    result_desc_dict = metric(y_true, y_pred, y_prob, empty=-1)
    result_desc_dict["loss"] = accu_loss.item() / (step + 1)
    result_desc_dict["desc"] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    result_dict = {
        "acc": result_desc_dict["accuracy"], "auc": result_desc_dict["ROCAUC"], "aupr": result_desc_dict["AUPR"],
    }

    return accu_loss.item() / (step + 1), result_dict, result_desc_dict


# save checkpoint
def save_finetune_ckpt(vision_model, gnn_model, fusion_model, predictor, optimizer, loss, epoch, save_path, filename_pre,
                       lr_scheduler=None, result_dict=None, logger=None):
    log = logger.info if logger is not None else print
    vision_model_cpu = {k: v.cpu() for k, v in vision_model.state_dict().items()} if vision_model is not None else None
    gnn_model_cpu = {k: v.cpu() for k, v in gnn_model.state_dict().items()} if gnn_model is not None else None
    fusion_model_cpu = {k: v.cpu() for k, v in fusion_model.state_dict().items()} if fusion_model is not None else None
    predictor_cpu = {k: v.cpu() for k, v in predictor.state_dict().items()} if predictor is not None else None
    lr_scheduler = None if lr_scheduler is None else lr_scheduler.state_dict()
    state = {
            'epoch': epoch,
            'vision_model': vision_model_cpu,
            'gnn_model': gnn_model_cpu,
            'fusion_model': fusion_model_cpu,
            'predictor': predictor_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            'loss': loss,
            'result_dict': result_dict
        }
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        log("Directory {} is created.".format(save_path))

    filename = '{}/{}.pth'.format(save_path, filename_pre)
    torch.save(state, filename)
    log('model has been saved as {}'.format(filename))

