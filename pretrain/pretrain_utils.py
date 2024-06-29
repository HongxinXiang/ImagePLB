import sys
from collections import defaultdict

import numpy as np
import torch
import torch_scatter
from tqdm import tqdm

from model.model_utils import write_result_dict_to_tb


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


def train_one_epoch_vfds_add_diff_loss(vision_model, graph_model, fusion_model,
                    nextMolPredictor, nextPocketPredictor, nextComplexPredictor,
                    optimizer, data_loader, criterionReg, device, epoch, weighted_loss=False,
                    lr_scheduler=None, tb_writer=None, args=None, logger=None, is_rank0=True,
                    lambda_next_mol=1, lambda_next_pocket=1, lambda_next_complex=1):
    criterionReg, criterionReg_none = criterionReg

    for model in [vision_model, graph_model, fusion_model, nextMolPredictor, nextPocketPredictor, nextComplexPredictor]:
        model.train()

    accu_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    train_dict = {}
    data_loader = tqdm(data_loader, total=len(data_loader))
    log_loss_dict = {"next_mol_loss": 0, "next_pocket_loss": 0, "next_complex_loss": 0,
                     "diff_mol": 0, "diff_pocket": 0, "diff_complex": 0}
    for step, data in enumerate(data_loader):
        loss_dict = {
            "next_mol_loss": 0,
            "next_pocket_loss": 0,
            "next_complex_loss": 0,

            "diff_mol": 0,
            "diff_pocket": 0,
            "diff_complex": 0
        }

        c_3d_image, n_3d_image = data["c_3d_image"].to(device), data["n_3d_image"].to(device)
        c_pocket_feats, c_pocket_coords, c_pocket_mask = data["c_pocket_feats"].to(device), data["c_pocket_coords"].to(device), data["c_pocket_mask"].to(device)
        n_pocket_feats, n_pocket_coords, n_pocket_mask = data["n_pocket_feats"].to(device), data["n_pocket_coords"].to(device), data["n_pocket_mask"].to(device)

        # extract image features
        n_samples, n_views, n_chanel, h, w = c_3d_image.shape
        c_3d_feats = vision_model(c_3d_image.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
        n_3d_feats = vision_model(n_3d_image.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
        c_3d_feats_mean, n_3d_feats_mean = c_3d_feats.mean(1), n_3d_feats.mean(1)

        # extract graph features
        c_feats_graph, c_coors_out = graph_model(c_pocket_feats, c_pocket_coords, mask=c_pocket_mask)
        c_global_feats = (c_feats_graph * c_pocket_mask.resize(c_pocket_mask.shape[0], c_pocket_mask.shape[1], 1)).sum(1) / torch.unsqueeze(c_pocket_mask.sum(1), dim=1)

        n_feats_graph, n_coors_out = graph_model(n_pocket_feats, n_pocket_coords, mask=n_pocket_mask)
        n_global_feats = (n_feats_graph * n_pocket_mask.resize(n_pocket_mask.shape[0], n_pocket_mask.shape[1], 1)).sum(1) / torch.unsqueeze(n_pocket_mask.sum(1), dim=1)

        # 交互
        batch_feat_3d = torch.arange(len(c_3d_feats_mean)).to(device)
        c_batch_feat_graph = torch.arange(len(c_global_feats)).repeat(c_feats_graph.shape[1], 1).T.flatten().to(device)[c_pocket_mask.flatten()]
        n_batch_feat_graph = torch.arange(len(n_global_feats)).repeat(c_feats_graph.shape[1], 1).T.flatten().to(device)[n_pocket_mask.flatten()]

        c_feat_3d_v_fusion_g, c_feat_3d_g_fusion_v = fusion_model(c_3d_feats_mean, batch_feat_3d, c_feats_graph[c_pocket_mask], c_batch_feat_graph)
        c_feat_3d_g_fusion_v_mean = torch_scatter.scatter_mean(c_feat_3d_g_fusion_v, c_batch_feat_graph, dim=0)

        n_feat_3d_v_fusion_g, n_feat_3d_g_fusion_v = fusion_model(n_3d_feats_mean, batch_feat_3d, n_feats_graph[n_pocket_mask], n_batch_feat_graph)
        n_feat_3d_g_fusion_v_mean = torch_scatter.scatter_mean(n_feat_3d_g_fusion_v, n_batch_feat_graph, dim=0)

        c_feats_complex = (c_feat_3d_v_fusion_g + c_feat_3d_g_fusion_v_mean) / 2
        n_feats_complex = (n_feat_3d_v_fusion_g + n_feat_3d_g_fusion_v_mean) / 2

        ################## loss
        loss_dict["next_mol_loss"] = criterionReg(nextMolPredictor(c_3d_feats_mean, condition=c_global_feats.detach()), n_3d_feats_mean)
        loss_dict["next_pocket_loss"] = criterionReg(nextPocketPredictor(c_global_feats, condition=c_3d_feats_mean.detach()), n_global_feats)
        loss_dict["next_complex_loss"] = criterionReg(nextComplexPredictor(c_feats_complex), n_feats_complex)

        # 差异性损失
        loss_dict["diff_mol"] = torch.exp(-criterionReg_none(c_3d_feats_mean, n_3d_feats_mean).mean(1)).mean()
        loss_dict["diff_pocket"] = torch.exp(-criterionReg_none(c_global_feats, n_global_feats).mean(1)).mean()
        loss_dict["diff_complex"] = torch.exp(-criterionReg_none(c_feats_complex, n_feats_complex).mean(1)).mean()

        # backward
        if weighted_loss:
            # 自动在帧预测和差异性之间权衡
            weight_L_mol = (loss_dict["next_mol_loss"] / (loss_dict["next_mol_loss"]+loss_dict["diff_mol"])).detach()
            weight_L_pocket = (loss_dict["next_pocket_loss"] / (loss_dict["next_pocket_loss"] + loss_dict["diff_pocket"])).detach()
            weight_L_complex = (loss_dict["next_complex_loss"] / (loss_dict["next_complex_loss"] + loss_dict["diff_complex"])).detach()

            loss = weight_L_mol * loss_dict["next_mol_loss"] + (1 - weight_L_mol) * loss_dict["diff_mol"] + \
                   weight_L_pocket * loss_dict["next_pocket_loss"] + (1 - weight_L_pocket) * loss_dict["diff_pocket"] + \
                   weight_L_complex * loss_dict["next_complex_loss"] + (1 - weight_L_complex) * loss_dict["diff_complex"]
        else:
            loss = lambda_next_mol * loss_dict["next_mol_loss"] + lambda_next_pocket * loss_dict["next_pocket_loss"] + lambda_next_complex * loss_dict["next_complex_loss"] + \
                   loss_dict["diff_mol"] + loss_dict["diff_pocket"] + loss_dict["diff_complex"]
        loss_non_weight = loss_dict["next_mol_loss"] + loss_dict["next_pocket_loss"] + loss_dict["next_complex_loss"] + loss_dict["diff_mol"] + loss_dict["diff_pocket"] + loss_dict["diff_complex"]

        loss.backward()

        # logger
        log_loss_dict["next_mol_loss"] += loss_dict["next_mol_loss"].item()
        log_loss_dict["next_pocket_loss"] += loss_dict["next_pocket_loss"].item()
        log_loss_dict["next_complex_loss"] += loss_dict["next_complex_loss"].item()

        log_loss_dict["diff_mol"] += loss_dict["diff_mol"].item()
        log_loss_dict["diff_pocket"] += loss_dict["diff_pocket"].item()
        log_loss_dict["diff_complex"] += loss_dict["diff_complex"].item()

        accu_loss += loss_non_weight.detach()

        data_loader.desc = "[train epoch {}] total loss: {:.3f}; mol loss: {:.3f}; pocket loss: {:.3f}; " \
                           "complex loss: {:.3f}; diff_mol: {:.3f}; diff_pocket: {:.3f}; diff_complex: {:.3f}"\
            .format(epoch, accu_loss.item() / (step + 1), log_loss_dict["next_mol_loss"] / (step + 1),
                    log_loss_dict["next_pocket_loss"] / (step + 1), log_loss_dict["next_complex_loss"] / (step + 1),
                    log_loss_dict["diff_mol"] / (step + 1), log_loss_dict["diff_pocket"] / (step + 1),
                    log_loss_dict["diff_complex"] / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if step % args.n_batch_step_optim == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_dict = {
            "step": (step + 1) + len(data_loader) * epoch,
            "epoch": epoch + (step + 1) / len(data_loader),
            "total_loss": accu_loss.item() / (step + 1),

            "next_mol_loss": log_loss_dict["next_mol_loss"] / (step + 1),
            "next_pocket_loss": log_loss_dict["next_pocket_loss"] / (step + 1),
            "next_complex_loss": log_loss_dict["next_complex_loss"] / (step + 1),

            "diff_mol": log_loss_dict["diff_mol"] / (step + 1),
            "diff_pocket": log_loss_dict["diff_pocket"] / (step + 1),
            "diff_complex": log_loss_dict["diff_complex"] / (step + 1),
        }

        if is_rank0 and step in np.arange(0, len(data_loader), args.tb_step_num).tolist():
            if tb_writer is not None:
                write_result_dict_to_tb(tb_writer, train_dict, optimizer_dict={"optimizer": optimizer}, show_epoch=False)

    # Update learning rates
    if lr_scheduler is not None:
        lr_scheduler.step()

    return train_dict

