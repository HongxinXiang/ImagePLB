import os
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# initializing weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "linear":
        return nn.Identity()
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_classifier(arch, in_features, num_tasks, inner_dim=None, dropout=0.2, activation_fn=None):
    if arch == "arch1":
        return nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, in_features // 2)),
            ("Softplus", nn.Softplus()),
            ("linear2", nn.Linear(in_features // 2, num_tasks))
        ]))
    elif arch == "arch2":
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features, 128)),
            ('leakyreLU', nn.LeakyReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('linear2', nn.Linear(128, num_tasks))
        ]))
    elif arch == "arch3":
        return nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_features, num_tasks))
        ]))
    elif arch == "arch4":  # Referene: Uni-Mol: https://github.com/dptech-corp/Uni-Mol/blob/main/unimol/unimol/models/unimol.py#L316
        return nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(dropout)),
            ("linear1", nn.Linear(in_features, inner_dim)),
            ("activator", get_activation_fn(activation_fn)),
            ("dropout2", nn.Dropout(dropout)),
            ("linear2", nn.Linear(inner_dim, num_tasks))
        ]))


def save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict, desc, epoch, save_path, name_pre, name_post='_best', logger=None):
    log = print if logger is None else logger.info

    state = {
        'epoch': epoch,
        'desc': desc
    }

    if model_dict is not None:
        for key in model_dict.keys():
            model = model_dict[key]
            state[key] = {k: v.cpu() for k, v in model.state_dict().items()}
    if optimizer_dict is not None:
        for key in optimizer_dict.keys():
            optimizer = optimizer_dict[key]
            state[key] = optimizer.state_dict()
    if lr_scheduler_dict is not None:
        for key in lr_scheduler_dict.keys():
            lr_scheduler = lr_scheduler_dict[key]
            state[key] = lr_scheduler.state_dict()

    try:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            log("Directory {} is created.".format(save_path))
    except:
        pass

    filename = '{}/{}{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    log('model has been saved as {}'.format(filename))


def load_checkpoint(pretrained_pth, model_dict, logger=None):
    """

    :param pretrained_pth:
    :param model_dict:
    :param logger:
    :return:
    """

    log = logger.info if logger is not None else print
    flag = False
    resume_desc = None
    if os.path.isfile(pretrained_pth):
        pretrained_model = torch.load(pretrained_pth)
        resume_desc = pretrained_model["desc"]

        for model_key, model in model_dict.items():
            try:
                model.load_state_dict(pretrained_model[model_key])
            except:
                ckp_keys = list(pretrained_model[model_key])
                cur_keys = list(model.state_dict())
                model_sd = model.state_dict()
                for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                    model_sd[cur_key] = pretrained_model[model_key][ckp_key]
                model.load_state_dict(model_sd)
            log("[resume info] resume {} completed.".format(model_key))
        flag = True
    else:
        log("===> No checkpoint found at '{}'".format(pretrained_pth))

    return flag, resume_desc


def write_result_dict_to_tb(tb_writer: SummaryWriter, result_dict: dict, optimizer_dict: dict, show_epoch=True):
    loop = result_dict["epoch"] if show_epoch else result_dict["step"]
    for key in result_dict.keys():
        if key == "epoch" or key == "step":
            continue
        tb_writer.add_scalar(key, result_dict[key], loop)
    for key in optimizer_dict.keys():
        optimizer = optimizer_dict[key]
        tb_writer.add_scalar(key, optimizer.param_groups[0]["lr"], loop)


