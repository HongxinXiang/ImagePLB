import torch
from torch import nn

from model.model_utils import weights_init, get_classifier


class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(Predictor, self).__init__()
        self.network = get_classifier(arch="arch1", in_features=in_features, num_tasks=out_features)

        self.apply(weights_init)

    def forward(self, x):
        logit = self.network(x)
        return logit


class NextComplexPredictor(nn.Module):
    def __init__(self, in_features, out_features):
        super(NextComplexPredictor, self).__init__()
        self.network = get_classifier(arch="arch4", in_features=in_features, num_tasks=out_features, inner_dim=in_features, activation_fn="softplus")
        self.apply(weights_init)

    def forward(self, x):
        logit = self.network(x)
        return logit


class NextPredictorWithCondition(nn.Module):
    def __init__(self, in_features, condition_dim):
        super(NextPredictorWithCondition, self).__init__()

        self.cond_fc = nn.Linear(condition_dim, in_features)
        self.network = get_classifier(arch="arch4", in_features=in_features*2, num_tasks=in_features, inner_dim=in_features*2, activation_fn="softplus")
        self.apply(weights_init)

    def forward(self, x, condition):
        feat_c = self.cond_fc(condition)
        logit = self.network(torch.hstack([x, feat_c]))
        return logit
