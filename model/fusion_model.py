import math

import torch
from torch import nn


class CrossModalityMultiHeadAttention(nn.Module):
    def __init__(self, n_dim1, n_dim2, num_of_attention_heads, hidden_size):
        super().__init__()
        assert hidden_size % num_of_attention_heads == 0, "The hidden size is not a multiple of the number of attention heads"

        self.n_dim1 = n_dim1
        self.n_dim2 = n_dim2

        self.num_attention_heads = num_of_attention_heads
        self.attention_head_size = int(hidden_size / num_of_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if n_dim1 != n_dim2:
            self.map = nn.Linear(n_dim1, n_dim2)

        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        self.key1 = nn.Linear(hidden_size, self.all_head_size)
        self.value1 = nn.Linear(hidden_size, self.all_head_size)

        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)

        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        # self.dense = nn.Identity()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2)

    def get_qkv(self, h, type="head"):
        if type == "head":
            mixed_query_layer = self.query1(h)  # [Batch_size x Node_length x Hidden_size]
            mixed_key_layer = self.key1(h)  # [Batch_size x Node_length x Hidden_size]
            mixed_value_layer = self.value1(h)  # [Batch_size x Node_length x Hidden_size]
        elif type == "tail":
            mixed_query_layer = self.query2(h)  # [Batch_size x Node_length x Hidden_size]
            mixed_key_layer = self.key2(h)  # [Batch_size x Node_length x Hidden_size]
            mixed_value_layer = self.value2(h)  # [Batch_size x Node_length x Hidden_size]
        else:
            raise NotImplementedError

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [Batch_size x Num_of_heads x Node_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Node_length x Head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [Batch_size x Num_of_heads x Node_length x Head_size]

        return query_layer, key_layer, value_layer

    def get_cross_attention_output(self, query_layer, cross_key_layer, value_layer, cross_value_layer, mask, type="head"):
        attention_scores = torch.matmul(query_layer, cross_key_layer.transpose(-1, -2))  # [Batch_size x Num_of_heads x Node_length x Node_length]
        attention_scores = attention_scores * mask
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [Batch_size x Num_of_heads x Node_length x Node_length]
        attention_scores[attention_scores == 0] = -9e10  # Softmax 时忽略0，即0经过softmax后还是0：在softmax之前，将0用极小负值替换。
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Node_length x Node_length]
        context_layer = value_layer + torch.matmul(attention_probs, cross_value_layer)  # [Batch_size x Num_of_heads x Node_length x Head_size]

        context_layer = context_layer.permute(1, 0, 2).contiguous()  # [Batch_size x Node_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:1] + (self.all_head_size,)  # [Batch_size x Node_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Node_length x Hidden_size]

        if type == "head":
            output = self.dense1(context_layer)
        elif type == "tail":
            output = self.dense2(context_layer)
        else:
            raise NotImplementedError

        return output

    def forward(self, h_head, batch_head, h_tail, batch_tail):
        if self.n_dim1 != self.n_dim2:
            h_head = self.map(h_head)

        query_layer_head, key_layer_head, value_layer_head = self.get_qkv(h_head, type="head")
        query_layer_tail, key_layer_tail, value_layer_tail = self.get_qkv(h_tail, type="tail")

        mask_head_tail = batch_head[:, None] == batch_tail[None, :]
        mask_head_tail = mask_head_tail.long()
        output_head = self.get_cross_attention_output(query_layer_head, key_layer_tail, value_layer_head, value_layer_tail, mask_head_tail, type="head")
        output_tail = self.get_cross_attention_output(query_layer_tail, key_layer_head, value_layer_tail, value_layer_head, mask_head_tail.T, type="tail")

        return output_head, output_tail

