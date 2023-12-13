import torch
import random
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from .RevIN import RevIN


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def generate_bert_mask(B, T, p=0.15):
    masked_indices = torch.bernoulli(torch.full((B, T), p)).bool()
    indices_replaced = torch.bernoulli(torch.full((B, T), 0.8)).bool() & masked_indices
    # prob_data = torch.bernoulli(torch.full((B, T), 0.5)).bool()
    indices_random = torch.bernoulli(torch.full((B, T), 0.5)).bool() & masked_indices & ~indices_replaced
    # indices_saved = masked_indices & ~indices_replaced & ~indices_random
    return masked_indices, indices_replaced, indices_random


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', use_revin=True):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.use_revin = use_revin

        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.projection = nn.Linear(output_dims, input_dims)

        if self.use_revin:
            self.revin_layer = RevIN(input_dims)
        
    def forward(self, x, mask_mode=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0

        if self.use_revin:
            x = self.revin_layer(x, 'norm')

        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask_mode is None:
            if self.training:
                mask_mode = self.mask_mode
            else:
                mask_mode = 'all_true'
        
        if mask_mode == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_mode == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_mode == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask_mode == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask_mode == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        elif mask_mode == 'bert':
            mask, indices_replaced, indices_random = generate_bert_mask(x.size(0), x.size(1))
            mask = ~mask.to(x.device)
            indices_replaced = indices_replaced.to(x.device)
            indices_random = indices_random.to(x.device)
            # indices_saved = indices_saved.to(x.device)

        if mask_mode == 'bert':
            x[indices_replaced] = 0
            random_vector = torch.randint(len(x), x.shape, dtype=torch.float).to(x.device)
            x[indices_random] = random_vector[indices_random]
        else:
            mask &= nan_mask
            x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B(batch) x Ch(channel) x T(time)
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T, torch.Size([8, 320, 528])
        # print("xshape\n", x.shape)
        x = x.transpose(1, 2)  # B x T x Co

        enc_out = self.projection(x.detach())

        if self.use_revin:
            enc_out = self.revin_layer(enc_out, 'denorm')

        return x, enc_out
        