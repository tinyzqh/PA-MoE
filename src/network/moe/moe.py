import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicExpert(nn.Module):
    # 一个 Expert 可以是一个最简单的， linear 层即可
    # 也可以是 MLP 层
    # 也可以是 更复杂的 MLP 层（active function 设置为 swiglu）
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.moe_net = nn.Sequential(nn.Linear(feature_in, feature_out), nn.ReLU(), nn.Linear(feature_out, feature_out), nn.ReLU())

    def forward(self, x):
        return self.moe_net(x)


class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_number):
        super().__init__()
        self.experts = nn.ModuleList([BasicExpert(feature_in, feature_out) for _ in range(expert_number)])
        # gate 就是选一个 expert
        self.gate = nn.Linear(feature_in, expert_number)

    def forward(self, x):
        # x 的 shape 是 （batch, feature_in)
        expert_weight = self.gate(x)  # shape 是 (batch, expert_number)
        expert_weight = F.softmax(expert_weight, dim=-1)
        expert_out_list = [expert(x).unsqueeze(1) for expert in self.experts]  # 里面每一个元素的 shape 是： (batch, ) ??

        # concat 起来 (batch, expert_number, feature_out)
        expert_output = torch.cat(expert_out_list, dim=1)

        # print(expert_output.size())

        expert_weight_ = expert_weight.unsqueeze(1)  # (batch, 1, expert_nuber)

        # expert_weight * expert_out_list
        output = expert_weight_ @ expert_output  # (batch, 1, feature_out)

        return output.squeeze(1), expert_weight
