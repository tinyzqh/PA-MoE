import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.FEATURE_NUM = 128
        self.s_dim = [6, 8]
        self.a_dim = 6

        self.fc1_router = nn.Linear(1, self.FEATURE_NUM)
        self.fc2_router = nn.Linear(1, self.FEATURE_NUM)
        self.conv1_router = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv2_router = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv3_router = nn.Linear(self.a_dim, self.FEATURE_NUM)
        self.fc3_router = nn.Linear(1, self.FEATURE_NUM)
        self.fc4_router = nn.Linear(self.FEATURE_NUM * self.s_dim[0], self.FEATURE_NUM)
        self.topkroute_linear = nn.Linear(self.FEATURE_NUM, num_experts)
        self.noise_linear = nn.Linear(self.FEATURE_NUM, num_experts)

        # ---- 用于伪噪声的固定随机投影向量 ----
        self.register_buffer("pseudo_proj", torch.randn(self.FEATURE_NUM, 1))

    def forward(self, x):
        router_split_0 = F.relu(self.fc1_router(x[:, 0:1, -1]))
        router_split_1 = F.relu(self.fc2_router(x[:, 1:2, -1]))
        router_split_2 = F.relu(self.conv1_router(x[:, 2:3, :]).view(-1, self.FEATURE_NUM))
        router_split_3 = F.relu(self.conv2_router(x[:, 3:4, :]).view(-1, self.FEATURE_NUM))
        router_split_4 = F.relu(self.conv3_router(x[:, 4:5, : self.a_dim]).view(-1, self.FEATURE_NUM))
        router_split_5 = F.relu(self.fc3_router(x[:, 5:6, -1]))
        router_merge_net = torch.cat([router_split_0, router_split_1, router_split_2, router_split_3, router_split_4, router_split_5], 1)
        mh_output = self.fc4_router(router_merge_net)
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        # noise = torch.randn_like(logits) * F.softplus(noise_logits)

        # ---- 改动：伪噪声替代随机噪声 ----
        # 对相同输入 x，伪噪声是确定的（保证 PPO ratio 稳定）
        pseudo_phase = torch.matmul(mh_output, self.pseudo_proj)  # [batch, 1]
        pseudo_noise = torch.sin(pseudo_phase).expand_as(logits)  # broadcast
        noise = pseudo_noise * F.softplus(noise_logits)

        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class BasicExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.FEATURE_NUM = 128
        self.s_dim = [6, 8]
        self.a_dim = 6

        self.fc1_expert = nn.Linear(1, self.FEATURE_NUM)
        self.fc2_expert = nn.Linear(1, self.FEATURE_NUM)
        self.conv1_expert = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv2_expert = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv3_expert = nn.Linear(self.a_dim, self.FEATURE_NUM)
        self.fc3_expert = nn.Linear(1, self.FEATURE_NUM)
        self.fc4_expert = nn.Linear(self.FEATURE_NUM * self.s_dim[0], self.FEATURE_NUM)

    def forward(self, x):
        expert_split_0 = F.relu(self.fc1_expert(x[:, 0:1, -1]))
        expert_split_1 = F.relu(self.fc2_expert(x[:, 1:2, -1]))
        expert_split_2 = F.relu(self.conv1_expert(x[:, 2:3, :]).view(-1, self.FEATURE_NUM))
        expert_split_3 = F.relu(self.conv2_expert(x[:, 3:4, :]).view(-1, self.FEATURE_NUM))
        expert_split_4 = F.relu(self.conv3_expert(x[:, 4:5, : self.a_dim]).view(-1, self.FEATURE_NUM))
        expert_split_5 = F.relu(self.fc3_expert(x[:, 5:6, -1]))
        expert_merge_net = torch.cat([expert_split_0, expert_split_1, expert_split_2, expert_split_3, expert_split_4, expert_split_5], 1)
        mh_output = self.fc4_expert(expert_merge_net)
        return mh_output


class Expert(BasicExpert):
    def __init__(self):
        super().__init__()

    def add_noise(self, std=0.000000000005):
        """
        给神经网络的参数添加高斯噪声

        Args:
            std (float): 高斯噪声的标准差, 默认为0.01
        """
        with torch.no_grad():  # 不计算梯度
            for param in self.parameters():
                # 生成与参数相同形状的高斯噪声
                noise = torch.randn_like(param) * std
                # 将噪声添加到参数中
                param.add_(noise)


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.FEATURE_NUM = 128
        self.s_dim = [6, 8]
        self.a_dim = act_dim
        self._entropy_weight = 0.015  # np.log(action_dim)
        self.H_target = 0.1
        self.num_experts = 3

        self.actor_router = NoisyTopkRouter(num_experts=3, top_k=1)
        self.critic_router = NoisyTopkRouter(num_experts=3, top_k=1)
        self.actor_experts = nn.ModuleList([Expert() for _ in range(self.num_experts)])
        self.critic_experts = nn.ModuleList([Expert() for _ in range(self.num_experts)])

        self.val_head = nn.Linear(self.FEATURE_NUM, 1)

        self.pi_head = nn.Linear(self.FEATURE_NUM, act_dim)

    def get_value(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        batch_size, _, _ = x.shape
        flat_x = x

        gating_output_critic, indices_critic = self.critic_router(x)
        flat_gating_critic_output = gating_output_critic.view(-1, gating_output_critic.size(-1))

        critic_updates = torch.zeros((batch_size, self.FEATURE_NUM)).to(x.device)
        for i, expert in enumerate(self.critic_experts):
            # 找出应该由这个专家处理的样本
            expert_mask = (indices_critic == i).any(dim=-1)  # some expert been selected
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            # limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if selected_indices.numel() > 0:
                expert_input = flat_x[selected_indices]

                expert_output = expert(expert_input)

                gating_scores = flat_gating_critic_output[selected_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                critic_updates.index_add_(0, selected_indices, weighted_output)

        value = self.val_head(critic_updates)

        return value

    def get_max_action_and_value(self, x, action=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        batch_size, _, _ = x.shape

        flat_x = x

        gating_output_actor, indices_actor = self.actor_router(x)
        flat_gating_actor_output = gating_output_actor.view(-1, gating_output_actor.size(-1))

        actor_updates = torch.zeros((batch_size, self.FEATURE_NUM)).to(x.device)

        for i, expert in enumerate(self.actor_experts):
            # 找出应该由这个专家处理的样本
            expert_mask = (indices_actor == i).any(dim=-1)  # some expert been selected
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            # limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if selected_indices.numel() > 0:
                expert_input = flat_x[selected_indices]

                expert_output = expert(expert_input)

                gating_scores = flat_gating_actor_output[selected_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                actor_updates.index_add_(0, selected_indices, weighted_output)

        logits = self.pi_head(actor_updates)

        value = self.get_value(x)

        probs = Categorical(logits=logits)
        if action is None:
            action = torch.argmax(logits, dim=-1)
        return action, probs.log_prob(action), probs.entropy(), value

    # def add_noise(self):
    #     for expert in self.actor_experts:
    #         expert.add_noise()

    #     for expert in self.critic_experts:
    #         expert.add_noise()
