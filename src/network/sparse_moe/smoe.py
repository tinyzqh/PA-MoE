import torch
import torch.nn as nn
from torch.nn import functional as F


from src.network.moe.moe import BasicExpert


# noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, feature_in, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(feature_in, num_experts)
        self.noise_linear = nn.Linear(feature_in, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, feature_in, feature_out, expert_number, top_k, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.feature_out = feature_out
        self.router = NoisyTopkRouter(feature_in, expert_number, top_k)
        self.experts = nn.ModuleList([BasicExpert(feature_in, feature_out) for _ in range(expert_number)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.expert_number = expert_number

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, _ = x.shape
        gating_output, indices = self.router(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        updates = torch.zeros((batch_size, self.feature_out))

        for i, expert in enumerate(self.experts):
            # 找出应该由这个专家处理的样本

            expert_mask = (indices == i).any(dim=-1)  # some expert been selected
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)

            if selected_indices.numel() > 0:
                expert_input = flat_x[selected_indices]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[selected_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                updates.index_add_(0, selected_indices, weighted_output)

        return updates, gating_output
