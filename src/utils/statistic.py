import torch
import torch.nn as nn

from src.network.moe.moe import BasicMOE
from src.network.sparse_moe.smoe import SparseMoE
from src.network.distill_moe.dmoe import DistillMoE
from src.utils.network_utils import compute_matrix_rank_summaries


@torch.inference_mode()
def get_redo_masks(net_output: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """

    score = net_output.abs().mean(dim=0)

    # Divide by activation mean to make the threshold independent of the layer size
    # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
    # https://github.com/google/dopamine/issues/209
    normalized_score = score / (score.mean() + 1e-9)

    layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
    if tau > 0.0:
        layer_mask[normalized_score <= tau] = 1
    else:
        layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
    return layer_mask


def WeightMagnitudeStatistic(model, model_name):
    if isinstance(model, nn.Sequential):
        weight_logs = {}
        for layer_idx in range((len(model) + 1) // 2):
            assert isinstance(model[2 * layer_idx], nn.Linear), "Only support nn.Linear layer!"
            weight_logs.setdefault("Weights/{}_{}_layer_mean".format(model_name, layer_idx), model[2 * layer_idx].weight.data.abs().mean().item())
    elif isinstance(model, BasicMOE) or isinstance(model, SparseMoE):
        weight_logs = {}
        for i, expert in enumerate(model.experts):
            for layer_idx in range((len(expert.moe_net) + 1) // 2):
                assert isinstance(expert.moe_net[2 * layer_idx], nn.Linear), "Only support nn.Linear layer!"
                weight_logs.setdefault("Weights/expert_{}_{}_{}_layer_mean".format(i, model_name, layer_idx), expert.moe_net[2 * layer_idx].weight.data.abs().mean().item())
    elif isinstance(model, DistillMoE):
        weight_logs = {}
        for i, expert in enumerate(model.experts):
            for layer_idx in range((len(expert.moe_net) + 1) // 2):
                assert isinstance(expert.moe_net[2 * layer_idx], nn.Linear), "Only support nn.Linear layer!"
                weight_logs.setdefault("Weights/expert_{}_{}_{}_layer_mean".format(i, model_name, layer_idx), expert.moe_net[2 * layer_idx].weight.data.abs().mean().item())

        # for layer_idx in range((len(model.student.moe_net) + 1) // 2):
        #     assert isinstance(model.student.moe_net[2 * layer_idx], nn.Linear), "Only support nn.Linear layer!"
        #     weight_logs.setdefault("Weights/student_{}_{}_layer_mean".format(model_name, layer_idx), model.student.moe_net[2 * layer_idx].weight.data.abs().mean().item())

    else:
        raise ValueError("Not Support Type {}".format(type(model)))
    return weight_logs


def RankStatistic(short_term_feature_activity):
    rank_logs = {}
    for key, value in short_term_feature_activity.items():
        if key == "policy_router" or key == "value_router":
            continue
        if value.size()[0] == 0:
            continue
        rank, eff_rank, approx_rank, abs_approx_rank = compute_matrix_rank_summaries(value)
        rank_logs.setdefault("Ranks/{}_{}".format(key, "rank"), rank.item())
        rank_logs.setdefault("Ranks/{}_{}".format(key, "eff_rank"), eff_rank.item())
        rank_logs.setdefault("Ranks/{}_{}".format(key, "approx_rank"), approx_rank.item())
        rank_logs.setdefault("Ranks/{}_{}".format(key, "abs_approx_rank"), abs_approx_rank.item())
    return rank_logs


def DormantStatistic(short_term_feature_activity, redo_tau):
    dormant_logs = {}
    all_zero_count, all_dorman_count, all_output = 0, 0, 0
    for key, value in short_term_feature_activity.items():
        if key == "policy_router" or key == "value_router":
            continue
        if value.size()[0] == 0:
            continue
        # Masks for tau=0 logging.
        zero_masks = get_redo_masks(net_output=value, tau=0)
        total_neurons = torch.numel(zero_masks)
        zero_count = torch.sum(zero_masks).item()
        zero_fraction = (zero_count / total_neurons) * 100
        # dormant_logs.setdefault("Dorman/{}_{}".format(key, "zero_count"), zero_count)
        dormant_logs.setdefault("Dorman/{}_{}".format(key, "zero_fraction"), zero_fraction)

        # Masks for tau=cfg.redo_tau logging.
        masks = get_redo_masks(net_output=value, tau=redo_tau)
        dormant_count = torch.sum(masks).item()
        dormant_fraction = (dormant_count / torch.numel(masks)) * 100
        # dormant_logs.setdefault("Dorman/{}_{}".format(key, "{}_count".format(redo_tau)), dormant_count)
        dormant_logs.setdefault("Dorman/{}_{}".format(key, "{}_fraction".format(redo_tau)), dormant_fraction)

        # Record All Info.
        all_zero_count += zero_count
        all_dorman_count += dormant_count
        all_output += total_neurons

    dormant_logs.setdefault("Dorman/total_zero_count", all_zero_count)
    dormant_logs.setdefault("Dorman/total_zero_fraction", (all_zero_count / all_output) * 100)
    dormant_logs.setdefault("Dorman/total_dorman_count", all_dorman_count)
    dormant_logs.setdefault("Dorman/total_dorman_fraction", (all_dorman_count / all_output) * 100)
    return dormant_logs


def RouterStatistic(short_term_feature_activity):
    router_logs = {}
    for key, value in short_term_feature_activity.items():
        if key == "policy_router" or key == "value_router":
            for i in range(value.size(1)):
                router_logs.setdefault("expert_{}_{}".format(i, key), float(sum(value[:, i])) / value.size(0))
    return router_logs


def OverleapDormantStatistic(last_time_short_term_feature_activity, short_term_feature_activity, redo_tau):
    if len(last_time_short_term_feature_activity) != 0:
        overleap_logs = {}
        for key, value in short_term_feature_activity.items():
            zero_masks = get_redo_masks(net_output=value, tau=0)
            last_time_zero_masks = get_redo_masks(net_output=last_time_short_term_feature_activity[key], tau=0)
            both_true = zero_masks & last_time_zero_masks
            overleap_logs.setdefault("Overleap/{}_dormant_zero_cof".format(key), both_true.sum().item() / both_true.numel())

            tau_masks = get_redo_masks(net_output=value, tau=redo_tau)
            last_time_tau_masks = get_redo_masks(net_output=last_time_short_term_feature_activity[key], tau=redo_tau)
            both_true = tau_masks & last_time_tau_masks
            overleap_logs.setdefault("Overleap/{}_dormant_tau_cof".format(key), both_true.sum().item() / both_true.numel())
        return overleap_logs
    else:
        return {}


def get_gradients(model, batch_obs):
    assert isinstance(model, nn.Sequential), "Only support nn.Sequential model!"
    ## --------- Clear Gradient --------- ##
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight"):
            if module.weight.grad is not None:
                module.weight.grad.zero_()  # 清空权重梯度
            if module.bias is not None and module.bias.grad is not None:
                module.bias.grad.zero_()  # 清空偏置梯度

    ## --------- Backward --------- ##
    bs, env_num = batch_obs.size()[0], batch_obs.size()[1]
    batch_obs = batch_obs.contiguous()
    batch_obs = batch_obs.view(bs, env_num, -1)
    batch_obs.requires_grad = True
    output = model(batch_obs)
    output.sum().backward()  # Compute gradients for all outputs

    ## --------- Record Gradient --------- ##
    gradients = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight"):
            if module.weight.grad is not None:
                # Compute the mean absolute gradient for each neuron
                gradients[str(int(int(name) / 2))] = module.weight.grad.abs().mean(dim=1)
    return gradients


def GradientInfoStatistic(model, batch_obs, model_name, threshold):
    gradients = get_gradients(model, batch_obs)
    logs = {}
    for k, v in gradients.items():
        logs.setdefault("regularization/{}_{}_weight_mean".format(model_name, k), torch.mean(v).item())
        logs.setdefault("regularization/{}_{}_weight_std".format(model_name, k), torch.std(v).item() if v.numel() > 1 else v.item())
        mask = v.abs() < threshold  # 掩盖绝对值小于阈值的梯度
        logs.setdefault("regularization/{}_{}_weight_mask_count".format(model_name, k), torch.sum(mask).item())
        logs.setdefault("regularization/{}_{}_weight_mask_fraction".format(model_name, k), torch.sum(mask).item() / mask.numel())
    return logs


def OverleapGradientStatistic(previous_model, model, batch_obs, model_name, threshold):
    if previous_model != None:
        previous_gradients = get_gradients(previous_model, batch_obs)
        gradients = get_gradients(model, batch_obs)
        overleap_logs = {}
        for k, v in gradients.items():
            mask = v.abs() < threshold  # 掩盖绝对值小于阈值的梯度
            previous_mask = previous_gradients[k].abs() < threshold
            both_true = mask & previous_mask
            overleap_logs.setdefault("Overleap/{}_{}_gradient_zero_overleap".format(model_name, k), both_true.sum().item() / both_true.numel())
        return overleap_logs
    else:
        return {}


def LosslandGradientStatistic(model, model_name):
    gradients = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight"):
            if module.weight.grad is not None:
                # Compute the mean absolute gradient for each neuron
                gradient = module.weight.grad.abs().mean(dim=1)
                normalized_gradient = gradient / gradient.norm()
                gradients.setdefault("{}_{}".format(model_name, name), normalized_gradient.unsqueeze(0))
    return gradients
