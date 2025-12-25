import torch
import numpy as np
from scipy.linalg import svd


def _freeze_module(module: torch.nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False


def compute_matrix_rank_summaries(m: torch.Tensor, prop=0.99, use_scipy=False):
    """
    Computes the rank, effective rank, and approximate rank of a matrix
    Refer to the corresponding functions for their definitions
    :param m: (float np array) a rectangular matrix
    :param prop: (float) proportion used for computing the approximate rank
    :param use_scipy: (bool) indicates whether to compute the singular values in the cpu, only matters when using
                                  a gpu
    :return: (torch int32) rank, (torch float32) effective rank, (torch int32) approximate rank
    """
    if use_scipy:
        np_m = m.cpu().numpy()
        sv = torch.tensor(svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device)
    else:
        sv = torch.linalg.svdvals(m)  # for large matrices, svdvals may fail to converge in gpu, but not cpu
    rank = torch.count_nonzero(sv).to(torch.int32)
    effective_rank = compute_effective_rank(sv)
    approximate_rank = compute_approximate_rank(sv, prop=prop)
    approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)
    return rank, effective_rank, approximate_rank, approximate_rank_abs


def compute_effective_rank(sv: torch.Tensor):
    """
    Computes the effective rank as defined in this paper: https://ieeexplore.ieee.org/document/7098875/
    When computing the shannon entropy, 0 * log 0 is defined as 0
    :param sv: (float torch Tensor) an array of singular values
    :return: (float torch Tensor) the effective rank
    """
    norm_sv = sv / torch.sum(torch.abs(sv))
    entropy = torch.tensor(0.0, dtype=torch.float32, device=sv.device)
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * torch.log(p)

    effective_rank = torch.tensor(np.e) ** entropy
    return effective_rank.to(torch.float32)


def compute_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv ** 2
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))  # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_abs_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper, just that we won't be squaring the singular values
    https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))  # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)
