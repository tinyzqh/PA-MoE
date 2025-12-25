from dataclasses import dataclass


@dataclass
class Config:
    exp_name: str = "MoE-More-LR-Sen"  # "MoEVSDMoE-every-update-distall-allloss-klweight0.025-addnoise0.001_linear"
    track: bool = True
    wandb_project_name: str = "MOE-VS-Finall-18-LR-Sen"
    seed: int = 2
    torch_deterministic: bool = True
    num_envs: int = 1
    envs_model: str = "change"  # ["normal", "change"]
    num_steps: int = 2000
    num_minibatches: int = 32
    total_timesteps: int = 2_000_000

    activation_name: str = "ReLU"

    policy_type: str = "dmoe"  # ["mlp", "moe", "smoe", "dmoe"]
    value_type: str = "dmoe"  # ["mlp", "moe", "smoe", "dmoe"]

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.99
    beta2: float = 0.99
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 5
    clip_coef: float = 0.2
    norm_adv: bool = True
    clip_vloss: bool = False
    vf_coef: float = 5
    ent_coef: float = 0.0

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    redo_tau: float = 0.5  # 0.5 for default
    gradient_tau: float = 0.05  # 0.05
