from dataclasses import dataclass


@dataclass
class ExperimentId(object):
    # Noise 0.001
    wandb_path = "tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18/runs"
    MLP_ID = ["zp3adp59", "f5jgubfr"]
    MOE_ID = ["vasq3c06", "yvmumzwq"]
    SMOE_ID = ["zaolho9o", "e43swc7v"]
    DMOE_ID = ["abpqrlny", "lohq03hi"]


@dataclass
class ExperimentId_2(object):
    # Noise 0.0005
    wandb_path = "tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-Plus-Noise-0.0005/runs"
    DMOE_ID = ["9qb83b0t", "kwx5dyzf"]


@dataclass
class ExperimentId_3(object):
    # Noise 0.005
    wandb_path = "tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-Plus-Noise-0.005/runs"
    DMOE_ID = ["adxsl0kl", "2ps6ys26"]


@dataclass
class ExperimentId_4(object):
    # Learning Rate 5e-5
    wandb_path = "tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-LR-Sen/runs"
    DMOE_ID = ["apxhtrf2", "8dl3l7bi"]


@dataclass
class ExperimentId_5(object):
    # Learning Rate 5e-3
    wandb_path = "tinyzqh-the-university-of-electro-communication/MOE-VS-Finall-18-LR-Sen/runs"
    DMOE_ID = ["12yfeap8", "e8actrlm"]
