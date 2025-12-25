import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.network.moe.moe import BasicMOE

FEATURE_NUM = 18


def register_hook(net, hook_fn):
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            register_hook(layer)
        else:
            # it's a non sequential. Register a hook
            layer.register_forward_hook(hook_fn)


class MoEPolicy(nn.Module):
    def __init__(self, envs, activation_name):
        super().__init__()
        self.state_space = envs.observation_space.shape
        self.action_dim = envs.action_space.n
        self.activation_name = activation_name

        o_dim = np.array(envs.observation_space.shape).prod()
        a_dim = np.prod(envs.action_space.n)

        self.moe = BasicMOE(o_dim, FEATURE_NUM, 3)
        self.last_layer = nn.Sequential(nn.Linear(FEATURE_NUM, a_dim))

        # Setup feature logging
        self.setup_feature_logging()

    def setup_feature_logging(self) -> None:
        """
        Input h_dim: A list describing the network architecture. Ex. [64, 64] describes a network with two hidden layers
                    of size 64 each.
        """
        self.to_log_features = False
        self.activations_expert1 = {}
        self.activations_expert2 = {}
        self.activations_expert3 = {}
        self.activations_router = []

        self.feature_keys = {}
        for i in range(len(self.moe.experts)):
            # for j in range(2):
            self.feature_keys.setdefault(i, [self.moe.experts[i].moe_net[j * 2 + 1] for j in range(2)])

        def hook_fn1(model, input, output):
            if self.to_log_features:
                self.activations_expert1[model] = output

        def hook_fn2(model, input, output):
            if self.to_log_features:
                self.activations_expert2[model] = output

        def hook_fn3(model, input, output):
            if self.to_log_features:
                self.activations_expert3[model] = output

        # Register "hook_fn" for each layer
        for idx, expert in enumerate(self.moe.experts):
            if idx == 0:
                register_hook(expert.moe_net, hook_fn1)
            elif idx == 1:
                register_hook(expert.moe_net, hook_fn2)
            elif idx == 2:
                register_hook(expert.moe_net, hook_fn3)
            else:
                raise ValueError("Not support {} experts".format(len(self.moe.experts)))

    def get_activations(self):
        return_dict = {}
        if len(self.activations_expert1) != 0:
            return_dict.setdefault("expert1", [self.activations_expert1[key] for key in self.feature_keys[0]])

        if len(self.activations_expert2) != 0:
            return_dict.setdefault("expert2", [self.activations_expert2[key] for key in self.feature_keys[1]])

        if len(self.activations_expert3) != 0:
            return_dict.setdefault("expert3", [self.activations_expert3[key] for key in self.feature_keys[2]])

        return_dict.setdefault("router", self.activations_router)
        return return_dict

    def forward(self, x, action=None):
        if len(x.size()) == 3:
            bs = x.size()[0]
            x = x.view(bs, -1)
        elif len(x.size()) == 4:
            bs, env_num = x.size()[0], x.size()[1]
            x = x.contiguous()
            x = x.view(bs, env_num, -1)
        else:
            raise ValueError("Length of x not Support!")

        self.to_log_features = True
        output, gating_output = self.moe(x)
        self.to_log_features = False
        self.activations_router = gating_output

        logits = self.last_layer(output)

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
