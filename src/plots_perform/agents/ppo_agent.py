import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.FEATURE_NUM = 128
        self.s_dim = [6, 8]
        self.a_dim = act_dim

        self.fc1_critic = nn.Linear(1, self.FEATURE_NUM)
        self.fc2_critic = nn.Linear(1, self.FEATURE_NUM)
        self.conv1_critic = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv2_critic = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv3_critic = nn.Linear(self.a_dim, self.FEATURE_NUM)
        self.fc3_critic = nn.Linear(1, self.FEATURE_NUM)
        self.fc4_critic = nn.Linear(self.FEATURE_NUM * self.s_dim[0], self.FEATURE_NUM)
        self.val_head = nn.Linear(self.FEATURE_NUM, 1)

        # actor
        self.fc1_actor = nn.Linear(1, self.FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, self.FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], self.FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.a_dim, self.FEATURE_NUM)
        self.fc3_actor = nn.Linear(1, self.FEATURE_NUM)
        self.fc4_actor = nn.Linear(self.FEATURE_NUM * self.s_dim[0], self.FEATURE_NUM)
        self.pi_head = nn.Linear(self.FEATURE_NUM, act_dim)

    def get_value(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        critic_split_0 = F.relu(self.fc1_critic(x[:, 0:1, -1]))
        critic_split_1 = F.relu(self.fc2_critic(x[:, 1:2, -1]))
        critic_split_2 = F.relu(self.conv1_critic(x[:, 2:3, :]).view(-1, self.FEATURE_NUM))
        critic_split_3 = F.relu(self.conv2_critic(x[:, 3:4, :]).view(-1, self.FEATURE_NUM))
        critic_split_4 = F.relu(self.conv3_critic(x[:, 4:5, : self.a_dim]).view(-1, self.FEATURE_NUM))
        critic_split_5 = F.relu(self.fc3_critic(x[:, 5:6, -1]))

        critic_merge_net = torch.cat([critic_split_0, critic_split_1, critic_split_2, critic_split_3, critic_split_4, critic_split_5], 1)

        value_net = F.relu(self.fc4_critic(critic_merge_net))
        value = self.val_head(value_net)

        return value

    def get_max_action_and_value(self, x, action=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        # Actor Net
        actor_split_0 = F.relu(self.fc1_actor(x[:, 0:1, -1]))
        actor_split_1 = F.relu(self.fc2_actor(x[:, 1:2, -1]))
        actor_split_2 = F.relu(self.conv1_actor(x[:, 2:3, :]).view(-1, self.FEATURE_NUM))
        actor_split_3 = F.relu(self.conv2_actor(x[:, 3:4, :]).view(-1, self.FEATURE_NUM))
        actor_split_4 = F.relu(self.conv3_actor(x[:, 4:5, : self.a_dim]).view(-1, self.FEATURE_NUM))
        actor_split_5 = F.relu(self.fc3_actor(x[:, 5:6, -1]))
        actor_merge_net = torch.cat([actor_split_0, actor_split_1, actor_split_2, actor_split_3, actor_split_4, actor_split_5], 1)
        pi_net = F.relu(self.fc4_actor(actor_merge_net))
        logits = self.pi_head(pi_net)

        # Critic Net
        value = self.get_value(x)

        probs = Categorical(logits=logits)
        if action is None:
            action = torch.argmax(logits, dim=-1)
        return action, probs.log_prob(action), probs.entropy(), value
