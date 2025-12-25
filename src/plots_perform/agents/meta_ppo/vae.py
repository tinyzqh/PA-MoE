import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        bs = x.size()[0]
        x = x.view(bs, -1)
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, z):
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        logits = self.fc_output(h)
        return logits


class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        logits = self.decoder(z)
        return logits, mean, log_var
