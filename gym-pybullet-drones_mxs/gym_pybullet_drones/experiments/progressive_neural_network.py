import torch
import torch.nn as nn
from gymnasium.spaces import Discrete, Box
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomPNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, base_columns, net_arch=None, activation_fn=nn.Tanh,
                 **kwargs):
        super(CustomPNNPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Save the base columns (frozen networks)
        self.base_columns = base_columns

        # Define the lateral connection layers
        self.lateral_layers = nn.ModuleList([
            nn.Linear(column.mlp_extractor.latent_dim_pi, column.mlp_extractor.latent_dim_pi)
            # Matching the output size of the base column
            for column in self.base_columns  # Create one lateral layer for each base column
        ])

        # Define default net architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])

        # Actor and critic networks
        actor_layers = net_arch["pi"]
        critic_layers = net_arch["vf"]

        # Input size for actor and critic networks
        combined_feature_dim = sum(
            column.mlp_extractor.latent_dim_pi for column in self.base_columns) + self.features_dim

        actor_layers.insert(0, combined_feature_dim)
        if isinstance(action_space, Discrete):
            actor_layers.append(action_space.n)
        elif isinstance(action_space, Box):
            actor_layers.append(action_space.shape[0])  # Assuming continuous action space

        critic_layers.insert(0, combined_feature_dim)
        critic_layers.append(1)  # Single value output for critic network

        self.actor_net = self.create_mlp(actor_layers, activation_fn)
        self.critic_net = self.create_mlp(critic_layers, activation_fn)

    def forward(self, obs, deterministic=False):
        # Extract features from the base columns
        base_features = [column.extract_features(obs) for column in self.base_columns]

        # Apply lateral layers to base features
        lateral_outputs = [layer(features) for layer, features in zip(self.lateral_layers, base_features)]

        # Combine lateral layer outputs with new features
        combined_features = torch.cat(lateral_outputs + [obs], dim=1)

        # Pass combined features through the policy and value networks
        actor_output = self.actor_net(combined_features)
        critic_output = self.critic_net(combined_features)

        return actor_output, critic_output

    def _predict(self, obs, deterministic=False):
        action_logits, _ = self.forward(obs, deterministic)
        return action_logits

    def create_mlp(self, layer_sizes, activation_fn):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_fn())
        return nn.Sequential(*layers)
