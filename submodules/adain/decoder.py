
import torch
import torch.nn as nn
import tinycudann as tcnn


class decoder_single_128(nn.Module):
    def __init__(
        self,
        ) -> None:
        super().__init__()

        self.color_net = tcnn.Network(
            n_input_dims=128,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 128,
                "n_hidden_layers": 1,
            },
            )
        

    def forward(self, input):
        out = self.color_net(input)
        return out


class VGG_linear(nn.Module):
    def __init__(
        self,
        ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(3,3),
            nn.Linear(3, 64),
            nn.ReLU(),  # relu1-1
            nn.Linear(64, 64),
            nn.ReLU(),  # relu1-2
            nn.Linear(64, 128),
            nn.ReLU(),  # relu2-1
        )


    def forward(self, input):
        out = self.model(input)
        return out
    