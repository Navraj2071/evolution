import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy
import time


class Organism(nn.Module):
    def __init__(
        self, input_size=600, output_size=203, hidden_size=600, n_hidden_layers=5
    ):
        super(Organism, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device="cuda"))
        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size, device="cuda"))
        self.layers.append(nn.Linear(hidden_size, output_size, device="cuda"))
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def get_action(self, x):
        x = torch.tensor(x, dtype=torch.float32, device="cuda")
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        output[0] = torch.tanh(output[0])
        output[1] = torch.tanh(output[1])
        output[2] = torch.clamp(output[2], 0, 255).int()
        output[3:] = torch.clamp(output[3:], -1200, 1200)
        return output

    def reproduce(self, noise_std=0.01):
        cloned_organism = Organism(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            n_hidden_layers=self.n_hidden_layers,
        )

        with torch.no_grad():
            for original_layer, cloned_layer in zip(
                self.layers, cloned_organism.layers
            ):
                if isinstance(original_layer, nn.Linear) and isinstance(
                    cloned_layer, nn.Linear
                ):
                    new_weights = (
                        original_layer.weight.clone()
                        + torch.randn_like(original_layer.weight) * noise_std
                    )
                    cloned_layer.weight.copy_(new_weights)
                    if original_layer.bias is not None:
                        new_bias = (
                            original_layer.bias.clone()
                            + torch.randn_like(original_layer.bias) * noise_std
                        )
                        cloned_layer.bias.copy_(new_bias)

        return cloned_organism

    def save(self, index):
        # print("saving player -- ", index)
        folder_path = f"./organisms/level2/{index}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        model_path = os.path.join(folder_path, "model.pth")
        params_path = os.path.join(folder_path, "params.pth")

        # Save the model state_dict
        torch.save(self.state_dict(), model_path)

        # Save the model parameters
        params = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
        }
        torch.save(params, params_path)

    @classmethod
    def load(cls, org_index):
        folder_path = f"./organisms/level2/{org_index}"
        params_path = os.path.join(folder_path, "params.pth")
        model_path = os.path.join(folder_path, "model.pth")

        if not os.path.exists(folder_path):
            organism = Organism()
            return organism

        # Load the parameters
        try:
            params = torch.load(params_path)
        except:
            time.sleep(30)
            params = torch.load(params_path)

        # Create a new instance with the loaded parameters
        organism = cls(
            input_size=params["input_size"],
            output_size=params["output_size"],
            hidden_size=params["hidden_size"],
            n_hidden_layers=params["n_hidden_layers"],
        )

        # Load the state dictionary
        try:
            state_dict = torch.load(model_path)
        except:
            time.sleep(30)
            state_dict = torch.load(model_path)

        organism.load_state_dict(state_dict)

        return organism
