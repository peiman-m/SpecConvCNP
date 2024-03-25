import torch
import torch.nn as nn
from utils import to_multiple
from modules import ConvDeepSet


class SConvCNP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        rho,
        points_per_unit
        ):
        super(SConvCNP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers

        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit
        
        self.encoder = ConvDeepSet(
            in_channels=self.in_channels,
            out_channels=self.rho.in_channels,
            init_length_scale=init_length_scale,
            normalize=True
            )
        
        self.mean_layer = ConvDeepSet(
            in_channels=self.rho.out_channels,
            out_channels=self.out_channels,
            init_length_scale=init_length_scale,
            normalize=False
            )
        self.sigma_layer = ConvDeepSet(
            in_channels=self.rho.out_channels,
            out_channels=self.out_channels,
            init_length_scale=init_length_scale,
            normalize=False)
        
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()


    def forward(self, x, y, x_out):
        x_grid = self.discretize_input(x, x_out)
        h = self.activation(self.encoder(x, y, x_grid))
        h = self.rho(h, x_grid)

        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        mean = self.mean_layer(x_grid, h, x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))
        
        return mean, sigma


    def discretize_input(self, x, x_out, x_min=-2.0, x_max=2.0, margin=0.1):
        x_min = min(torch.min(x).cpu().numpy(),
                    torch.min(x_out).cpu().numpy(), x_min) - margin
        x_max = max(torch.max(x).cpu().numpy(),
                    torch.max(x_out).cpu().numpy(), x_max) + margin
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(x.device)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)
        return x_grid
