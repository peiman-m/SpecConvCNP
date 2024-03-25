import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_dists



class BatchLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True
        ):
        super(BatchLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
            )
        nn.init.xavier_normal_(self.weight, gain=1)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        num_functions, num_inputs = x.shape[0], x.shape[1]
        x = x.reshape(num_functions * num_inputs, self.in_features)
        out = super(BatchLinear, self).forward(x)
        return out.reshape(num_functions, num_inputs, self.out_features)



class SpectralConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes
        ):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, out_length=None):
        if out_length is None:
            out_length = x.size(-1)
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels,  out_length // 2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)
        out = torch.fft.irfft(out_ft, n=out_length)
        return out



class Affine1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Affine1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x, out_length=None):
        if out_length is None:
            out_length = x.size(-1)
        h = self.conv(x)
        out = F.interpolate(h, size=out_length, mode='linear', align_corners=True)
        return out



class FourierBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes
        ):
        super(FourierBlock1d, self).__init__()
        self.activation = nn.GELU()
        self.sconv = SpectralConv1d(in_channels, out_channels, modes)
        self.skip_path = Affine1d(in_channels, out_channels)

    def forward(self, x, out_length=None):
        h1 = self.sconv(x, out_length)
        h2 = self.skip_path(x, out_length)
        out = self.activation(h1 + h2)
        return out



class UNO(nn.Module):
    def __init__(
        self,
        in_channels=16,
        out_channels=16
        ):
        super(UNO, self).__init__()
        self.activation = nn.GELU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_halving_layers = 3

        self.fc1 = BatchLinear(self.in_channels + 1, 32)
        self.fc2 = BatchLinear(32, 64)

        self.l1 = FourierBlock1d(in_channels=64, out_channels=64, modes=32)
        self.l2 = FourierBlock1d(in_channels=64, out_channels=128, modes=32)
        self.l3 = FourierBlock1d(in_channels=128, out_channels=256, modes=32)
        self.l4 = FourierBlock1d(in_channels=256, out_channels=128, modes=32)
        self.l5 = FourierBlock1d(in_channels=128 + 64, out_channels=64, modes=32)

        self.fc3 = BatchLinear(2 * 64, 32)
        self.fc4 = BatchLinear(32, self.out_channels)

    def forward(self, x, grid):
        h = torch.cat((x, grid), dim=-1)
        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = h.permute(0, 2, 1)

        d = h.size(-1)

        h1 = self.activation(self.l1(h, d//2))
        h2 = self.activation(self.l2(h1, d//4))
        h3 = self.activation(self.l3(h2, d//4))
        h4 = self.activation(self.l4(h3, d//2))
        h4 = torch.cat((h4, h1), dim=1)
        h5 = self.activation(self.l5(h4, d))
        h5 = torch.cat((h5, h), dim=1)

        h = h5.permute(0, 2, 1)
        h = self.activation(self.fc3(h))
        out = self.fc4(h)
        return out



class ConvDeepSet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        init_length_scale,
        normalize
        ):
        super(ConvDeepSet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = 2 * in_channels if normalize else in_channels
        self.normalize = normalize
        self.sigma = nn.Parameter(
            np.log(init_length_scale) * torch.ones(self.hidden_channels),
            requires_grad=True)
        self.fc = BatchLinear(self.hidden_channels, out_channels)
        self.sigma_fn = torch.exp

    def rbf(self, dists):
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

    def forward(self, x, y, t):
        batch_size = x.shape[0]
        n_in = x.shape[1]

        dists = compute_dists(x, t)
        wt = self.rbf(dists)

        if self.normalize:
            density = torch.ones((batch_size, n_in, self.in_channels), device=x.device)
            y_out = torch.cat([density, y], dim=2)
        else:
            y_out = y

        y_out = y_out.unsqueeze(2) * wt
        y_out = y_out.sum(1)

        if self.normalize:
            density, conv = y_out.split(self.in_channels, dim=-1)
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        y_out = self.fc(y_out)
        return y_out



