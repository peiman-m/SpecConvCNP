import math
import torch
from torch.distributions import MultivariateNormal



class CurveSampler(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def sample(
        self,
        batch_size=16,
        num_context=None,
        max_num_context=50,
        num_target=None,
        max_num_target=50,
        x_range=(-2, 2),
        device='cpu'
        ):
        num_context = num_context or torch.randint(low=3, high=max(3, max_num_context+1), size=[1]).item()
        num_target = num_target or torch.randint(low=3, high=max(3, max_num_target+1), size=[1]).item()
        num_points = num_context + num_target

        x = x_range[0] + (x_range[1] - x_range[0]) \
            * torch.rand([batch_size, num_points, 1], device=device)

        y = self.kernel(x)

        x_context = x[:,:num_context]
        x_target = x[:,num_context:]
        y_context = y[:,:num_context]
        y_target = y[:,num_context:]

        return {'x': x,
                'y': y,
                'x_context': x_context,
                'y_context': y_context,
                'x_target': x_target,
                'y_target': y_target}



class RBFKernel(object):
    def __init__(
        self,
        sigma_eps=2e-2,
        length=0.25,
        scale=0.75
        ):
        self.sigma_eps = sigma_eps
        self.length = length
        self.scale = scale

    def __call__(self, x):
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3)) / self.length

        cov = (self.scale**2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
              + self.sigma_eps**2 * torch.eye(x.shape[-2], device=x.device)
        mean = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)
        return y



class Matern52Kernel(object):
    def __init__(
        self,
        sigma_eps=2e-2,
        length=0.25,
        scale=0.75
        ):
        self.sigma_eps = sigma_eps
        self.length = length
        self.scale = scale

    def __call__(self, x):
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3)) / self.length, dim=-1)

        cov = (self.scale**2) * (1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
              * torch.exp(-math.sqrt(5.0) * dist) \
              + self.sigma_eps**2 * torch.eye(x.shape[-2], device=x.device)
        mean = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)
        return y



class PeriodicKernel(object):
    def __init__(
        self,
        sigma_eps=2e-2,
        length=0.6,
        scale=0.75,
        period=1
        ):
        self.sigma_eps = sigma_eps
        self.length = length
        self.scale = scale
        self.period = period

    def __call__(self, x):
        dist = x.unsqueeze(-2) - x.unsqueeze(-3)

        cov = (self.scale ** 2) * torch.exp(\
              - 2*(torch.sin(math.pi*dist.abs().sum(-1)/self.period) / self.length).pow(2)) \
              + self.sigma_eps**2 * torch.eye(x.shape[-2], device=x.device)
        mean = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)
        return y



class SawtoothKernel(object):
    def __init__(
        self,
        amp_range=(1, 2),
        freq_range=(3, 5),
        shift_range=(-5, 5),
        trunc_range=(10, 20)
        ):
        self.amp_range = amp_range
        self.freq_range = freq_range
        self.shift_range = shift_range
        self.trunc_range = trunc_range

    def __call__(self, x):
        amp = self.amp_range[0] + (self.amp_range[1] - self.amp_range[0]) \
              * torch.rand([x.shape[0]], device=x.device)
        freq = self.freq_range[0] + (self.freq_range[1] - self.freq_range[0]) \
               * torch.rand([x.shape[0]], device=x.device)
        shift = self.shift_range[0] + (self.shift_range[1] - self.shift_range[0]) \
                * torch.rand([x.shape[0]], device=x.device)
        trunc = torch.randint(low=self.trunc_range[0], high=self.trunc_range[1]+1, size=[x.shape[0]], device=x.device)

        y_values = []
        for i in range(x.shape[0]):
            x_values = torch.tile(x[i, :, :, None], dims=[1, 1, trunc[i]]) + shift[i]
            k = torch.tile(torch.reshape(torch.arange(start=1, end=trunc[i] + 1, device=x_values.device), shape=(1, 1, -1)),
                           dims=[x.shape[1], 1, 1])
            y = amp[i]/2 - amp[i] / math.pi * torch.sum((-1)**k * torch.sin(2 * math.pi * k * freq[i] * x_values) / k, dim=-1)
            y -= 0.5 * amp[i]
            y_values.append(y)
        
        y = torch.stack(y_values)
        return y



class SquareWaveKernel(object):
    def __init__(
        self,
        amp_range=(1, 2),
        freq_range=(2, 5)
        ):
        self.amp_range = amp_range
        self.freq_range = freq_range

    def __call__(self, x):
        amp = self.amp_range[0] + (self.amp_range[1] - self.amp_range[0]) \
              * torch.rand([x.shape[0]], device=x.device)
        freq = self.freq_range[0] + (self.freq_range[1] - self.freq_range[0]) \
               * torch.rand([x.shape[0]], device=x.device)
        phase_shift = 1.0 / freq + (1 - 1.0 / freq) \
                      * torch.rand([x.shape[0]], device=x.device)

        cond = torch.floor(freq[:, None, None] * x - phase_shift[:, None, None]) % 2
        y = amp[:, None, None] * (torch.where(cond==0, 1, 0) - 0.5)
        return y



