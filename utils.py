import torch
from torch.distributions import Normal



def to_multiple(x, multiple):
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple



def compute_dists(x, y):
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2



def gaussian_logpdf(
    inputs,
    mean,
    sigma,
    reduction=None
    ):
    dist = Normal(loc=mean, scale=sigma)
    logp = dist.log_prob(inputs)

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')


def train(
    model,
    optimizer,
    num_tasks,
    batch_size,
    data_generator,
    device
    ):
    model.train()
    log_p = []
    for _ in range(num_tasks):
        task = data_generator.sample(
            batch_size=batch_size,
            device=device
            )
        optimizer.zero_grad()
        y_mean, y_std = model(task['x_context'], task['y_context'], task['x_target'])
        loss = -gaussian_logpdf(task['y_target'], y_mean, y_std, 'mean')
        loss.backward()
        optimizer.step()
        log_p.append(-loss.item())

    return model, sum(log_p) / len(log_p)



@torch.no_grad()
def eval(
    model,
    num_tasks,
    batch_size,
    data_generator,
    device
    ):
    model.eval()
    log_p = []
    for _ in range(num_tasks):
        task = data_generator.sample(
            batch_size=batch_size,
            device=device
            )
        y_mean, y_std = model(task['x_context'], task['y_context'], task['x_target'])
        ll = gaussian_logpdf(task['y_target'], y_mean, y_std, 'mean')
        log_p.append(ll.item())

    return sum(log_p) / len(log_p)



def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"ConvCNP -> trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable %: {100 * trainable_params / all_param}"
    )
