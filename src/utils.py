import torch


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


def tensor_sample_preproc(data_, nsamples=100):
    '''
    100 samples seems enough
    1. Average per-channel distributions
    2. Random sample from that distribution
    '''
    if not isinstance(data_, torch.Tensor):
        return data_
    per_channel_mean = data_.mean((0, 1)).flatten() # mean per batch and per-channel
    if nsamples >= per_channel_mean.numel():
        return data_.flatten().cpu().detach()
    sampled_indices = torch.multinomial(per_channel_mean - per_channel_mean.min(), 
                                        nsamples, replacement=False)  # shift values get non-negative values
    sampled_distribution = per_channel_mean[sampled_indices]

    return sampled_distribution.flatten().cpu().detach()


def tensor_preproc(data_, avg_batch=False):
    if not isinstance(data_, torch.Tensor):
        return data_
    if avg_batch:
        data = torch.stack([data_[:, channel, :, :].mean(0).flatten()
                           for channel in range(data_.shape[1])])
    else:
        data = torch.stack([data_[:, channel, :, :].flatten()
                           for channel in range(data_.shape[1])])
    data = data.to(torch.device('cpu'))
    return data.detach()
