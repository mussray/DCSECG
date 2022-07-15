import torch


def loss_fn(outputs, inputs):
    mse = ((inputs.reshape(256) - outputs.reshape(256)) ** 2).mean()
    loss = torch.sqrt((torch.sqrt(mse) ** 2).mean())
    return mse
