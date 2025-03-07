import torch


def efficient_one_hot(index, num_classes, dtype=torch.float32):
    """
    :param index:
    :param num_classes:
    :param dtype:
    :return:
    """
    one_hot = torch.zeros(
        *index.shape, num_classes, device=index.device, dtype=dtype)
    one_hot.scatter_(-1, index.unsqueeze(-1), 1)
    return one_hot
