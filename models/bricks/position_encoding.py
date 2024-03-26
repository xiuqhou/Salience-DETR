import math
import functools
from typing import Tuple, Union

import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model. See `End-to-End Object Detection 
    with Transformers <https://arxiv.org/pdf/2005.12872>`_ for more details.

    :param num_pos_feats: The feature dimension for each position along x-axis or y-axis.
        The final returned dimension for each position is 2 times of the input value, 
        defaults to 64
    :param temperature: The temperature used for scaling the position embedding, defaults to 10000
    :param normalize: Whether to normalize the position embedding, defaults to False
    :param scale: A scale factor that scales the position embedding, which is used only when
        `normalize` is True, defaults to 2*math.pi
    :param eps: A value added to the denominator for numerical stability, defaults to 1e-6
    :param offset: An offset added to embed, defaults to 0.0
    """
    def __init__(
        self,
        num_pos_feats=64,
        temperature: Union[int, Tuple[int, int]] = 10000,
        normalize=False,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
    ):
        super().__init__()
        dim_t = 2 * torch.arange(num_pos_feats).div(2, rounding_mode="floor") / num_pos_feats
        if isinstance(temperature, int):
            dim_tx = dim_ty = temperature**dim_t
        else:
            assert len(temperature) == 2, "Only support two elements as (t_x, t_y) in temperature"
            dim_tx, dim_ty = [t**dim_t for t in temperature]
        self.register_buffer("dim_tx", dim_tx)
        self.register_buffer("dim_ty", dim_ty)

        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor):
        mask = mask.to(torch.int)
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        else:
            # RT-DETR uses unnormalized encoding with index from 0
            y_embed = y_embed + self.offset
            x_embed = x_embed + self.offset

        pos_x = x_embed[:, :, :, None] / self.dim_tx
        pos_y = y_embed[:, :, :, None] / self.dim_ty
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""
    def __init__(self, num_embeddings: int = 50, num_pos_feats: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
        self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask: Tensor):
        h, w = mask.shape[-2:]
        i = torch.arange(w, device=mask.device)
        j = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            ).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


@functools.lru_cache  # use lru_cache to avoid redundant calculation for dim_t
def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature**(2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    return dim_t


def get_sine_pos_embed(
    pos_tensor: Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
    exchange_xy: bool = True,
) -> Tensor:
    """Generate sine position embedding for a position tensor

    :param pos_tensor: shape as (..., 2*n).
    :param num_pos_feats: projected shape for each float in the tensor, defaults to 128
    :param temperature: the temperature used for scaling the position embedding, defaults to 10000
    :param exchange_xy: exchange pos x and pos. For example, 
        input tensor is [x, y], the results will be [pos(y), pos(x)], defaults to True
    :return: position embedding with shape (None, n * num_pos_feats)
    """
    dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)

    pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
    pos_res = torch.stack((pos_res[..., 0::2].sin(), pos_res[..., 1::2].cos()), dim=-1).flatten(-2)
    if exchange_xy:
        index = torch.cat([
            torch.arange(1, -1, -1, device=pos_res.device),
            torch.arange(2, pos_res.shape[-2], device=pos_res.device),
        ])
        pos_res = torch.index_select(pos_res, -2, index)
    pos_res = pos_res.view(*pos_tensor.shape[:-1], -1)
    return pos_res
