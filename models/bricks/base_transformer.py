import torch
import torchvision
from torch import nn


class DETRBaseTransformer(nn.Module):
    """A base class that contains some methods commonly used in DETR transformer,
    such as DeformableTransformer, DabTransformer, DINOTransformer, AlignTransformer.

    """
    def __init__(self, num_feature_levels, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_feature_levels = num_feature_levels
        self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dim))
        self._init_weights_detr_transformer()

    def _init_weights_detr_transformer(self):
        nn.init.normal_(self.level_embeds)

    @staticmethod
    def flatten_multi_level(multi_level_elements):
        multi_level_elements = torch.cat([e.flatten(-2) for e in multi_level_elements], -1)  # (b, [c], s)
        if multi_level_elements.ndim == 3:
            multi_level_elements.transpose_(1, 2)
        return multi_level_elements

    def get_lvl_pos_embed(self, multi_level_pos_embeds):
        multi_level_pos_embeds = [
            p + l.view(1, -1, 1, 1) for p, l in zip(multi_level_pos_embeds, self.level_embeds)
        ]
        return self.flatten_multi_level(multi_level_pos_embeds)

    def multi_level_misc(self, multi_level_masks):
        if torchvision._is_tracing():
            # torch.Tensor.shape exports not well for ONNX
            # use operators.shape_as_tensor istead
            from torch.onnx import operators
            spatial_shapes = [operators.shape_as_tensor(m)[-2:] for m in multi_level_masks]
            spatial_shapes = torch.stack(spatial_shapes).to(multi_level_masks[0].device)
        else:
            spatial_shapes = [m.shape[-2:] for m in multi_level_masks]
            spatial_shapes = torch.as_tensor(spatial_shapes, device=multi_level_masks[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratios(m) for m in multi_level_masks], 1)
        return spatial_shapes, level_start_index, valid_ratios

    @staticmethod
    def get_valid_ratios(mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # [n, 2]
        return valid_ratio


class TwostageTransformer(DETRBaseTransformer):
    """A base class that contains some methods commonly used in two-stage transformer,
    such as DeformableTransformer, DabTransformer, DINOTransformer, AlignTransformer.

    """
    def __init__(self, num_feature_levels, embed_dim):
        super().__init__(num_feature_levels, embed_dim)
        self.enc_output = nn.Linear(embed_dim, embed_dim)
        self.enc_output_norm = nn.LayerNorm(embed_dim)
        self._init_weights_two_stage_transformer()

    def _init_weights_two_stage_transformer(self):
        nn.init.xavier_uniform_(self.enc_output.weight)
        nn.init.constant_(self.enc_output.bias, 0.0)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        n, s, c = memory.shape
        proposals = []
        cur = 0
        if torchvision._is_tracing():
            # avoid iteration warning on torch.Tensor
            # convert Tensor to list[Tensor] instead
            spatial_shapes = [b.unbind(0) for b in spatial_shapes.unbind(0)]
        else:
            # use list to avoid small kernel launching when indexing spatial shapes
            spatial_shapes = spatial_shapes.tolist()

        for lvl, (h, w) in enumerate(spatial_shapes):
            mask_flatten = memory_padding_mask[:, cur:(cur + h * w)].view(n, h, w, 1)
            valid_h = torch.sum(~mask_flatten[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, h - 1, h, dtype=torch.float32, device=memory.device),
                torch.linspace(0, w - 1, w, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], -1)  # [h, w, 2]
            scale = torch.stack([valid_w, valid_h], -1).view(n, 1, 1, 2)
            grid = (grid.expand(n, -1, -1, -1) + 0.5) / scale  # [n, h, w, 2]
            wh = torch.ones_like(grid) * 0.05 * 2.0**lvl
            proposal = torch.cat([grid, wh], -1).view(n, -1, 4)
            proposals.append(proposal)
            cur += h * w
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse_sigmoid
        output_proposals.masked_fill_(
            memory_padding_mask.unsqueeze(-1) | ~output_proposals_valid, float("inf")
        )

        output_memory = memory * (~memory_padding_mask.unsqueeze(-1)) * (output_proposals_valid)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals
