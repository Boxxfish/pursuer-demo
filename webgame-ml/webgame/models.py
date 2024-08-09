import torch
from torch import nn, Tensor
from typing import *


class Backbone(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        size: int,
        use_pos: bool = False,
        objs_shape: Optional[Tuple[int, int]] = None,
        use_bn: bool = False,
    ):
        super().__init__()
        num_channels = channels
        if use_pos:
            num_channels += 2
        self.use_objs = objs_shape is not None

        # Grid + scalar feature processing
        mid_channels = 16
        self.grid_net = nn.Sequential(
            nn.Conv2d(num_channels, mid_channels, 5, padding="same", dtype=torch.float),
            (
                nn.BatchNorm2d(mid_channels, dtype=torch.float)
                if use_bn
                else nn.Identity()
            ),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels * 2, 3, padding="same", dtype=torch.float),
            (
                nn.BatchNorm2d(mid_channels * 2, dtype=torch.float)
                if use_bn
                else nn.Identity()
            ),
            nn.SiLU(),
            nn.Conv2d(mid_channels * 2, out_channels, 3, padding="same", dtype=torch.float),
            (
                nn.BatchNorm2d(out_channels, dtype=torch.float)
                if use_bn
                else nn.Identity()
            ),
            nn.SiLU(),
        )

        # Positional encoding
        x_channel = torch.tensor([list(range(size))] * size, dtype=torch.float) / size
        y_channel = x_channel.T
        self.pos = torch.stack(
            [x_channel, y_channel]
        )  # Shape: (2, grid_size, grid_size)
        self.pos.requires_grad = False
        self.use_pos = use_pos
        self.use_bn = use_bn

        # Fusion of object features into grid + scalar features
        if self.use_objs:
            assert objs_shape is not None
            _, obj_dim = objs_shape
            n_heads = 4
            self.proj = nn.Conv1d(obj_dim, out_channels, 1)
            self.attn1 = nn.MultiheadAttention(
                out_channels, n_heads, batch_first=True, dtype=torch.float
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.attn2 = nn.MultiheadAttention(
                out_channels, n_heads, batch_first=True, dtype=torch.float
            )
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.attn3 = nn.MultiheadAttention(
                out_channels, n_heads, batch_first=True, dtype=torch.float
            )
            self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(
        self,
        grid: Tensor,  # Shape: (batch_size, channels, size, size)
        objs: Optional[Tensor],  # Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Optional[Tensor],  # Shape: (batch_size, max_obj_size)
    ) -> Tensor:  # Shape: (batch_size, grid_features_dim, grid_size, grid_size)
        # Concat pos encoding to grid
        device = grid.device
        if self.use_pos:
            grid = torch.concat(
                [
                    grid,
                    self.pos.unsqueeze(0)
                    .tile([grid.shape[0], 1, 1, 1])
                    .to(device=device),
                ],
                dim=1,
            )
        grid_features = self.grid_net(
            grid
        )  # Shape: (batch_size, grid_features_dim, grid_size, grid_size)

        if self.use_objs:
            assert objs is not None
            assert objs_attn_mask is not None
            batch_attn_mask = ~(
                (~objs_attn_mask).sum(1).to(torch.bool)
            )  # True if a batch doesn't have objects
            objs_attn_mask[:, 0] = (
                False  # Makes it so at least one element is always attendable, batches with no objects get masked out
            )
            objs = self.proj(objs.permute(0, 2, 1)).permute(
                0, 2, 1
            )  # Shape: (batch_size, max_obj_size, proj_dim)
            grid_features = grid_features.permute(
                0, 2, 3, 1
            )  # Shape: (batch_size, grid_size, grid_size, grid_features_dim)
            orig_shape = grid_features.shape
            grid_features = grid_features.flatten(
                1, 2
            )  # Shape: (batch_size, grid_size * grid_size, grid_features_dim)
            attns = [self.attn1, self.attn2, self.attn3]
            bns = [self.bn1, self.bn2, self.bn3]
            for attn, bn in zip(attns, bns):
                attn_grid_features = attn(
                    query=grid_features,
                    key=objs,
                    value=objs,
                    key_padding_mask=objs_attn_mask,
                )[0]
                attn_grid_features[batch_attn_mask, :, :] = 0.0
                if self.use_bn:
                    attn_grid_features = bn(
                        attn_grid_features.permute(0, 2, 1)
                    ).permute(0, 2, 1)
                attn_grid_features = nn.functional.silu(attn_grid_features)
                attn_grid_features[batch_attn_mask, :, :] = 0.0
                grid_features = grid_features + attn_grid_features
            grid_features = grid_features.view(orig_shape).permute(
                0, 3, 1, 2
            )  # Shape: (batch_size, grid_features_dim, grid_size, grid_size)

        return grid_features


class MeasureModel(nn.Module):
    def __init__(
        self,
        channels: int,
        size: int,
        use_pos: bool = False,
        objs_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        proj_dim = 32
        self.backbone = Backbone(channels, proj_dim, size, use_pos, objs_shape, True)

        # Convert features into liklihood map
        self.out_net = nn.Sequential(
            nn.Conv2d(proj_dim, 32, 3, padding="same", dtype=torch.float),
            nn.BatchNorm2d(32, dtype=torch.float),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding="same", dtype=torch.float),
            nn.BatchNorm2d(32, dtype=torch.float),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding="same", dtype=torch.float),
            nn.Sigmoid(),
        )

    def forward(
        self,
        grid: Tensor,  # Shape: (batch_size, channels, size, size)
        objs: Optional[Tensor],  # Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Optional[Tensor],  # Shape: (batch_size, max_obj_size)
    ) -> Tensor:
        grid_features = self.backbone(grid, objs, objs_attn_mask)

        return self.out_net(grid_features).squeeze(
            1
        )  # Shape: (batch_size, grid_size, grid_size)


class PolicyNet(nn.Module):
    def __init__(
        self,
        channels: int,
        size: int,
        action_count: int,
        use_pos: bool = False,
        objs_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        proj_dim = 64
        self.backbone = Backbone(channels, proj_dim, size, use_pos, objs_shape)
        self.net1 = nn.Sequential(
            nn.Conv2d(proj_dim, 64, 3, padding="same", dtype=torch.float),
            nn.SiLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, action_count),
        )

    def forward(
        self,
        grid: Tensor,  # Shape: (batch_size, channels, size, size)
        objs: Optional[Tensor],  # Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Optional[Tensor],  # Shape: (batch_size, max_obj_size)
    ) -> Tensor:
        features = self.backbone(grid, objs, objs_attn_mask)
        values = self.net1(features)
        values = values.amax(-1).amax(-1)
        values = self.net2(values)
        return values
