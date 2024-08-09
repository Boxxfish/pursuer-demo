use candle_core::{Device, Tensor, D};
use candle_nn::{self as nn, Module, VarBuilder};

use crate::net::LoadableNN;

const BN_EPS: f64 = 1e-5;

pub struct BatchNorm1d {
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub bias: Tensor,
    pub weight: Tensor,
}

impl BatchNorm1d {
    pub fn new(num_features: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let running_mean = vb.get(&[num_features], "running_mean")?;
        let running_var = vb.get(&[num_features], "running_var")?;
        let bias = vb.get(&[num_features], "bias")?;
        let weight = vb.get(&[num_features], "weight")?;
        Ok(Self {
            running_mean,
            running_var,
            bias,
            weight,
        })
    }
}

impl Module for BatchNorm1d {
    fn forward(
        &self,
        xs: &Tensor, // Shape: (batch_size, num_features)
    ) -> candle_core::Result<Tensor> {
        let running_mean = self.running_mean.unsqueeze(0)?;
        let running_var = self.running_var.unsqueeze(0)?;
        let bias = self.bias.unsqueeze(0)?;
        let weight = self.weight.unsqueeze(0)?;
        (xs.broadcast_sub(&running_mean)?
            .broadcast_div(&((running_var + BN_EPS)?.sqrt()? * weight)?)?)
        .broadcast_add(&bias)
    }
}

pub struct BatchNorm2d {
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub bias: Tensor,
    pub weight: Tensor,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let running_mean = vb.get(&[num_features], "running_mean")?;
        let running_var = vb.get(&[num_features], "running_var")?;
        let bias = vb.get(&[num_features], "bias")?;
        let weight = vb.get(&[num_features], "weight")?;
        Ok(Self {
            running_mean,
            running_var,
            bias,
            weight,
        })
    }
}

impl Module for BatchNorm2d {
    fn forward(
        &self,
        xs: &Tensor, // Shape: (batch_size, num_features, w, h)
    ) -> candle_core::Result<Tensor> {
        let num_features = self.running_mean.shape().dims()[0];
        let running_mean = self.running_mean.reshape(&[1, num_features, 1, 1])?;
        let running_var = self.running_var.reshape(&[1, num_features, 1, 1])?;
        let weight = self.weight.reshape(&[1, num_features, 1, 1])?;
        let bias = self.bias.reshape(&[1, num_features, 1, 1])?;
        (xs.broadcast_sub(&running_mean)?
            .broadcast_div(&((running_var + BN_EPS)?.sqrt()? * weight)?)?)
        .broadcast_add(&bias)
    }
}

pub struct Backbone {
    pub grid_net: nn::Sequential,
    pub pos: Tensor,
    pub use_pos: bool,
}

impl Backbone {
    pub fn new(
        use_pos: bool,
        use_bn: bool,
        channels: usize,
        out_channels: usize,
        size: usize,
        objs_shape: Option<(usize, usize)>,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let mut num_channels = channels;
        if use_pos {
            num_channels += 2;
        }

        // Grid + scalar feature processing
        let mid_channels = 16;
        let grid_vb = vb.pp("grid_net");
        let grid_net = nn::seq()
            .add(nn::conv2d(
                num_channels,
                mid_channels,
                5,
                nn::Conv2dConfig {
                    padding: 5 / 2,
                    ..Default::default()
                },
                grid_vb.pp("0"),
            )?)
            .add(if use_bn {
                nn::seq().add(BatchNorm2d::new(mid_channels, grid_vb.pp("1"))?)
            } else {
                nn::seq()
            })
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                mid_channels,
                mid_channels * 2,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                grid_vb.pp("3"),
            )?)
            .add(if use_bn {
                nn::seq().add(BatchNorm2d::new(mid_channels * 2, grid_vb.pp("4"))?)
            } else {
                nn::seq()
            })
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                mid_channels * 2,
                out_channels,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                grid_vb.pp("6"),
            )?)
            .add(if use_bn {
                nn::seq().add(BatchNorm2d::new(out_channels, grid_vb.pp("7"))?)
            } else {
                nn::seq()
            })
            .add(nn::Activation::Silu);

        // Positional encoding
        let x_channel = (Tensor::arange(0, size as u32, &Device::Cpu)?
            .unsqueeze(0)?
            .repeat(&[size, 1])?
            .to_dtype(candle_core::DType::F64)?
            / size as f64)?
            .to_dtype(candle_core::DType::F32)?;
        let y_channel = x_channel.t()?;
        let pos = Tensor::stack(&[x_channel, y_channel], 0)?; // Shape: (2, grid_size, grid_size)

        // // Fusion of object features into grid + scalar features
        // if use_objs {
        //     _, obj_dim = objs_shape
        //     n_heads = 4
        //     self.proj = nn.Conv1d(obj_dim, out_channels, 1)
        //     self.attn1 = nn.MultiheadAttention(
        //         out_channels, n_heads, batch_first=True
        //     )
        //     self.bn1 = nn.BatchNorm1d(out_channels)
        //     self.attn2 = nn.MultiheadAttention(
        //         out_channels, n_heads, batch_first=True
        //     )
        //     self.bn2 = nn.BatchNorm1d(out_channels)
        //     self.attn3 = nn.MultiheadAttention(
        //         out_channels, n_heads, batch_first=True
        //     )
        //     self.bn3 = nn.BatchNorm1d(out_channels)
        // }

        Ok(Self {
            grid_net,
            pos,
            use_pos,
        })
    }
}

impl Backbone {
    fn forward(
        &self,
        grid: &Tensor,                   // Shape: (batch_size, channels, size, size)
        objs: Option<&Tensor>,           // Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Option<&Tensor>, // Shape: (batch_size, max_obj_size)
    ) -> candle_core::Result<Tensor> {
        // Shape: (batch_size, grid_features_dim, grid_size, grid_size) {
        // Concat pos encoding to grid
        let device = grid.device();
        let mut grid = grid.clone();
        let batch_size = grid.shape().dims()[0];
        if self.use_pos {
            grid = Tensor::cat(
                &[
                    grid,
                    self.pos
                        .unsqueeze(0)?
                        .repeat(&[batch_size, 1, 1, 1])?
                        .to_device(device)?,
                ],
                1,
            )?;
        }
        let grid_features = self.grid_net.forward(&grid)?; // Shape: (batch_size, grid_features_dim, grid_size, grid_size)

        // if self.use_objs {
        //     assert objs is not None;
        //     assert objs_attn_mask is not None;
        //     objs = self.proj(objs.permute(0, 2, 1)).permute(
        //         0, 2, 1
        //     )  // Shape: (batch_size, max_obj_size, proj_dim)
        //     grid_features = grid_features.permute(
        //         0, 2, 3, 1
        //     )  // Shape: (batch_size, grid_size, grid_size, grid_features_dim)
        //     orig_shape = grid_features.shape
        //     grid_features = grid_features.flatten(
        //         1, 2
        //     )  // Shape: (batch_size, grid_size * grid_size, grid_features_dim)
        //     attns = [self.attn1, self.attn2, self.attn3]
        //     bns = [self.bn1, self.bn2, self.bn3]
        //     for attn, bn in zip(attns, bns):
        //         attn_grid_features = torch.nan_to_num(
        //             attn(
        //                 query=grid_features,
        //                 key=objs,
        //                 value=objs,
        //                 key_padding_mask=objs_attn_mask,
        //             )[0],
        //             0.0,
        //         )  // Sometimes there are no objects, causing NANs
        //         grid_features = grid_features + attn_grid_features;
        //         if self.use_bn{
        //             grid_features = bn(grid_features.permute(0, 2, 1)).permute(0, 2, 1);}
        //         grid_features = nn.functional.silu(grid_features);
        //     grid_features = grid_features.view(orig_shape).permute(
        //         0, 3, 1, 2
        //     );  // Shape: (batch_size, grid_features_dim, grid_size, grid_size)
        // }

        Ok(grid_features)
    }
}

pub struct MeasureModel {
    pub backbone: Backbone,
    pub out_net: nn::Sequential,
}

// Not great, but we don't break these invariants
unsafe impl Send for MeasureModel {}
unsafe impl Sync for MeasureModel {}

impl LoadableNN for MeasureModel {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let channels = 9;
        let size = 8;
        let use_pos = true;
        let objs_shape = None;
        let proj_dim = 32;

        let backbone = Backbone::new(
            use_pos,
            true,
            channels,
            proj_dim,
            size,
            objs_shape,
            vb.pp("backbone"),
        )?;

        // Convert features into liklihood map
        let out_vb = vb.pp("out_net");
        let out_net = nn::seq()
            .add(nn::conv2d(
                proj_dim,
                32,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                out_vb.pp("0"),
            )?)
            .add(BatchNorm2d::new(32, out_vb.pp("1"))?)
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                32,
                32,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                out_vb.pp("3"),
            )?)
            .add(BatchNorm2d::new(32, out_vb.pp("4"))?)
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                32,
                1,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                out_vb.pp("6"),
            )?)
            .add(nn::Activation::Sigmoid);

        Ok(MeasureModel { out_net, backbone })
    }
}

impl MeasureModel {
    pub fn forward(
        &self,
        grid: &Tensor,                   // Shape: (batch_size, channels, size, size)
        objs: Option<&Tensor>,           // Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Option<&Tensor>, // Shape: (batch_size, max_obj_size)
    ) -> candle_core::Result<Tensor> {
        let grid_features = self.backbone.forward(grid, objs, objs_attn_mask)?;
        self.out_net.forward(&grid_features)?.squeeze(1) // Shape: (batch_size, grid_size, grid_size)
    }
}

pub struct PolicyNet {
    pub backbone: Backbone,
    pub net: nn::Sequential,
}

// Not great, but we don't break these invariants
unsafe impl Send for PolicyNet {}
unsafe impl Sync for PolicyNet {}

impl LoadableNN for PolicyNet {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let channels = 7;
        let size = 16;
        let use_pos = true;
        let objs_shape = None;
        let proj_dim = 64;
        let action_count = 9;

        let backbone = Backbone::new(
            use_pos,
            false,
            channels,
            proj_dim,
            size,
            objs_shape,
            vb.pp("backbone"),
        )?;
        
        let net1_vb = vb.pp("net1");
        let net2_vb = vb.pp("net2");
        let net = nn::seq()
            .add(nn::conv2d(
                proj_dim,
                64,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                net1_vb.pp("0"),
            )?)
            .add(nn::Activation::Silu)
            .add_fn(|t| t.max(D::Minus1)?.max(D::Minus1))
            .add(nn::linear(64, 256, net2_vb.pp("0"))?)
            .add(nn::Activation::Silu)
            .add(nn::linear(256, action_count, net2_vb.pp("2"))?);
        Ok(PolicyNet { net, backbone })
    }
}

impl PolicyNet {
    pub fn forward(
        &self,
        grid: &Tensor,                   // Shape: (batch_size, channels, size, size)
        objs: Option<&Tensor>,           // Shape: (batch_size, max_obj_size, obj_dim)
        objs_attn_mask: Option<&Tensor>, // Shape: (batch_size, max_obj_size)
    ) -> candle_core::Result<Tensor> {
        let grid_features: Tensor = self.backbone.forward(grid, objs, objs_attn_mask)?;
        self.net.forward(&grid_features) // Shape: (batch_size, num_actions)
    }
}
