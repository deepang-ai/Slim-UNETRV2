import torch
import torch.nn as nn
from mamba_ssm import Mamba
from monai.networks.layers.utils import get_act_layer


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, shallow=True):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        if shallow:
            self.act = get_act_layer(name="gelu")
        else:
            self.act = get_act_layer(name="swish")
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SFM(nn.Module):
    def __init__(self, in_channles, shallow=True) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        if shallow:
            self.nonliner = get_act_layer(name="gelu")
        else:
            self.nonliner = get_act_layer(name="swish")

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        if shallow:
            self.nonliner2 = get_act_layer(name="gelu")
        else:
            self.nonliner2 = get_act_layer(name="swish")

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        if shallow:
            self.nonliner3 = get_act_layer(name="gelu")
        else:
            self.nonliner3 = get_act_layer(name="swish")

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        if shallow:
            self.nonliner4 = get_act_layer(name="gelu")
        else:
            self.nonliner4 = get_act_layer(name="swish")

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class SlimUNETRBlockV2(nn.Module):
    def __init__(
        self, dim, d_state=16, d_conv=4, expand=2, head=4, num_slices=4, step=1
    ):
        super(SlimUNETRBlockV2, self).__init__()
        self.dim = dim
        self.step = step
        self.num_heads = head
        self.head_dim = dim // head
        self.norm = nn.LayerNorm(dim)
        if self.step != 3 and self.step != 2:
            self.compression = nn.Conv3d(
                in_channels=dim, out_channels=dim, kernel_size=3, stride=4, padding=0
            )
            self.upsample = nn.ConvTranspose3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=4,
                padding=0,
                output_padding=1,
            )
        elif self.step == 2:
            self.compression = nn.Conv3d(
                in_channels=dim, out_channels=dim, kernel_size=1, stride=2, padding=0
            )
            self.upsample = nn.ConvTranspose3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            )

        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
            # nslices=num_slices,
        )
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x_skip = x
        if self.step != 3:
            x = self.compression(x)
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        B, C, H, W, Z = x.shape
        q, k, v = (
            self.qkv(out)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        out = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)

        if self.step != 3:
            out = self.upsample(out)

        out = out + x_skip

        return out


class SlimMambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans=4,
        kernel_sizes=[4, 2, 2, 2],
        depths=[1, 1, 1, 1],
        dims=[48, 96, 192, 384],
        num_slices_list=[64, 32, 16, 8],
        out_indices=[0, 1, 2, 3],
        heads=[1, 2, 4, 4],
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv3d(
            in_chans, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
        )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_sizes[i + 1],
                    stride=kernel_sizes[i + 1],
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.sfms = nn.ModuleList()
        cur = 0
        for i in range(4):
            shallow = True
            if i > 1:
                shallow = False
            sfm = SFM(dims[i], shallow)

            stage = nn.Sequential(
                *[
                    SlimUNETRBlockV2(
                        dim=dims[i],
                        num_slices=num_slices_list[i],
                        head=heads[i],
                        step=i,
                    )
                    for _ in range(depths[i])
                ]
            )

            self.stages.append(stage)

            self.sfms.append(sfm)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i_layer in range(4):
            self.norms.append(nn.InstanceNorm3d(dims[i_layer]))
            if i_layer >= 2:
                self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], False))
            else:
                self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], True))

    def forward(self, x):
        feature_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.sfms[i](x)

            feature_out.append(self.stages[i](x))

            if i in self.out_indices:
                x = self.norms[i](x)
                x = self.mlps[i](x)
        return x, feature_out


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, head, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed1 = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.transposed2 = nn.ConvTranspose3d(
            dim_out * 2, dim_out, kernel_size=1, stride=1
        )

    def forward(self, x, feature):
        x = self.transposed1(x)
        x = torch.cat((x, feature), dim=1)
        x = self.transposed2(x)
        x = self.norm(x)
        return x


class SlimUNETRV2(nn.Module):
    def __init__(
        self,
        in_chans=4,
        out_chans=3,
        kernel_sizes=[2, 2, 2, 2],
        num_slices_list=[16, 8, 4, 2],
        depths=[1, 1, 1, 1],
        dims=[48, 96, 192, 384],
        heads=[1, 2, 4, 4],
        hidden_size=768,
        out_indices=[0, 1, 2, 3],
    ):
        super(SlimUNETRV2, self).__init__()
        self.encoder = SlimMambaEncoder(
            in_chans=in_chans,
            kernel_sizes=kernel_sizes,
            depths=depths,
            dims=dims,
            num_slices_list=num_slices_list,
            out_indices=out_indices,
            heads=heads,
        )

        self.hidden_downsample = nn.Conv3d(
            dims[3], hidden_size, kernel_size=2, stride=2
        )

        self.tsconv1 = TransposedConvLayer(
            dim_in=hidden_size, dim_out=dims[3], head=heads[3], r=2
        )

        self.tsconv2 = TransposedConvLayer(
            dim_in=dims[3], dim_out=dims[2], head=heads[2], r=kernel_sizes[3]
        )
        self.tsconv3 = TransposedConvLayer(
            dim_in=dims[2], dim_out=dims[1], head=heads[1], r=kernel_sizes[2]
        )
        self.tsconv4 = TransposedConvLayer(
            dim_in=dims[1], dim_out=dims[0], head=heads[0], r=kernel_sizes[1]
        )

        self.seghead = nn.ConvTranspose3d(
            dims[0], out_chans, kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
        )

    def forward(self, x):
        outs, feature_out = self.encoder(x)

        deep_feature = self.hidden_downsample(outs)

        x = self.tsconv1(deep_feature, feature_out[-1])
        x = self.tsconv2(x, feature_out[-2])
        x = self.tsconv3(x, feature_out[-3])
        x = self.tsconv4(x, feature_out[-4])
        x = self.seghead(x)

        return x

