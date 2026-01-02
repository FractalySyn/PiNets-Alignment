###### Part of the code comes from Prithvi and terrratorch 
      # https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
      # https://github.com/IBM/terratorch


from functools import partial
from typing import List, Tuple

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block


from abc import ABC, abstractmethod
import math
import warnings




class PiPrithvi(nn.Module):
  
  def __init__(self, backbone_cfg, C_out, dim_in, dim_out, DIR='', load_pretrained=True, freeze_backbone=False, device='cpu'):

    super().__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.freeze_backbone = freeze_backbone
    self.load_pretrained = load_pretrained
    self.ft_layers = list(range(0, backbone_cfg['depth']))[-4:]

    self.redim = lambda x, d: F.interpolate(x, size=(d, d), mode='bilinear', align_corners=False)
    self.encoder = PrithviViT(**backbone_cfg).to(device)
    self.reshape = ReshapeTokensToImage(self.ft_layers)
    self.decoder = UperNetDecoder([backbone_cfg['embed_dim']] * len(self.ft_layers), channels=32, scale_modules=True).to(device)
    self.decoder2 = Decoder(32, C_out, 64, dim_out).to(device)
    self.activ = lambda pi: F.softmax(pi, dim=1) if C_out > 1 else torch.sigmoid(pi)

    if load_pretrained:
    #   url = '/kaggle/working/Prithvi-EO-1.0-100M/Prithvi_EO_V1_100M.pt' 
      state_dict = torch.load(DIR+'Prithvi_EO_V1_100M.pt', map_location=torch.device(device))
      for k in list(state_dict.keys()):
          if 'pos_embed' in k:
              del state_dict[k]
      self.load_state_dict(state_dict, strict=False)

    if freeze_backbone:
      for param in self.encoder.parameters():
        param.requires_grad = False

  def forward(self, x):

    if self.dim_in != x.shape[-1]:
      x = self.redim(x, self.dim_in)
    if len(x.shape) == 4:
      x = x.unsqueeze(2)
    
    if self.freeze_backbone:
      self.encoder.eval()
      with torch.no_grad():
        h = self.encoder.forward_features(x)
    else:          
      h = self.encoder.forward_features(x)

    h = [h[i] for i in self.ft_layers]
    h = self.reshape(h)
    
    pi = self.decoder(h)
    pi = self.decoder2(pi)
    pi = self.activ(pi).squeeze()

    y = pi.sum(dim=(-2,-1))
    return pi, y






class Decoder(nn.Module):

  def __init__(self, C_in, C_out, dim_in, dim_out):
    super().__init__()
    self.interpolate = False

    self.depth = -int(np.log2(dim_in) - np.log2(dim_out))
    if dim_in * 2**self.depth < dim_out:
      self.depth += 1
      self.interpolate = True

    self.redim = lambda x: F.interpolate(x, size=(dim_out, dim_out), mode='bilinear', align_corners=False)

    self.net = nn.ModuleList()
    for i in range(self.depth):
      in_ch = C_in; out_ch = C_in // (3 if i == 0 else 2)
      self.net.append(nn.Sequential(
          nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(),
          nn.Dropout(0.1)
      ))
      C_in = out_ch
    self.head = nn.Conv2d(out_ch, C_out, kernel_size=1)

  def forward(self, h):
    for layer in self.net:
      h = layer(h)
    if self.interpolate:
      h = self.redim(h)
    pi = self.head(h)
    return pi
  







class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: int | None = None):
        """
        temporal_coords: year and day-of-year info with shape (B, T, 2).
        tokens_per_frame: number of tokens for each frame in the sample. If provided, embeddings will be
            repeated over T dimension, and final shape is (B, T*tokens_per_frame, embed_dim).
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = _get_1d_sincos_embed_from_grid_torch(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(shape)
        julian_day = _get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding  # B, T*tokens_per_frame, embed_dim


class PrithviViT(nn.Module):
    """ Prithvi ViT Encoder"""
    def __init__(self,
                 img_size: int | Tuple[int, int] = 224,
                 patch_size: int | Tuple[int, int, int] = (1, 16, 16),
                 num_frames: int = 1,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = partial(torch.nn.LayerNorm, eps=1e-6),
                 coords_encoding: List[str] | None = None,
                 coords_scale_learn: bool = False,
                 encoder_only: bool = True,  # needed for timm
                 ** kwargs,
                ):
        super().__init__()

        self.feature_info = []
        self.encoder_only = encoder_only
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)

        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            assert patch_size[0] == 1, f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        # Transformer layers
        self.blocks = []
        for i in range(depth):
            self.blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
            self.feature_info.append(
                {"num_chs": embed_dim * self.patch_embed.patch_size[0], "reduction": 1, "module": f"blocks.{i}"}
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def random_masking(self, sequence, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.
        Args:
            sequence (`torch.FloatTensor` of shape `(batch_size, sequence_length, dim)`)
            mask_ratio (float): mask ratio to use.
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def _get_pos_embed(self, x):
        t, h, w = x.shape[-3:]

        pos_embed = torch.from_numpy(get_3d_sincos_pos_embed(
            self.embed_dim,
            (
                t // self.patch_embed.patch_size[0],
                h // self.patch_embed.patch_size[1],
                w // self.patch_embed.patch_size[2],
            ),
            add_cls_token=True,
        )).float().unsqueeze(0).to(x)

        return pos_embed
    

    def forward(
        self, x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio=0.75
    ):
        if x.shape[-3:] != self.patch_embed.input_size:
            # changed input size
            pos_embed = self._get_pos_embed(x)
        else:
            pos_embed = self.pos_embed

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)

        if x.shape[-3:] != self.patch_embed.input_size:
            pos_embed = self._get_pos_embed(x)
        else:
            pos_embed = self.pos_embed

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding:
            num_tokens_per_frame = x.shape[1] // self.patch_embed.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x.clone())

        x = self.norm(x)
        out[-1] = x
        return out

    def prepare_features_for_image_model(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        effective_time_dim = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]
        for x in features:
            x_no_token = x[:, 1:, :]
            number_of_tokens = x_no_token.shape[1]
            tokens_per_timestep = number_of_tokens // effective_time_dim
            h = int(np.sqrt(tokens_per_timestep))
            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                e=self.embed_dim,
                t=effective_time_dim,
                h=h,
            )
            out.append(encoded)
        return out








def _get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor):
  """ This is the torch version of *get_1d_sincos_pos_embed_from_grid()*. However,
      it was modified to cast omega values to pos.dtype which must be float (and not int as in
      regular positional embeddings). This was required in order to allow for native FSDP mixed
      precision support: modify omega to appropriate dtype (pos carries the correct float dtype),
      instead of manually forcing float32.
      embed_dim: output dimension for each position
      pos: a list of positions to be encoded: size (M,) - must be float dtype!
      out: (M, D)
  """
  assert embed_dim % 2 == 0
  assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16]

  omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
  omega /= embed_dim / 2.0
  omega = 1.0 / 10000**omega  # (D/2,)

  pos = pos.reshape(-1)  # (M,)
  out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

  emb_sin = torch.sin(out)  # (M, D/2)
  emb_cos = torch.cos(out)  # (M, D/2)

  emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

  return emb












def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 3D sin/cos positional embeddings.
    Args:
        embed_dim (int):
            Embedding dimension.
        grid_size (tuple[int, int, int] | list[int]):
            The grid depth, height and width.
        add_cls_token (bool, *optional*, defaults to False):
            Whether or not to add a classification (CLS) token.
    Returns:
        (`torch.FloatTensor` of shape (grid_size[0]*grid_size[1]*grid_size[2], embed_dim) or
        (1+grid_size[0]*grid_size[1]*grid_size[2], embed_dim): the position embeddings (with or without cls token)
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed






def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb










def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)











class LocationEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = _get_1d_sincos_embed_from_grid_torch(
                self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = _get_1d_sincos_embed_from_grid_torch(
                self.lon_embed_dim, location_coords[:, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding  # B, 1, embed_dim
    










class PatchEmbed(nn.Module):
    """3D version of timm.models.vision_transformer.PatchEmbed"""
    def __init__(
            self,
            input_size: Tuple[int, int, int] = (1, 224, 224),
            patch_size: Tuple[int, int, int] = (1, 16, 16),
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: nn.Module | None = None,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape

        if T / self.patch_size[0] % 1 or H / self.patch_size[1] % 1 or W / self.patch_size[2] % 1:
            logging.warning(f"Input {x.shape[-3:]} is not divisible by patch size {self.patch_size}."
                            f"The border will be ignored, add backbone_padding for pixel-wise tasks.")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x
    













class Neck(ABC, nn.Module):
    """Base class for Neck

    A neck must must implement `self.process_channel_list` which returns the new channel list.
    """

    def __init__(self, channel_list: list[int]) -> None:
        super().__init__()
        self.channel_list = channel_list

    @abstractmethod
    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return channel_list

    @abstractmethod
    def forward(self, channel_list: list[torch.Tensor], **kwargs) -> list[torch.Tensor]: ...

class ReshapeTokensToImage(Neck):
    def __init__(self, channel_list: list[int], remove_cls_token=True, effective_time_dim: int = 1, h: int | None = None    ):
        super().__init__(channel_list)
        self.remove_cls_token = remove_cls_token
        self.effective_time_dim = effective_time_dim
        self.h = h

    def forward(self, features: list[torch.Tensor], image_size=None, **kwargs) -> list[torch.Tensor]:
        out = []
        for x in features:
            x_no_token = x[:, 1:, :] if self.remove_cls_token else x
            x_no_token = x_no_token.reshape(x.shape[0], -1, x.shape[-1])
            number_of_tokens = x_no_token.shape[1]
            tokens_per_timestep = number_of_tokens // self.effective_time_dim

            # Assume square images first
            h = self.h or math.sqrt(tokens_per_timestep)
            if h - int(h) == 0:
                h = int(h)

            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                batch=x_no_token.shape[0],
                t=self.effective_time_dim,
                h=h,
            )

            out.append(encoded)
        return out

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return super().process_channel_list(channel_list)
    
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=0,
                 dilation=1, stride=1,
                 inplace=False, transpose=False,
                 scale_factor=None) -> None:

        super().__init__()

        if transpose:
            kind = "Transpose"
        else:
            kind = ""

        conv_name = f"Conv{kind}2d"

        if transpose:

            stride = scale_factor
            padding = (kernel_size - scale_factor) // 2

        conv_template = getattr(nn, conv_name)
        self.conv= conv_template(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):
  
        return self.act(self.norm(self.conv(x)))
    
class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        """Constructor

        Args:
            pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
                Module.
            in_channels (int): Input channels.
            channels (int): Channels after modules, before conv_seg.
            align_corners (bool): align_corners argument of F.interpolate.
        """
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1, inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
    
class UperNetDecoder(nn.Module):
    """UperNetDecoder. Adapted from MMSegmentation."""

    def __init__(
        self,
        embed_dim: list[int],
        pool_scales: tuple[int] = (1, 2, 3, 6),
        channels: int = 256,
        align_corners: bool = True,  # noqa: FBT001, FBT002
        scale_modules: bool = False,
    ):
        """Constructor

        Args:
            embed_dim (list[int]): Input embedding dimension for each input.
            pool_scales (tuple[int], optional): Pooling scales used in Pooling Pyramid
                Module applied on the last feature. Default: (1, 2, 3, 6).
            channels (int, optional): Channels used in the decoder. Defaults to 256.
            align_corners (bool, optional): Wheter to align corners in rescaling. Defaults to True.
            scale_modules (bool, optional): Whether to apply scale modules to the inputs. Needed for plain ViT.
                Defaults to False.
        """
        super().__init__()

        self.scale_modules = scale_modules
        if scale_modules:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim[0],
                                embed_dim[0] // 2, 2, 2),
                nn.BatchNorm2d(embed_dim[0] // 2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim[0] // 2,
                                embed_dim[0] // 4, 2, 2))
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim[1],
                                embed_dim[1] // 2, 2, 2))
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.embed_dim = [embed_dim[0] // 4, embed_dim[1] // 2, embed_dim[2], embed_dim[3]]
        else:
            self.embed_dim = embed_dim

        self.out_channels = channels
        self.channels = channels
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.embed_dim[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.embed_dim[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1, inplace=True
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for embed_dim in self.embed_dim[:-1]:  # skip the top layer
            l_conv = ConvModule(
                embed_dim,
                self.channels,
                1,
                inplace=False,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(len(self.embed_dim) * self.channels, self.channels, 3, padding=1, inplace=True)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """

        if self.scale_modules:
            scaled_inputs = []
            scaled_inputs.append(self.fpn1(inputs[0]))
            scaled_inputs.append(self.fpn2(inputs[1]))
            scaled_inputs.append(self.fpn3(inputs[2]))
            scaled_inputs.append(self.fpn4(inputs[3]))
            inputs = scaled_inputs
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = torch.nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats
    











def otsu_thresholding(band, t=None):
    
    if t is not None:
        binary_mask = (band < t).float()
        return binary_mask
    
    VH_flat = band.flatten()
    hist, bin_edges = torch.histogram(VH_flat, bins=256, range=(float(VH_flat.min()), float(VH_flat.max())))
    hist = hist.float() / hist.sum()
    
    cumulative_sum = torch.cumsum(hist, dim=0)
    cumulative_mean = torch.cumsum(hist * torch.arange(256, dtype=torch.float32), dim=0)
    global_mean = cumulative_mean[-1]
    
    between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum) + 1e-6)
    optimal_threshold = torch.argmax(between_class_variance).item()
    t = bin_edges[optimal_threshold]
    binary_mask = (band < t).float()

    return binary_mask, t