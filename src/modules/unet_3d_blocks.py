# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.utils import deprecate, is_torch_version, logging
from diffusers.models.attention import Attention
from diffusers.models.resnet import (
    Downsample2D,
    ResnetBlock2D,
    SpatioTemporalResBlock,
    TemporalConvLayer,
)
from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.transformer_2d import Transformer2DModel
from .transformer_temporal import (
    TransformerSpatioTemporalModel,
    TransformerTemporalModel,
    TransformerFullTemporalModel,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    num_attention_heads: int,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = True,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    temporal_num_attention_heads: int = 8,
    temporal_max_seq_length: int = 32,
    transformer_layers_per_block: int = 1,
) -> Union[
    "DownBlockSpatioTemporal",
    "CrossAttnDownBlockSpatioTemporal",
]:
    if down_block_type == "DownBlockSpatioTemporal":
        # added for SDV
        return DownBlockSpatioTemporal(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "CrossAttnDownBlockSpatioTemporal":
        # added for SDV
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlockSpatioTemporal")
        return CrossAttnDownBlockSpatioTemporal(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            add_downsample=add_downsample,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
        )

    raise ValueError(f"{down_block_type} does not exist.")


class UNetMidBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # there is always at least one resnet
        resnets = [
            SpatioTemporalResBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=1e-5,
            )
        ]
        attentions = []
        fullattentions = []

        for i in range(num_layers):
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )

            fullattentions.append(
                TransformerFullTemporalModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=None,
                )
            )

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )

        self.fullattentions = nn.ModuleList(fullattentions)
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.resnets[0](
            hidden_states,
            temb,
            image_only_indicator=image_only_indicator,
        )

        for attn, fullattn, resnet in zip(self.attentions, self.fullattentions, self.resnets[1:]):
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states_0 = hidden_states
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states_0 = fullattn(
                    hidden_states_0,
                    encoder_hidden_states=None,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states += hidden_states_0

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
            else:
                hidden_states_0 = hidden_states
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states_0 = fullattn(
                    hidden_states_0,
                    encoder_hidden_states=None,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states += hidden_states_0

                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

        return hidden_states


class DownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                    )
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        fullattentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-6,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )
            fullattentions.append(
                TransformerFullTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=None,
                )
            )

        self.fullattentions = nn.ModuleList(fullattentions)
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=1,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions, self.fullattentions))
        for resnet, attn, fullattn in blocks:
            if self.training and self.gradient_checkpointing:  # TODO

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )

                hidden_states_0 = hidden_states
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states_0 = fullattn(
                    hidden_states_0,
                    encoder_hidden_states=None,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states += hidden_states_0
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states_0 = hidden_states
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states_0 = fullattn(
                    hidden_states_0,
                    encoder_hidden_states=None,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                hidden_states += hidden_states_0

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

