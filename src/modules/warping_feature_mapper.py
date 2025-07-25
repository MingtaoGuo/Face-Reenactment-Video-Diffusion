from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging

from .unet_3d_blocks import get_down_block, UNetMidBlockSpatioTemporal
from src.modules.unet import UNetSpatioTemporalConditionModel
from src.modules.internal_feature_modulator import InternalFeatureModulator

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class WarpingFeatureMapperOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class WarpingFeatureMapper(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state,
    and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", 
            "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", 
            "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`], 
            [`~models.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 8,
            out_channels: int = 4,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlockSpatioTemporal",
                    "CrossAttnDownBlockSpatioTemporal",
                    "CrossAttnDownBlockSpatioTemporal",
                    "DownBlockSpatioTemporal",
            ),
            up_block_types: Tuple[str] = (
                    "UpBlockSpatioTemporal",
                    "CrossAttnUpBlockSpatioTemporal",
                    "CrossAttnUpBlockSpatioTemporal",
                    "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            addition_time_embed_dim: int = 256,
            projection_class_embeddings_input_dim: int = 768,
            layers_per_block: Union[int, Tuple[int]] = 2,
            cross_attention_dim: Union[int, Tuple[int]] = 1024,
            transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
            num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
            num_frames: int = 25,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. " \
                f"`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. " \
                f"`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. " \
                f"`num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. " \
                f"`cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. " \
                f"`layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.warping_feature_mapper_down_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        warping_feature_mapper_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        warping_feature_mapper_block = zero_module(warping_feature_mapper_block)
        self.warping_feature_mapper_down_blocks.append(warping_feature_mapper_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block[i]):
                warping_feature_mapper_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                warping_feature_mapper_block = zero_module(warping_feature_mapper_block)
                self.warping_feature_mapper_down_blocks.append(warping_feature_mapper_block)

            if not is_final_block:
                warping_feature_mapper_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                warping_feature_mapper_block = zero_module(warping_feature_mapper_block)
                self.warping_feature_mapper_down_blocks.append(warping_feature_mapper_block)

        # mid
        mid_block_channel = block_out_channels[-1]
        warping_feature_mapper_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        warping_feature_mapper_block = zero_module(warping_feature_mapper_block)
        self.warping_feature_mapper_mid_block = warping_feature_mapper_block

        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        self.fusion_block1 = InternalFeatureModulator(1280, 1280)
        self.fusion_block2 = InternalFeatureModulator(640, 1280)
        self.fusion_block3 = InternalFeatureModulator(320, 640)
        self.fusion_proj = zero_module(nn.Conv2d(320, out_channels, kernel_size=1))
        # count how many layers upsample the images
        self.num_upsamplers = 0

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
                name: str,
                module: torch.nn.Module,
                processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` " \
                f"when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @classmethod
    def from_unet(
        cls,
        unet: UNetSpatioTemporalConditionModel,
        load_weights_from_unet: bool = True,
    ):
        r"""
        Instantiate a [`WarpingFeatureMapper`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`WarpingFeatureMapper`]. All configuration options are also copied
                where applicable.
        """
        warping_feature_mapper = cls(
            sample_size=unet.config.sample_size,
            in_channels=unet.config.in_channels,
            out_channels=unet.config.out_channels,
            down_block_types=unet.config.down_block_types,
            up_block_types=unet.config.up_block_types,
            block_out_channels=unet.config.block_out_channels,
            addition_time_embed_dim=unet.config.addition_time_embed_dim,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            layers_per_block=unet.config.layers_per_block,
            cross_attention_dim=unet.config.cross_attention_dim,
            transformer_layers_per_block=unet.config.transformer_layers_per_block,
            num_attention_heads=unet.config.num_attention_heads,
            num_frames=unet.config.num_frames,
        )

        if load_weights_from_unet:
            warping_feature_mapper.conv_in.load_state_dict(unet.conv_in.state_dict())
            warping_feature_mapper.time_proj.load_state_dict(unet.time_proj.state_dict())
            warping_feature_mapper.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            warping_feature_mapper.add_time_proj.load_state_dict(unet.add_time_proj.state_dict())
            warping_feature_mapper.add_embedding.load_state_dict(unet.add_embedding.state_dict())

            if hasattr(warping_feature_mapper, "add_embedding"):
                warping_feature_mapper.add_embedding.load_state_dict(unet.add_embedding.state_dict())


            down_blocks_statedict = unet.down_blocks.state_dict()
            for k, v, in warping_feature_mapper.down_blocks.state_dict().items():
                if "fullattentions" in k:
                    try:
                        down_blocks_statedict[k] = unet.down_blocks.state_dict()[k.replace("fullattentions", "attentions")]
                    except:
                        continue

            mid_blocks_statedict = unet.mid_block.state_dict()
            for k, v, in warping_feature_mapper.mid_block.state_dict().items():
                if "fullattentions" in k:
                    try:
                        mid_blocks_statedict[k] = unet.mid_block.state_dict()[k.replace("fullattentions", "attentions")]
                    except:
                        continue

            down_miss_key = warping_feature_mapper.down_blocks.load_state_dict(down_blocks_statedict, strict=False)
            mid_miss_key = warping_feature_mapper.mid_block.load_state_dict(mid_blocks_statedict, strict=False)
            
            print(down_miss_key)
            print(mid_miss_key)
        return warping_feature_mapper

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            added_time_ids: torch.Tensor,
            head_latents: torch.Tensor = None,
            image_only_indicator: bool = False,
            return_dict: bool = True,
            conditioning_scale: float = 1.0,
    ) -> Union[WarpingFeatureMapperOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            pose_latents: (`torch.FloatTensor`):
                The additional latents for pose sequences.
            image_only_indicator (`bool`, *optional*, defaults to `False`):
                Whether or not training with all images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] 
                instead of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, 
                an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is returned, 
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)
        if head_latents is not None:
            sample = sample + head_latents

        image_only_indicator = torch.ones(batch_size, num_frames, dtype=sample.dtype, device=sample.device) \
            if image_only_indicator else torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        lateral_down_block_res_samples = down_block_res_samples
        # 5. WarpingFeatureMapper net blocks
        warping_feature_mapper_down_block_res_samples = ()

        for down_block_res_sample, warping_feature_mapper_block in zip(down_block_res_samples, self.warping_feature_mapper_down_blocks):
            down_block_res_sample = warping_feature_mapper_block(down_block_res_sample)
            warping_feature_mapper_down_block_res_samples = warping_feature_mapper_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = warping_feature_mapper_down_block_res_samples

        mid_block_res_sample = self.warping_feature_mapper_mid_block(sample)
        
        # 6. scaling
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale
        
        lateral_1 = lateral_down_block_res_samples[8]
        sample = self.fusion_block1(lateral_1, sample)
        lateral_2 = lateral_down_block_res_samples[5]
        sample = self.fusion_block2(lateral_2, sample)
        lateral_3 = lateral_down_block_res_samples[2]
        sample = self.fusion_block3(lateral_3, sample)
        sample = self.fusion_proj(sample)
        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample, sample)

        return WarpingFeatureMapperOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample, sample=sample
        )

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module