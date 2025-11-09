# In src/patch_qwen.py

import torch
import torch.nn as nn
import math
import types
import functools
from typing import Any, Dict, List, Optional, Tuple, Union

from comfy.model_patcher import ModelPatcher
from comfy import model_sampling

# We will reuse the core mathematical engine from the FLUX implementation
from .rope import get_1d_rotary_pos_embed

# --- Step 1: Define the custom, DyPE-aware Qwen positional embedder ---
class QwenDyPEPosEmbed(nn.Module):
    """
    A replacement for QwenEmbedRope that injects DyPE (Dynamic Position Extrapolation) logic.
    It dynamically recalculates RoPE frequencies based on the diffusion timestep
    when the target resolution exceeds the base training resolution.
    """
    def __init__(self, original_pos_embed, method: str = 'yarn', dype: bool = True, dype_exponent: float = 2.0):
        super().__init__()
        
        # --- Copy essential attributes from the original embedder ---
        self.theta = original_pos_embed.theta
        self.axes_dim = original_pos_embed.axes_dim
        self.scale_rope = original_pos_embed.scale_rope
        # Keep the original pre-computed frequencies for the 'base' method or low resolutions
        self.pos_freqs = original_pos_embed.pos_freqs
        self.neg_freqs = original_pos_embed.neg_freqs

        # --- DyPE specific attributes ---
        self.method = method
        self.dype = dype if method != 'base' else False
        self.dype_exponent = dype_exponent
        self.current_timestep = 1.0
        
        # Qwen's base patch resolution (1024px image -> 64x64 latent patches)
        self.base_patches = 64

    def set_timestep(self, timestep: float):
        """Sets the current normalized timestep [0.0, 1.0] for DyPE calculations."""
        self.current_timestep = timestep

    def forward(
        self,
        video_fhw: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        txt_seq_lens: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method largely replicates the original logic but calls our modified
        _compute_video_freqs which contains the DyPE logic.
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        # In ComfyUI, this seems to be always a list of one tuple.
        if not isinstance(video_fhw, list):
             video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            # Call our new, DyPE-aware frequency computation method
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens) if txt_seq_lens else 0
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        """
        The core of the DyPE injection. This method dynamically computes frequencies
        if DyPE is enabled and resolution is high, otherwise it falls back to the
        original static frequency lookup.
        
        NOTE: The lru_cache has been removed because the output now depends on
        `self.current_timestep`, which is not an argument. Caching would lead to
        stale results during the denoising process.
        """
        # Check if we need to use dynamic position extrapolation
        use_dype_logic = self.dype and (height > self.base_patches or width > self.base_patches)

        if not use_dype_logic:
            # --- Original Static Logic (for low-res or DyPE disabled) ---
            seq_lens = frame * height * width
            freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
            freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

            freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
            if self.scale_rope:
                freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
                freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
            else:
                freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            
            freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
            return freqs.clone().contiguous()
        else:
            # --- New Dynamic DyPE Logic (for high-res with DyPE enabled) ---
            seq_lens = frame * height * width
            device = self.pos_freqs.device
            freqs_dtype = torch.bfloat16 # A good default for performance

            # Common arguments for our DyPE math function
            dype_kwargs = {'dype': True, 'current_timestep': self.current_timestep, 'dype_exponent': self.dype_exponent}
            common_kwargs = {'theta': self.theta, 'use_real': False, 'freqs_dtype': freqs_dtype, **dype_kwargs}

            # 1. Frame axis (usually size 1 for images, no scaling needed)
            pos_frame = torch.arange(idx, idx + frame, device=device)
            freqs_frame = get_1d_rotary_pos_embed(dim=self.axes_dim[0], pos=pos_frame, **common_kwargs)
            freqs_frame = freqs_frame.view(frame, 1, 1, -1).expand(frame, height, width, -1)

            # 2. Height axis (apply DyPE scaling)
            pos_height = torch.arange(height, device=device)
            if self.scale_rope: # Centered positions for 'scale_rope'
                pos_height = torch.cat([torch.arange(0, height // 2), torch.arange(-(height - height // 2), 0)])
            
            if self.method == 'yarn':
                freqs_height = get_1d_rotary_pos_embed(dim=self.axes_dim[1], pos=pos_height, yarn=True, max_pe_len=height, ori_max_pe_len=self.base_patches, **common_kwargs)
            elif self.method == 'ntk':
                ntk_scale = height / self.base_patches
                freqs_height = get_1d_rotary_pos_embed(dim=self.axes_dim[1], pos=pos_height, ntk_factor=ntk_scale, **common_kwargs)
            else: # base
                freqs_height = get_1d_rotary_pos_embed(dim=self.axes_dim[1], pos=pos_height, **common_kwargs)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)

            # 3. Width axis (apply DyPE scaling)
            pos_width = torch.arange(width, device=device)
            if self.scale_rope: # Centered positions for 'scale_rope'
                pos_width = torch.cat([torch.arange(0, width // 2), torch.arange(-(width - width // 2), 0)])

            if self.method == 'yarn':
                freqs_width = get_1d_rotary_pos_embed(dim=self.axes_dim[2], pos=pos_width, yarn=True, max_pe_len=width, ori_max_pe_len=self.base_patches, **common_kwargs)
            elif self.method == 'ntk':
                ntk_scale = width / self.base_patches
                freqs_width = get_1d_rotary_pos_embed(dim=self.axes_dim[2], pos=pos_width, ntk_factor=ntk_scale, **common_kwargs)
            else: # base
                freqs_width = get_1d_rotay_pos_embed(dim=self.axes_dim[2], pos=pos_width, **common_kwargs)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

            # Combine all axes
            freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
            return freqs.clone().contiguous()


# --- Step 2: Define the main patching function for Qwen-Image ---
def apply_dype_to_qwen(model: ModelPatcher, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float) -> ModelPatcher:
    """
    Applies DyPE to a Qwen-Image model by replacing its positional embedding module
    and wrapping its forward pass to inject the current timestep.
    """
    m = model.clone()
    
    # Qwen-Image does not require the noise schedule (sigma) patch that FLUX uses.
    # The primary benefit comes from patching the positional embeddings.

    try:
        # The path to Qwen's positional embedder is `diffusion_model.pos_embed`
        orig_embedder = m.model.diffusion_model.pos_embed
    except AttributeError:
        raise ValueError("The provided model is not a compatible Qwen-Image model. Could not find 'diffusion_model.pos_embed'.")

    # Create our new, DyPE-aware embedder using the original as a template
    new_pe_embedder = QwenDyPEPosEmbed(orig_embedder, method, enable_dype, dype_exponent)
    
    # Apply the patch, replacing the original embedder with our new one
    m.add_object_patch("diffusion_model.pos_embed", new_pe_embedder)
    
    # --- Step 3: Set up the wrapper to intercept the timestep ---
    sigma_max = m.model.model_sampling.sigma_max.item()

    def dype_wrapper_function(model_function, args_dict):
        # Check if DyPE is active for this generation
        if new_pe_embedder.dype:
            timestep_tensor = args_dict.get("timestep")
            if timestep_tensor is not None and timestep_tensor.numel() > 0:
                current_sigma = timestep_tensor.item()
                if sigma_max > 0:
                    normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                    # This call is what feeds the live data to our patched module
                    new_pe_embedder.set_timestep(normalized_timestep)
        
        # Continue the original model call.
        input_x = args_dict.get("input")
        timestep = args_dict.get("timestep")
        c = args_dict.get("c", {})
        return model_function(input_x, timestep, **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)
    
    return m