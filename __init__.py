import torch
from comfy_api.latest import ComfyExtension, io
from .src.patch_flux import apply_dype_to_flux

from .src.patch_qwen import apply_dype_to_qwen

class DyPE_Universal(io.ComfyNode):
    """
    Applies DyPE (Dynamic Position Extrapolation) to a FLUX model or Qwen.
    This allows generating images at resolutions far beyond the model's training scale
    by dynamically adjusting positional encodings and the noise schedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_Universal",
            display_name="DyPE Patcher (Universal)",
            category="model_patches/unet",
            description="Applies DyPE to a compatible model (FLUX, Qwen-Image) for ultra-high-resolution generation.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model (FLUX or Qwen-Image) to patch with DyPE.",
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image width. Must match the width of your empty latent."
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image height. Must match the height of your empty latent."
                ),
                io.Combo.Input(
                    "method",
                    options=["yarn", "ntk", "base"],
                    default="yarn",
                    tooltip="Position encoding extrapolation method (YARN recommended).",
                ),
                io.Boolean.Input(
                    "enable_dype",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable Dynamic Position Extrapolation for RoPE.",
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0, min=0.0, max=4.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE strength over time (λt). 2.0=Exponential (best for 4K+), 1.0=Linear, 0.5=Sub-linear (better for ~2K)."
                ),

                # Note: These noise schedule shifts are specific to the FLUX patch.
                # They will be ignored by the Qwen patcher if not implemented there.
                io.Float.Input(
                    "base_shift",
                    default=0.5, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="[FLUX Only] Advanced: Base shift for the noise schedule (mu). Default is 0.5."
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="[FLUX Only] Advanced: Max shift for the noise schedule (mu) at high resolutions. Default is 1.15."
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The model patched with DyPE.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float = 2.0, base_shift: float = 0.5, max_shift: float = 1.15) -> io.NodeOutput:
        """
        Clones the model, detects the model type, and applies the appropriate DyPE patch.
        """
        try:
            model_class_name = model.model.diffusion_model.__class__.__name__
        except Exception as e:
            raise ValueError(f"Could not identify the diffusion model class. Is this a valid ComfyUI model? Error: {e}")

        patched_model = None
        
        # --- THIS IS THE FIX ---
        # We now check if the class name is IN a tuple of known FLUX names.
        if model_class_name in ("FluxTransformer2DModel", "Flux"):
            print(f"ComfyUI-DyPE: Detected FLUX model ('{model_class_name}'). Applying FLUX patch.")
            patched_model = apply_dype_to_flux(model, width, height, method, enable_dype, dype_exponent, base_shift, max_shift)
        
        elif model_class_name == "QwenImageTransformer2DModel":
            print(f"ComfyUI-DyPE: Detected Qwen-Image model. Applying Qwen patch.")
            # This will call your function once it's implemented. For now, it will raise the NotImplementedError.
            patched_model = apply_dype_to_qwen(model, width, height, method, enable_dype, dype_exponent)
        
        else:
            raise TypeError(f"Unsupported model type for DyPE: '{model_class_name}'. This node supports 'FluxTransformer2DModel', 'Flux', and 'QwenImageTransformer2DModel'.")

        return io.NodeOutput(patched_model)

class DyPEExtension(ComfyExtension):
    """Registers the Universal DyPE node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_Universal]

async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()