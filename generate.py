"""
Image & Video Generator
Generate images with Flux.1-schnell or Flux.1-dev, then animate with LTX-2 Distilled.
"""

import gc
import io
import os
import tempfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import gradio as gr
from PIL import Image
from diffusers.utils import export_to_video

# ─────────────────────────────────────────────
# Availability checks
# ─────────────────────────────────────────────

try:
    from diffusers import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False

try:
    from diffusers import LTX2ImageToVideoPipeline
    LTX2_AVAILABLE = True
except ImportError:
    LTX2_AVAILABLE = False

# LTX-2 Distilled uses the ltx-pipelines package (not diffusers)
try:
    from ltx_pipelines.distilled import DistilledPipeline  # noqa: F401
    LTX2_DISTILLED_PKG_AVAILABLE = True
except ImportError:
    LTX2_DISTILLED_PKG_AVAILABLE = False

# ─────────────────────────────────────────────
# Config — identical to app.py
# ─────────────────────────────────────────────

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

FLUX_SCHNELL_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
FLUX_DEV_MODEL_ID     = "black-forest-labs/FLUX.1-dev"
LTX2_DISTILLED_MODEL_ID = "Lightricks/LTX-2"

IMAGE_MODEL_CONFIGS = {
    "Flux.1-schnell (Fast)": {
        "manager": "flux_schnell",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
    },
    "Flux.1-dev (Quality)": {
        "manager": "flux_dev",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
    },
}

# LTX-2 Distilled video config — identical to app.py VIDEO_MODEL_CONFIGS entry
LTX2_DISTILLED_CONFIG = {
    "model_id": LTX2_DISTILLED_MODEL_ID,
    "num_frames": 121,
    "fps": 24,
    "min_frames": 9,
    "max_frames": 481,
    "frame_step": 8,
    "frame_offset": 1,
    "width":  {"16:9": 1280, "9:16": 704},
    "height": {"16:9": 704,  "9:16": 1280},
    "dtype": "bfloat16",
}

# GPU mode detection — identical to app.py
USE_FULL_GPU = False
if torch.cuda.is_available():
    _gpu_props = torch.cuda.get_device_properties(0)
    _vram_gb = _gpu_props.total_memory / (1024 ** 3)
    if _vram_gb >= 40:
        USE_FULL_GPU = True
        print(f"🎮 {_gpu_props.name} — {_vram_gb:.1f} GB VRAM, full GPU mode")
    else:
        print(f"🎮 {_gpu_props.name} — {_vram_gb:.1f} GB VRAM, CPU-offload mode")

# ─────────────────────────────────────────────
# Image model managers — copied verbatim from app.py
# ─────────────────────────────────────────────

class FluxSchnellModelManager:
    """Persistent Flux.1-schnell pipeline manager"""
    _pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            if not FLUX_AVAILABLE:
                raise RuntimeError("FluxPipeline not available. Install a compatible version of diffusers.")
            print("🚀 Loading Flux.1-schnell...")
            cls._pipeline = FluxPipeline.from_pretrained(
                FLUX_SCHNELL_MODEL_ID,
                torch_dtype=torch.bfloat16,
            )
            cls._pipeline.enable_model_cpu_offload()
            print("   ✅ Flux.1-schnell loaded")
        return cls._pipeline

    @classmethod
    def unload(cls):
        if cls._pipeline is not None:
            del cls._pipeline
            cls._pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print("🗑️ Flux.1-schnell pipeline unloaded")


class FluxDevModelManager:
    """Persistent Flux.1-dev pipeline manager"""
    _pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            if not FLUX_AVAILABLE:
                raise RuntimeError("FluxPipeline not available. Install a compatible version of diffusers.")
            hf_token = os.environ.get("HF_TOKEN") or True  # True = use cached token
            print("🚀 Loading Flux.1-dev...")
            try:
                cls._pipeline = FluxPipeline.from_pretrained(
                    FLUX_DEV_MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    token=hf_token,
                )
            except Exception as e:
                cls._pipeline = None
                if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
                    raise RuntimeError(
                        "Flux.1-dev is a gated model. To authenticate:\n"
                        "  Option A: Run `huggingface-cli login` (persistent)\n"
                        "  Option B: export HF_TOKEN=hf_... then restart\n"
                        "  Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev"
                    ) from e
                raise
            cls._pipeline.enable_model_cpu_offload()
            print("   ✅ Flux.1-dev loaded")
        return cls._pipeline

    @classmethod
    def unload(cls):
        if cls._pipeline is not None:
            del cls._pipeline
            cls._pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print("🗑️ Flux.1-dev pipeline unloaded")


# ─────────────────────────────────────────────
# LTX-2 Distilled helpers — copied verbatim from app.py
# ─────────────────────────────────────────────

def _amend_forward_with_upcast(text_encoder):
    """Patch fp8 Gemma to upcast to bfloat16 for stable attention."""
    import types, torch
    orig_forward = text_encoder.forward

    def _patched_forward(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return orig_forward(*args, **kwargs)

    text_encoder.forward = types.MethodType(_patched_forward, text_encoder)


def _load_fp8_gemma_encoder(fp8_path, gemma_root, device, registry):
    """Load the fp8 Gemma text encoder from a safetensors file."""
    from ltx_core.loader.registry import StateDictRegistry
    from ltx_pipelines.utils import ModelLedger
    from ltx_pipelines.utils.types import PipelineComponents

    dummy_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=fp8_path,
        gemma_root_path=gemma_root,
        spatial_upsampler_path=None,
        loras=(),
        registry=registry,
    )
    text_encoder = dummy_ledger.text_encoder()
    _amend_forward_with_upcast(text_encoder)
    return text_encoder.to(device).eval()


class _MemoryEfficientDistilledPipeline:
    """
    Two-stage distilled pipeline using fp8 Gemma text encoder (~12 GB VRAM) that is
    loaded, used, and freed before loading the fp8 transformer (~12 GB VRAM), keeping
    peak VRAM well within 48 GB on an A40.

    Copied verbatim from app.py.
    """
    def __init__(self, model_ledger, pipeline_components, device, fp8_gemma_path, gemma_root):
        self.model_ledger = model_ledger
        self.pipeline_components = pipeline_components
        self.device = device
        self.dtype = torch.bfloat16
        self.fp8_gemma_path = fp8_gemma_path
        self.gemma_root = gemma_root
        from ltx_core.loader.registry import StateDictRegistry
        self._gemma_registry = StateDictRegistry()

    @torch.inference_mode()
    def __call__(self, prompt, seed, height, width, num_frames, frame_rate, images,
                 tiling_config=None, enhance_prompt=False):
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import TilingConfig, decode_video as vae_decode_video
        from ltx_core.types import LatentState, VideoPixelShape
        from ltx_pipelines.utils import euler_denoising_loop
        from ltx_pipelines.utils.helpers import (
            assert_resolution, cleanup_memory, combined_image_conditionings,
            denoise_audio_video, simple_denoising_func,
        )
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

        assert_resolution(height=height, width=width, is_two_stage=True)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = self.dtype

        print("   📝 Loading fp8 Gemma text encoder...")
        text_encoder = _load_fp8_gemma_encoder(
            self.fp8_gemma_path, self.gemma_root, self.device, self._gemma_registry
        )
        if enhance_prompt:
            from ltx_pipelines.utils.helpers import generate_enhanced_prompt
            enhance_image = images[0][0] if images else None
            prompt = generate_enhanced_prompt(text_encoder, prompt, enhance_image, seed=seed)
        raw_output = text_encoder.encode(prompt)
        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()

        embeddings_processor = self.model_ledger.gemma_embeddings_processor()
        (ctx_p,) = [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in [raw_output]]
        del embeddings_processor
        cleanup_memory()

        video_context = ctx_p.video_encoding.to(self.device)
        audio_context = ctx_p.audio_encoding.to(self.device) if ctx_p.audio_encoding is not None else None

        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        stage_1_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate
        )
        stage_1_conditionings = combined_image_conditionings(
            images=images, height=stage_1_output_shape.height, width=stage_1_output_shape.width,
            video_encoder=video_encoder, dtype=dtype, device=self.device,
        )

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context, audio_context=audio_context, transformer=transformer
                ),
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape, conditionings=stage_1_conditionings, noiser=noiser,
            sigmas=stage_1_sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=self.pipeline_components, dtype=dtype, device=self.device,
        )

        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1], video_encoder=video_encoder,
            upsampler=self.model_ledger.spatial_upsampler()
        )
        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
        )
        video_encoder = self.model_ledger.video_encoder()
        stage_2_conditionings = combined_image_conditionings(
            images=images, height=stage_2_output_shape.height, width=stage_2_output_shape.width,
            video_encoder=video_encoder, dtype=dtype, device=self.device,
        )
        transformer = self.model_ledger.transformer()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape, conditionings=stage_2_conditionings, noiser=noiser,
            sigmas=stage_2_sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=self.pipeline_components, dtype=dtype, device=self.device,
            noise_scale=stage_2_sigmas[0], initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        tiling_config = tiling_config or TilingConfig()
        decoded_video = vae_decode_video(
            video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )
        return decoded_video, decoded_audio


class LTX2DistilledModelManager:
    """Persistent LTX-2 Distilled pipeline manager (fp8, memory-efficient) — copied from app.py"""
    _pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            try:
                from ltx_pipelines.distilled import DistilledPipeline
                from ltx_pipelines.utils import ModelLedger
                from ltx_core.loader.registry import StateDictRegistry
                from ltx_pipelines.utils.types import PipelineComponents
                from huggingface_hub import hf_hub_download, snapshot_download
            except ImportError as e:
                raise RuntimeError(f"ltx-pipelines not installed: {e}")

            print("📥 Preparing LTX-2 Distilled model files...")
            repo     = "Lightricks/LTX-2"
            fp8_repo = "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn"
            checkpoint   = hf_hub_download(repo, "ltx-2-19b-distilled-fp8.safetensors")
            upsampler    = hf_hub_download(repo, "ltx-2-spatial-upscaler-x2-1.0.safetensors")
            gemma_root   = snapshot_download(repo, allow_patterns=["text_encoder/*", "tokenizer/*"])
            fp8_gemma_path = hf_hub_download(fp8_repo, "gemma_3_12B_it_fp8_e4m3fn.safetensors")
            print("   ✅ Model files ready")

            print("🚀 Loading LTX-2 Distilled pipeline...")
            device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            registry = StateDictRegistry()
            model_ledger = ModelLedger(
                dtype=torch.bfloat16,
                device=device,
                checkpoint_path=checkpoint,
                gemma_root_path=None,
                spatial_upsampler_path=upsampler,
                loras=(),
                registry=registry,
            )
            pipeline_components = PipelineComponents(dtype=torch.bfloat16, device=device)
            cls._pipeline = _MemoryEfficientDistilledPipeline(
                model_ledger, pipeline_components, device, fp8_gemma_path, gemma_root
            )
            print("   ✅ LTX-2 Distilled loaded (fp8 Gemma, memory-efficient)")
        return cls._pipeline

    @classmethod
    def unload(cls):
        if cls._pipeline is not None:
            del cls._pipeline
            cls._pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print("🗑️ LTX-2 Distilled pipeline unloaded")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compress_image_for_ltx(img: Image.Image, quality: int = 85) -> Image.Image:
    """Apply JPEG compression to simulate video-frame artifacts for LTX-2.
    LTX-2 was trained on compressed video and uses compression cues for motion.
    Copied from app.py."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def compute_num_frames(duration: float, config: dict) -> int:
    """Snap frame count up to the nearest valid count so video >= target duration.
    Copied from app.py."""
    fps        = config["fps"]
    step       = config.get("frame_step", 1)
    offset     = config.get("frame_offset", 0)
    min_frames = config["min_frames"]
    max_frames = config["max_frames"]

    ideal = round(duration * fps)
    # Snap up to nearest (step * k + offset) >= ideal
    if step > 1:
        k = max(0, (ideal - offset + step - 1) // step)
        n = step * k + offset
    else:
        n = ideal

    return max(min_frames, min(n, max_frames))


# ─────────────────────────────────────────────
# Core generation functions
# ─────────────────────────────────────────────

def generate_image(prompt: str, model_name: str, width: int, height: int) -> tuple:
    """Generate an image using Flux.1-schnell or Flux.1-dev."""
    if not prompt.strip():
        return None, "Please enter a prompt."

    config      = IMAGE_MODEL_CONFIGS[model_name]
    manager_key = config["manager"]

    try:
        if manager_key == "flux_schnell":
            pipe = FluxSchnellModelManager.get_pipeline()
        else:
            pipe = FluxDevModelManager.get_pipeline()

        print(f"🎨 Generating image with {model_name} ({width}×{height})…")
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
        ).images[0]
        print("   ✅ Image generated")
        return image, "Image generated successfully."
    except Exception as e:
        print(f"   ❌ Image generation failed: {e}")
        return None, f"Error: {e}"


def generate_video(
    image_source,          # PIL Image from image generator output, or None
    uploaded_image,        # PIL Image from upload component, or None
    video_prompt: str,
    aspect_ratio: str,
    duration: float,
    seed: int,
) -> tuple:
    """Generate a video from an image using LTX-2 Distilled."""
    # Resolve source image (generated takes priority; fall back to upload)
    source = image_source if image_source is not None else uploaded_image
    if source is None:
        return None, "Please generate or upload an image first."

    if isinstance(source, dict):
        # Gradio can return {"composite": ..., "background": ...} for image editors
        source = source.get("composite") or source.get("background") or source

    img = source.convert("RGB") if hasattr(source, "convert") else Image.fromarray(source).convert("RGB")

    cfg  = LTX2_DISTILLED_CONFIG
    w    = cfg["width"][aspect_ratio]
    h    = cfg["height"][aspect_ratio]
    fps  = cfg["fps"]

    num_frames = compute_num_frames(duration, cfg)

    # Align resolution to 64 — DistilledPipeline requirement
    w64 = (w // 64) * 64
    h64 = (h // 64) * 64

    prompt = video_prompt.strip() or "natural movement, dynamic motion"

    try:
        from ltx_pipelines.utils.args import ImageConditioningInput

        pipe = LTX2DistilledModelManager.get_pipeline()

        # Save image to temp file — DistilledPipeline takes a file path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_img_path = tmp.name
        img.resize((w64, h64), Image.LANCZOS).save(tmp_img_path)

        print(f"🎥 Generating video with LTX-2 Distilled ({w64}×{h64}, {num_frames} frames @ {fps} fps)…")
        try:
            decoded_video, _ = pipe(
                prompt=prompt,
                seed=seed,
                height=h64,
                width=w64,
                num_frames=num_frames,
                frame_rate=fps,
                images=[ImageConditioningInput(path=tmp_img_path, frame_idx=0, strength=1.0)],
            )
            frames = []
            with torch.inference_mode():
                for chunk in decoded_video:
                    for frame_array in chunk.to("cpu").numpy():
                        frames.append(Image.fromarray(frame_array))
        finally:
            os.unlink(tmp_img_path)

        # Save to temp mp4
        out_path = tempfile.mktemp(suffix=".mp4")
        export_to_video(frames, out_path, fps=fps)
        print(f"   ✅ Video saved → {out_path}")
        return out_path, f"Video generated: {len(frames)} frames @ {fps} fps"

    except Exception as e:
        print(f"   ❌ Video generation failed: {e}")
        return None, f"Error: {e}"


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Image & Video Generator") as demo:
        gr.Markdown("# 🎨 Image & Video Generator")
        gr.Markdown(
            "Generate images with **Flux.1-schnell** or **Flux.1-dev**, "
            "then animate them with **LTX-2 Distilled**."
        )

        # ── Image generation ──────────────────────────────────────────────
        gr.Markdown("## Step 1 — Generate Image")
        with gr.Row():
            with gr.Column(scale=2):
                img_prompt = gr.Textbox(
                    label="Image Prompt",
                    placeholder="A cinematic shot of a glowing forest at dusk…",
                    lines=3,
                )
                img_model = gr.Radio(
                    choices=list(IMAGE_MODEL_CONFIGS.keys()),
                    value="Flux.1-schnell (Fast)",
                    label="Image Model",
                )
                with gr.Row():
                    img_width = gr.Slider(
                        minimum=512, maximum=2048, step=64, value=1024,
                        label="Width"
                    )
                    img_height = gr.Slider(
                        minimum=512, maximum=2048, step=64, value=1024,
                        label="Height"
                    )
                img_btn    = gr.Button("Generate Image", variant="primary")
                img_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                img_output = gr.Image(label="Generated Image", type="pil")

        # ── Video generation ──────────────────────────────────────────────
        gr.Markdown("## Step 2 — Generate Video")
        gr.Markdown(
            "Use the image generated above **or** upload your own image. "
            "The generated image takes priority if both are present."
        )
        with gr.Row():
            with gr.Column(scale=2):
                vid_upload = gr.Image(
                    label="Upload Image (optional — overridden by generated image)",
                    type="pil",
                )
                vid_prompt = gr.Textbox(
                    label="Video Prompt",
                    placeholder="Natural movement, gentle breeze, cinematic motion…",
                    lines=2,
                )
                with gr.Row():
                    vid_aspect = gr.Radio(
                        choices=["16:9", "9:16"],
                        value="16:9",
                        label="Aspect Ratio",
                    )
                    vid_duration = gr.Slider(
                        minimum=1.0, maximum=20.0, step=0.5, value=5.0,
                        label="Duration (seconds)",
                    )
                vid_seed = gr.Number(value=42, precision=0, label="Seed")
                vid_btn    = gr.Button("Generate Video", variant="primary")
                vid_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                vid_output = gr.Video(label="Generated Video")

        # ── Event wiring ──────────────────────────────────────────────────
        img_btn.click(
            fn=generate_image,
            inputs=[img_prompt, img_model, img_width, img_height],
            outputs=[img_output, img_status],
        )

        vid_btn.click(
            fn=generate_video,
            inputs=[img_output, vid_upload, vid_prompt, vid_aspect, vid_duration, vid_seed],
            outputs=[vid_output, vid_status],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
