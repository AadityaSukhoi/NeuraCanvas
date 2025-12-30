import os
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
import traceback

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN not found. Set it in your .env file.")
    st.stop()

# -----------------------------
# CUDA CHECK
# -----------------------------
if not torch.cuda.is_available():
    st.error("CUDA not available. Please install CUDA-enabled PyTorch.")
    st.markdown(
        "Install with:\n"
        "`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`"
    )
    st.stop()

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda"

        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
        st.write(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=HF_TOKEN,
            safety_checker=None,
            requires_safety_checking=False,
        )

        pipe = pipe.to(device)

        # Performance optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        if not next(pipe.unet.parameters()).is_cuda:
            raise RuntimeError("Model did not move to GPU")

        return pipe, device

    except Exception as e:
        st.error("Model loading failed")
        st.code(traceback.format_exc())
        return None, None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="NeuraCanvas", layout="centered")
st.title("🎨 NeuraCanvas")
st.markdown("Text → Image using Stable Diffusion (GPU powered)")

with st.spinner("Loading model (first run may take time)..."):
    pipe, device = load_model()

if pipe is None:
    st.stop()

st.success(f"Running on {device.upper()}")

prompt = st.text_input(
    "Prompt",
    "A cinematic sunset over the mountains, ultra realistic, 4k",
)

with st.expander("Advanced Options"):
    width = st.select_slider("Width", [256, 384, 512], value=512)
    height = st.select_slider("Height", [256, 384, 512], value=512)
    steps = st.slider("Inference Steps", 20, 50, 30)
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
    seed = st.number_input("Seed (-1 = random)", -1, 2_147_483_647, -1)

# -----------------------------
# GENERATION
# -----------------------------
if st.button("🚀 Generate Image"):
    try:
        torch.cuda.empty_cache()

        generator = None
        if seed != -1:
            generator = torch.Generator(device=device).manual_seed(seed)

        with st.spinner("Generating image..."):
            with torch.cuda.amp.autocast():
                result = pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )

        image = result.images[0]
        st.image(image, caption=prompt, use_column_width=True)

        # Download
        image_path = "generated_image.jpg"
        image.save(image_path)

        with open(image_path, "rb") as f:
            st.download_button(
                "Download Image",
                data=f,
                file_name="generated_image.jpg",
                mime="image/jpeg",
            )

        os.remove(image_path)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error("CUDA OOM. Reduce resolution or restart app.")
        else:
            st.error("Runtime error occurred")
            st.code(traceback.format_exc())

    except Exception:
        st.error("Unexpected error occurred")
        st.code(traceback.format_exc())