import os
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from dotenv import load_dotenv
import traceback

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Please set your Hugging Face token in the .env file as HF_TOKEN=your_token_here")
    st.stop()

# Add CUDA availability check at startup
if not torch.cuda.is_available():
    st.error("CUDA is not available. Please check your PyTorch installation and GPU drivers.")
    st.markdown("""
    To fix this:
    1. Make sure you have an NVIDIA GPU
    2. Install NVIDIA drivers
    3. Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
    """)
    st.stop()

@st.cache_resource
def load_model():
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Force CUDA device
        device = "cuda"
        
        # Print GPU info for debugging
        st.write(f"GPU Found: {torch.cuda.get_device_name(0)}")
        st.write(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load model with explicit CUDA settings
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checking=False
        )
        
        # Force model to GPU
        pipe = pipe.to(device)
        
        # Enable optimizations
        pipe.enable_attention_slicing(1)
        pipe.enable_vae_slicing()
        
        # Verify model is on GPU
        if not next(pipe.unet.parameters()).is_cuda:
            raise RuntimeError("Model failed to move to GPU")
            
        return pipe, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

st.title("NeuraCanvas")
st.markdown("Enter a text prompt below and generate an image!")

# Add VRAM usage warning
st.warning("This application requires at least 4GB of VRAM. Using smaller image sizes will reduce VRAM usage.")

with st.spinner("Loading model... (this may take a few minutes on first run)"):
    pipe, device = load_model()
    
if pipe is None:
    st.error("Failed to load the model. Please check the error message above.")
    st.stop()

st.info(f"Running on: {device.upper()}")

prompt = st.text_input("Enter your prompt:", "A scenic view of a sunset over the mountains")

with st.expander("Advanced Options"):
    # Start with smaller default sizes to ensure GPU memory compatibility
    height = st.select_slider("Image Height", options=[256, 384, 512, 768], value=384)
    width = st.select_slider("Image Width", options=[256, 384, 512, 768], value=384)
    num_inference_steps = st.slider("Number of inference steps", 20, 50, 30)
    guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5)
    seed = st.number_input("Random seed (leave at -1 for random)", -1, 2147483647, -1)

if st.button("Generate Image"):
    try:
        with st.spinner("Generating image..."):
            # Clear CUDA cache before generation
            torch.cuda.empty_cache()
            
            generator = None if seed == -1 else torch.Generator(device=device).manual_seed(seed)
            
            # Generate with explicit GPU settings
            with torch.cuda.amp.autocast():
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width
                ).images[0]
            
            st.image(image, caption=f"Generated Image\nPrompt: {prompt}", use_column_width=True)
            
            # Add download button
            buf = image.convert('RGB').save('temp.jpg', 'JPEG', quality=90)
            with open('temp.jpg', 'rb') as f:
                image_data = f.read()
            os.remove('temp.jpg')
            
            st.download_button(
                label="Download Image",
                data=image_data,
                file_name="generated_image.jpg",
                mime="image/jpeg"
            )
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            st.error("GPU out of memory! Try using a smaller image size or clearing your GPU memory.")
            st.markdown("To clear GPU memory, please restart the application.")
        else:
            st.error(f"Error generating image: {str(e)}")
            st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        st.code(traceback.format_exc())