"""
Z-Image Character Consistency Studio
-------------------------------------
A Streamlit app to compare a base Z-Image model against a trained
character LoRA, side by side, across multiple seeds, so you can
visually judge how consistent your trained character looks.

Run with:
    streamlit run app.py

Requirements (install first):
    pip install streamlit diffusers torch accelerate safetensors pillow --break-system-packages

Notes:
- Requires a CUDA GPU (the pipeline is moved to "cuda").
- The base model is loaded once and cached across reruns via
  st.cache_resource, so clicking "Generate" repeatedly does not
  reload multi-GB weights each time.
- The LoRA is loaded as a named adapter and toggled on/off
  (enable_lora/disable_lora) rather than reloading the model,
  which is much faster than instantiating two separate pipelines.
"""

import io

import streamlit as st
import torch
from diffusers import ZImagePipeline

ADAPTER_NAME = "character_lora"

st.set_page_config(page_title="Z-Image Character Consistency Studio", layout="wide")
st.title("🎨 Z-Image Character Consistency Studio")
st.caption(
    "Load your base Z-Image model and a trained character LoRA, "
    "then generate matched pairs across several seeds to check consistency."
)

# ----------------------------------------------------------------------
# Sidebar — model paths + generation settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("1. Model paths")
    model_path = st.text_input("/media/avidmech/data/Z_image_model_image_generation/Z-Image", value="./Z-Image")
    lora_path = st.text_input("/media/avidmech/data/Z_image_model_image_generation/trained-z-image-lora_for_brad_pitt_chracter", value="")

    col_a, col_b = st.columns(2)
    with col_a:
        load_base_btn = st.button("Load base model", use_container_width=True)
    with col_b:
        load_lora_btn = st.button("Load / reload LoRA", use_container_width=True)

    unload_btn = st.button("🗑️ Unload everything (free VRAM)", use_container_width=True)

    st.divider()
    st.header("2. Generation settings")
    height = st.select_slider("Height", options=[512, 768, 1024, 1280], value=1024)
    width = st.select_slider("Width", options=[512, 768, 1024, 1280], value=1024)
    steps = st.slider("Inference steps", 4, 60, 50)
    guidance = st.slider("Guidance scale", 0.0, 10.0, 5.0, step=0.5)
    lora_strength = st.slider("LoRA strength", 0.0, 1.5, 1.0, step=0.05)

    st.divider()
    st.header("3. Seeds")
    seed_mode = st.radio("Seed mode", ["Fixed sequence", "Random"], horizontal=True)
    base_seed = st.number_input("Starting seed", value=42, step=1)
    num_variations = st.slider("Number of seeds to test", 1, 6, 3)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "pipe" not in st.session_state:
    st.session_state.pipe = None
if "lora_loaded" not in st.session_state:
    st.session_state.lora_loaded = False
if "history" not in st.session_state:
    st.session_state.history = []  # list of (label, seed, png_bytes)


@st.cache_resource(show_spinner=False)
def _load_pipeline(path: str):
    pipe = ZImagePipeline.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    return pipe


def load_base_model(path: str):
    with st.spinner("Loading base Z-Image pipeline onto GPU..."):
        st.session_state.pipe = _load_pipeline(path)
        st.session_state.lora_loaded = False
    st.success("Base model loaded.")


def load_lora(path: str):
    if st.session_state.pipe is None:
        st.error("Load the base model first.")
        return
    if not path.strip():
        st.error("Enter a path to your LoRA .safetensors file first.")
        return
    with st.spinner("Loading LoRA weights..."):
        # Remove any previously loaded adapter with the same name so
        # reloading a new LoRA file doesn't collide with the old one.
        try:
            st.session_state.pipe.delete_adapters(ADAPTER_NAME)
        except Exception:
            pass
        st.session_state.pipe.load_lora_weights(path, adapter_name=ADAPTER_NAME)
    st.session_state.lora_loaded = True
    st.success(f"LoRA loaded as adapter '{ADAPTER_NAME}'.")


def unload_everything():
    if st.session_state.pipe is not None:
        try:
            st.session_state.pipe.to("cpu")
        except Exception:
            pass
        del st.session_state.pipe
    st.session_state.pipe = None
    st.session_state.lora_loaded = False
    _load_pipeline.clear()
    torch.cuda.empty_cache()
    st.success("Model unloaded, VRAM freed.")


if load_base_btn:
    load_base_model(model_path)
if load_lora_btn:
    load_lora(lora_path)
if unload_btn:
    unload_everything()

# ----------------------------------------------------------------------
# Status banner
# ----------------------------------------------------------------------
if st.session_state.pipe is None:
    st.warning("No model loaded yet — click **Load base model** in the sidebar.")
elif not st.session_state.lora_loaded:
    st.info("Base model loaded. LoRA not loaded yet — you can still generate base-only images.")
else:
    st.success("Base model + LoRA ready — generation will show both side by side.")

# ----------------------------------------------------------------------
# Prompt input
# ----------------------------------------------------------------------
prompt = st.text_area("Prompt", height=100, placeholder="Describe the character and the scene...")
negative_prompt = st.text_input("Negative prompt (optional)", value="")

generate_btn = st.button("🚀 Generate comparison", type="primary", use_container_width=True)


def make_generator(seed: int):
    return torch.Generator("cuda").manual_seed(int(seed))


def generate_image(pipe, seed: int, use_lora: bool):
    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=make_generator(seed),
    )
    if negative_prompt.strip():
        kwargs["negative_prompt"] = negative_prompt

    if use_lora and st.session_state.lora_loaded:
        pipe.set_adapters([ADAPTER_NAME], adapter_weights=[lora_strength])
        pipe.enable_lora()
    else:
        try:
            pipe.disable_lora()
        except Exception:
            pass

    return pipe(**kwargs).images[0]


def img_to_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------
# Generation loop
# ----------------------------------------------------------------------
if generate_btn:
    if not prompt.strip():
        st.error("Enter a prompt first.")
    elif st.session_state.pipe is None:
        st.error("Load the base model first.")
    else:
        if seed_mode == "Fixed sequence":
            seeds = [int(base_seed) + i for i in range(num_variations)]
        else:
            seeds = [int(torch.seed() % (2**31 - 1)) for _ in range(num_variations)]

        for seed in seeds:
            st.markdown(f"**Seed: {seed}**")
            c1, c2 = st.columns(2)

            with c1:
                st.caption("Base model (no LoRA)")
                with st.spinner("Generating base image..."):
                    img_base = generate_image(st.session_state.pipe, seed, use_lora=False)
                st.image(img_base, use_container_width=True)
                base_bytes = img_to_bytes(img_base)
                st.download_button(
                    "Download", base_bytes, file_name=f"base_seed{seed}.png",
                    mime="image/png", key=f"dl_base_{seed}",
                )
                st.session_state.history.append(("base", seed, base_bytes))

            with c2:
                if st.session_state.lora_loaded:
                    st.caption("With character LoRA")
                    with st.spinner("Generating LoRA image..."):
                        img_lora = generate_image(st.session_state.pipe, seed, use_lora=True)
                    st.image(img_lora, use_container_width=True)
                    lora_bytes = img_to_bytes(img_lora)
                    st.download_button(
                        "Download", lora_bytes, file_name=f"lora_seed{seed}.png",
                        mime="image/png", key=f"dl_lora_{seed}",
                    )
                    st.session_state.history.append(("lora", seed, lora_bytes))
                else:
                    st.info("Load a LoRA in the sidebar to see this comparison.")

            st.divider()

# ----------------------------------------------------------------------
# Session history
# ----------------------------------------------------------------------
if st.session_state.history:
    with st.expander(f"📜 Session history ({len(st.session_state.history)} images)"):
        cols = st.columns(4)
        for i, (kind, seed, img_bytes) in enumerate(st.session_state.history):
            with cols[i % 4]:
                st.image(img_bytes, caption=f"{kind} · seed {seed}", use_container_width=True)
