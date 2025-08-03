import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from diffusers import StableDiffusionPipeline
import torch

# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# === Load Scene Splitting LLM ===
scene_split_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Lightweight and fast

print("🔄 Loading scene splitting model...")
tokenizer = AutoTokenizer.from_pretrained(scene_split_model_id)
model = AutoModelForCausalLM.from_pretrained(
    scene_split_model_id, torch_dtype=dtype, device_map="auto"
)
llm_pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256
)
print("✅ Scene splitter loaded.")

# === Load Stable Diffusion ===
print("🔄 Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None,
)
pipe.to(device)
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()
print("✅ Stable Diffusion loaded.")

# === Scene Splitting via LLM ===
def split_story(prompt):
    instruction = (
        "Split the following story into short visual scenes (1 per line) suitable for illustration:\n"
        f"{prompt.strip()}\n\nScenes:\n"
    )
    output = llm_pipe(instruction)[0]["generated_text"]
    # Extract only the list of scenes
    scene_lines = output.split("Scenes:")[-1].split("\n")
    scenes = [line.strip("0123456789.:-• ") for line in scene_lines if line.strip()]
    return scenes

# === Main Image Generation Function ===
def generate_story_images(prompt, max_scenes):
    scenes = split_story(prompt)[:max_scenes]
    images = []
    for i, scene in enumerate(scenes):
        if scene.strip():
            print(f"🎨 Generating image {i+1}/{len(scenes)}: {scene}")
            image = pipe(scene).images[0]
            images.append(image)
    return images

# === Gradio UI ===
interface = gr.Interface(
    fn=generate_story_images,
    inputs=[
        gr.Textbox(label="Enter Narrative Prompt", lines=4, placeholder="Once upon a time in a forgotten forest..."),
        gr.Slider(1, 8, step=1, value=4, label="Max Scenes")
    ],
    outputs=gr.Gallery(label="Visual Story").style(grid=(2, 2)),
    title="DreamBoard: AI-Powered Visual Storytelling",
    description="Turn your story into images using Stable Diffusion and a small language model."
)

if __name__ == "__main__":
    interface.launch()
