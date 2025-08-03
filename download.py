from huggingface_hub import snapshot_download

print("📥 Downloading scene splitting model...")
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="./models/scene-splitter",
    resume_download=True
)

print("📥 Downloading Stable Diffusion model...")
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="./models/stable-diffusion",
    resume_download=True
)
