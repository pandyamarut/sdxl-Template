import os
from huggingface_hub import snapshot_download

# Get the hugging face token
HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_WRITE_TOKEN', None)
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = os.environ.get('MODEL_REVISION', "main")
MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/workspace/')

# Download the model from hugging face
download_kwargs = {}

if HUGGING_FACE_HUB_TOKEN:
    download_kwargs["token"] = HUGGING_FACE_HUB_TOKEN

snapshot_download(
    MODEL_NAME,
    revision="main",
    local_dir=f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}",
    **download_kwargs
)