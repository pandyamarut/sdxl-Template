# import os
# from huggingface_hub import snapshot_download

# # Get the hugging face token
# HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_WRITE_TOKEN', None)
# MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
# MODEL_REVISION = os.environ.get('MODEL_REVISION', "main")
# MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/workspace/')

# # Download the model from hugging face
# download_kwargs = {}

# if HUGGING_FACE_HUB_TOKEN:
#     download_kwargs["token"] = HUGGING_FACE_HUB_TOKEN

# snapshot_download(
#     MODEL_NAME,
#     revision="main",
#     # allow_patterns="*.safetensors",
#     local_dir=f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}",
#     **download_kwargs
# )


# builder/model_fetcher.py

import torch
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, PretrainedConfig



MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"



def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
      # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer", revision=None, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer_2", revision=None, use_fast=False
    )
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(
       MODEL_NAME, subfolder="unet", revision=None
    )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        MODEL_NAME, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        MODEL_NAME, None, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", revision=None
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
         MODEL_NAME, subfolder="text_encoder_2", revision=None
    )
    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)

    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()