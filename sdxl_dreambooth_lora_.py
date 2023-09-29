# -*- coding: utf-8 -*-
"""SDXL_DreamBooth_LoRA_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I6mnFxErQVt_2U_XDVrYC4RSXmxukZII

## Fine-tuning Stable Diffusion XL with DreamBooth and LoRA on a free-tier Colab Notebook 🧨

In this notebook, we show how to fine-tune [Stable Diffusion XL (SDXL)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl) with [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth) and [LoRA](https://huggingface.co/docs/diffusers/main/en/training/lora) on a T4 GPU.

SDXL consists of a much larger UNet and two text encoders that make the cross-attention context quite larger than the previous variants.

So, to pull this off, we will make use of several tricks such as gradient checkpointing, mixed-precision, and 8-bit Adam. So, hang tight and let's get started 🧪

## Setup 🪓
"""

# Check the GPU
!nvidia-smi

# Install dependencies.
!pip install xformers bitsandbytes transformers accelerate -q

"""Make sure to install `diffusers` from `main`."""

!pip install git+https://github.com/huggingface/diffusers.git@e731ae0ec817649bf2c15f9f719269d57062696c -q

"""Download `diffusers` SDXL DreamBooth training script."""

!wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py

"""## Dataset 🐶

Let's download some example images:
"""

from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)

"""Preview the images:"""

from PIL import Image

def image_grid(imgs, rows, cols, resize=256):
    assert len(imgs) == rows * cols

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

import glob

imgs = [Image.open(path) for path in glob.glob("./dog/*.jpeg")]
image_grid(imgs, 1, 5)

"""## Prep for training 💻

Initialize `accelerate`:
"""

!accelerate config default

"""Make sure to log into [your Hugging Face account](https://huggingface.co/) and pass [your access token](https://huggingface.co/docs/hub/security-tokens) so that we can push the trained checkpoints to the Hugging Face Hub:"""

!huggingface-cli login

"""## Train! 🔬

Alright let's launch a training. Make sure to add `push_to_hub` so that the checkpoint is automatically pushed to the Hub and doesn't get lost.

To ensure we can DreamBooth with LoRA on a heavy pipeline like Stable Diffusion XL, we're using:

* Gradient checkpointing (`--gradient_accumulation_steps`)
* Memory-efficient attention (`--enable_xformers_memory_efficient_attention`)
* 8-bit Adam (`--use_8bit_adam`)
* Mixed-precision training (`--mixed-precision="fp16"`)

The `--push_to_hub` argument ensures that the trained checkpoints are automatically pushed to the Hugging Face Hub.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile /usr/local/lib/python3.10/dist-packages/diffusers/utils/__init__.py
# # Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# 
# 
# import os
# 
# from packaging import version
# 
# from .. import __version__
# from .accelerate_utils import apply_forward_hook
# from .constants import (
#     CONFIG_NAME,
#     DEPRECATED_REVISION_ARGS,
#     DIFFUSERS_CACHE,
#     DIFFUSERS_DYNAMIC_MODULE_NAME,
#     FLAX_WEIGHTS_NAME,
#     HF_MODULES_CACHE,
#     HUGGINGFACE_CO_RESOLVE_ENDPOINT,
#     ONNX_EXTERNAL_WEIGHTS_NAME,
#     ONNX_WEIGHTS_NAME,
#     SAFETENSORS_WEIGHTS_NAME,
#     WEIGHTS_NAME,
# )
# from .deprecation_utils import deprecate
# from .doc_utils import replace_example_docstring
# from .dynamic_modules_utils import get_class_from_dynamic_module
# from .hub_utils import (
#     HF_HUB_OFFLINE,
#     _add_variant,
#     _get_model_file,
#     extract_commit_hash,
#     http_user_agent,
# )
# from .import_utils import (
#     BACKENDS_MAPPING,
#     ENV_VARS_TRUE_AND_AUTO_VALUES,
#     ENV_VARS_TRUE_VALUES,
#     USE_JAX,
#     USE_TF,
#     USE_TORCH,
#     DummyObject,
#     OptionalDependencyNotAvailable,
#     is_accelerate_available,
#     is_accelerate_version,
#     is_bs4_available,
#     is_flax_available,
#     is_ftfy_available,
#     is_inflect_available,
#     is_invisible_watermark_available,
#     is_k_diffusion_available,
#     is_k_diffusion_version,
#     is_librosa_available,
#     is_note_seq_available,
#     is_omegaconf_available,
#     is_onnx_available,
#     is_safetensors_available,
#     is_scipy_available,
#     is_tensorboard_available,
#     is_tf_available,
#     is_torch_available,
#     is_torch_version,
#     is_torchsde_available,
#     is_transformers_available,
#     is_transformers_version,
#     is_unidecode_available,
#     is_wandb_available,
#     is_xformers_available,
#     requires_backends,
# )
# from .logging import get_logger
# from .outputs import BaseOutput
# from .pil_utils import PIL_INTERPOLATION, numpy_to_pil, pt_to_pil
# from .torch_utils import is_compiled_module, randn_tensor
# 
# 
# if is_torch_available():
#     from .testing_utils import (
#         floats_tensor,
#         load_hf_numpy,
#         load_image,
#         load_numpy,
#         load_pt,
#         nightly,
#         parse_flag_from_env,
#         print_tensor_test,
#         require_torch_2,
#         require_torch_gpu,
#         skip_mps,
#         slow,
#         torch_all_close,
#         torch_device,
#     )
#     from .torch_utils import maybe_allow_in_graph
# 
# from .testing_utils import export_to_gif, export_to_obj, export_to_ply, export_to_video
# 
# 
# logger = get_logger(__name__)
# 
# 
# def check_min_version(min_version):
#     return None
#     if version.parse(__version__) < version.parse(min_version):
#         if "dev" in min_version:
#             error_message = (
#                 "This example requires a source install from HuggingFace diffusers (see "
#                 "`https://huggingface.co/docs/diffusers/installation#install-from-source`),"
#             )
#         else:
#             error_message = f"This example requires a minimum version of {min_version},"
#         error_message += f" but the version found is {__version__}.\n"
#         raise ImportError(error_message)

#!/usr/bin/env bash
!accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="dog" \
  --output_dir="lora-trained-xl-colab" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --max_train_steps=1 \
  --checkpointing_steps=717 \
  --seed="0" \
  --push_to_hub

"""Cool the model has been uploaded to https://huggingface.co/sayakpaul/lora-trained-xl-colab 🔥🚀

Let's generate some images with it!

## Inference 🐕
"""

from diffusers import DiffusionPipeline
import torch

from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights("mwiki/lora-trained-xl-colab")

_ = pipe.to("cuda")

prompt = "a photo of sks dog in a car"

image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image