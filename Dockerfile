# Base image
FROM runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel

ENV DEBIAN_FRONTEND=noninteractive

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y git
# Install required packages
RUN pip install git+https://github.com/huggingface/diffusers.git@e731ae0ec817649bf2c15f9f719269d57062696c -q && \
    pip install xformers bitsandbytes transformers accelerate -q

# Download the training script
# RUN wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py

# Create a directory for the dataset
RUN mkdir /app/dog

# Download the example images
RUN pip install huggingface_hub
RUN python3.8 -c "from huggingface_hub import snapshot_download; local_dir = '/app/dog'; snapshot_download('diffusers/dog-example', local_dir=local_dir, repo_type='dataset', ignore_patterns='.gitattributes')"

COPY check_version.py /usr/local/lib/python3.8/dist-packages/diffusers/utils/__init__.py
COPY check_version.py /usr/local/lib/python3/dist-packages/diffusers/utils/__init__.py

RUN accelerate config default

# Log in to Hugging Face (if needed)
# RUN huggingface-cli login

# Run your specific command
CMD [ "bash", "-c", "accelerate launch src/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0' \
  --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' \
  --instance_data_dir='dog' \
  --output_dir='lora-trained-xl-colab' \
  --mixed_precision='fp16' \
  --instance_prompt='a photo of sks dog' \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler='constant' \
  --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision='fp16' \
  --use_8bit_adam \
  --max_train_steps=1 \
  --checkpointing_steps=717 \
  --seed='0' \
  --push_to_hub" ]
