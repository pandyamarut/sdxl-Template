# Base image
FROM runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel

ENV DEBIAN_FRONTEND=noninteractive

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace


# Install required packages
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y git
# Install required packages
RUN pip install git+https://github.com/huggingface/diffusers.git@e731ae0ec817649bf2c15f9f719269d57062696c -q

# Test with abov
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Download the training script
# RUN wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
COPY src/train_dreambooth_lora_sdxl.py /workspace/src/train_dreambooth_lora_sdxl.py


# Create a directory for the dataset and download the example images to it, this is test
RUN mkdir /workspace/dog

# Download the example images
#RUN pip install huggingface_hub
RUN python3 -c "from huggingface_hub import snapshot_download; local_dir = '/workspace/dog'; snapshot_download('diffusers/dog-example', local_dir=local_dir, repo_type='dataset', ignore_patterns='.gitattributes')"

# COPY check_version.py /usr/local/lib/python3.8/dist-packages/diffusers/utils/__init__.py
COPY __init__.py /usr/local/lib/python3.10/dist-packages/diffusers/utils/__init__.py

# RUN accelerate config default

ADD src .

CMD ["bash", "-c", "accelerate config default && python -u handler.py"]

