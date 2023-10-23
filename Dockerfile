# Base image
FROM runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel

ENV DEBIAN_FRONTEND=noninteractive
ARG HUGGING_FACE_HUB_WRITE_TOKEN
ENV HUGGING_FACE_HUB_WRITE_TOKEN=$HUGGING_FACE_HUB_WRITE_TOKEN

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

# Install required packages
RUN apt-get update && apt-get install -y python3-pip

# Test with abov
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Download the training script
# RUN wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
COPY src/train_dreambooth_lora_sdxl.py /workspace/src/train_dreambooth_lora_sdxl.py

ADD src .

CMD huggingface-cli login --token $HUGGING_FACE_HUB_WRITE_TOKEN

RUN python -u /download_model.py

CMD ["bash", "-c", "accelerate config default && python -u handler.py"]

