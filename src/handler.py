#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

import runpod

# Load models into VRAM here so they can be warm between requests
import subprocess

def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    print(event)

    # Define the commands to execute
    commands = [
        "accelerate config default",  # Set the default configuration for accelerate
        "accelerate launch src/train_dreambooth_lora_sdxl.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        "--instance_data_dir=dog",
        "--output_dir=lora-trained-xl-colab",
        "--mixed_precision=fp16",
        "--instance_prompt='a photo of sks dog'",
        "--resolution=1024",
        "--train_batch_size=2",
        "--gradient_accumulation_steps=2",
        "--gradient_checkpointing",
        "--learning_rate=1e-4",
        "--lr_scheduler='constant'",
        "--lr_warmup_steps=0",
        "--enable_xformers_memory_efficient_attention",
        "--mixed_precision=fp16",
        "--use_8bit_adam",
        "--max_train_steps=1",
        "--checkpointing_steps=717",
        "--seed=0",
        "--push_to_hub"
    ]

    try:
        # Execute the commands and capture the output
        output = subprocess.check_output(" && ".join(commands), stderr=subprocess.STDOUT, text=True, shell=True)
        
        # Return the output directory or a message indicating success
        return output

    except subprocess.CalledProcessError as e:
        error_message = f"Error running commands: {e}\nOutput: {e.output}"
        print(error_message)
        # Return an error message or status code
        return error_message

# Call the handler function

runpod.serverless.start({"handler": handler})
