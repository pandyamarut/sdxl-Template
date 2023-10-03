import runpod
import subprocess
import os

def huggingface_login():
    try:
        # Get the value of the TOKEN environment variable
        token = os.environ.get("HUGGING_FACE_WRITE_TOKEN")

        if token:
            # Run the huggingface-cli login command with the TOKEN environment variable
            subprocess.run(["huggingface-cli", "login", "--token", token], check=True)

            # If the command was successful, you can print a success message or perform other actions
            print("Hugging Face login successful!")

        else:
            # Handle the case where the TOKEN environment variable is not set
            print("TOKEN environment variable is not set. Please set it before running the command.")

    except subprocess.CalledProcessError as e:
        # If the command failed, you can print an error message or handle the error as needed
        error_message = f"Error running huggingface-cli login: {e}"
        print(error_message)


def run_accelerate_config():
    try:
        subprocess.run(["accelerate", "config", "default"], check=True)
        print("Accelerate config successful!")
    except subprocess.CalledProcessError as e:
        error_message = f"Error running accelerate config: {e}"
        print(error_message)   

def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    print(event)

    # Define the command to execute
    command = (
        "accelerate launch src/train_dreambooth_lora_sdxl.py "
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 "
        "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix "
        "--instance_data_dir=dog "
        "--output_dir=lora_xdsl "
        "--mixed_precision=fp16 "
        "--instance_prompt='a photo of sks dog' "
        "--resolution=1024 "
        "--train_batch_size=2 "
        "--gradient_accumulation_steps=2 "
        "--gradient_checkpointing "
        "--learning_rate=1e-4 "
        "--lr_scheduler='constant' "
        "--lr_warmup_steps=0 "
        "--enable_xformers_memory_efficient_attention "
        "--mixed_precision=fp16 "
        "--use_8bit_adam "
        "--max_train_steps=1 "
        "--checkpointing_steps=717 "
        "--seed=0 "
        "--push_to_hub"
    )

    try:
        # Execute the command and capture the output
        huggingface_login()
        run_accelerate_config()
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, shell=True)
        
        # Return the output directory or a message indicating success
        return output

    except subprocess.CalledProcessError as e:
        error_message = f"Error running command: {e}\nOutput: {e.output}"
        print(error_message)
        # Return an error message or status code
        return error_message

# Call the handler function

runpod.serverless.start({"handler": handler})
