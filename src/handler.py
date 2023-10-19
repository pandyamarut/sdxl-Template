import runpod
import subprocess
import os
from huggingface_hub import snapshot_download

def huggingface_login():
    try:
        # Get the value of the TOKEN environment variable
        token = os.environ.get("HUGGING_FACE_HUB_WRITE_TOKEN")

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

def download_dataset(dataset_name, local_dir="./"):
    try:
        full_local_dir = os.path.join(local_dir, dataset_name)
        snapshot_download(dataset_name, local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes")
        print(f"Downloaded '{dataset_name}' to '{local_dir}' successfully.")
        return full_local_dir
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def handler(job):
    '''
    This is the handler function that will be called by the serverless.
    '''

    job_input = job["input"]

    # Get the parameters from the job input
    dataset_directory_path = job_input["dataset_directory_path"]
    output_directory = job_input["output_directory"]
    instance_prompt = job_input["instance_prompt"]
    batch_size = job_input["batch_size"]
    training_steps = job_input["training_steps"]

    local_directory = "./dog"

    dataset_path = download_dataset(dataset_directory_path, local_dir=local_directory)
    job_output = {}

    # most of the parameteres will be path (Network storage)
    
    training_command = (
        "accelerate launch src/train_dreambooth_lora_sdxl.py "
        "--pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0' "
        "--pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' "
        f"--instance_data_dir={dataset_path} "
        f"--output_dir={output_directory} "
        "--mixed_precision=fp16 "
        f"--instance_prompt='{instance_prompt}' "
        "--resolution=1024 "
        f"--train_batch_size={batch_size} "
        "--gradient_accumulation_steps=2 "
        "--gradient_checkpointing "
        "--learning_rate=1e-4 "
        "--lr_scheduler='constant' "
        "--lr_warmup_steps=0 "
        "--enable_xformers_memory_efficient_attention "
        "--mixed_precision=fp16 "
        "--use_8bit_adam "
        f"--max_train_steps={training_steps} "
        "--checkpointing_steps=717 "
        "--seed=0 "
        "--push_to_hub"
    )
    try:
        # Execute the command and capture the output
        huggingface_login()
        run_accelerate_config()
        output = subprocess.run(training_command, stderr=subprocess.STDOUT, text=True, shell=True, check=True)
        
        # Return the output directory or a message indicating success
        job_output["output_directory"] == output_directory
        return job_output

    except subprocess.CalledProcessError as e:
        error_message = f"Error running command: {e}\nOutput: {e.output}"
        print(error_message)
        # Return an error message or status code
        return error_message

# Call the handler function

runpod.serverless.start({"handler": handler})
