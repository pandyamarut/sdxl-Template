<div align="center">

<h1>SDXL Fine Tuning | Worker</h1>

[![CI | Test Worker](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_worker.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_worker.yml)
&nbsp;
[![Docker Image](https://github.com/runpod-workers/worker-template/actions/workflows/CD-docker_dev.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CD-docker_dev.yml)

🚀 | This serverless worker is used for fine tuning SDXL.  It supports dynamic auto-scaling using the built-in RunPod autoscaling feature.
</div>

#### Docker Arguments:
1. `HUGGING_FACE_HUB_WRITE_TOKEN`: Your private Hugging Face token. This token is required for pushing models to the hub.


#### Build an Image:
`docker build -t <your_dockerhub_directory>/image_name:tag --build-arg HUGGING_FACE_HUB_WRITE_TOKEN=<your_token_value> .` 

Please make sure to replace your_hugging_face_token_here with your actual Hugging Face token to enable model downloads that require it.


Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand.
## Test Inputs
The following inputs can be used for testing the model:
```json
{
    "input": {
        "model_name": "/workspace/pretrained_model",
        "dataset_directory_path": "/workspace/data/dataset",
        "output_directory": "/workspace/output",
        "instance_prompt": "a photo of sks dog",
        "batch_size": 32,
        "training_steps": 1000,
    }
}

```
