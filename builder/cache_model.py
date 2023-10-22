# cache_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def cache_pretrained_model():
    model_name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

if __name__ == "__main__":
    cache_pretrained_model()
