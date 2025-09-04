from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from zenml import step
from peft import PeftModel
import logging

class LoadBaseModel:
    def load_model(self, model_url: str) -> DiffusionPipeline:
        pipe = DiffusionPipeline.from_pretrained(model_url)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = pipe.to(device)

        return pipe
    
@step
def loading_base_model(model_url:str, tokenizer_url:str) -> DiffusionPipeline:
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        base_pipe = DiffusionPipeline.load_model(model_url).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_url)
        text_encoder = CLIPTextModel.from_pretrained(tokenizer_url).to(device)

        return base_pipe, tokenizer, text_encoder
    
    except Exception as e:
        logging.error(f'Error while loading base model: {e}')

@step
def load_peft_model(model_path: str) -> DiffusionPipeline:
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pipe = DiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5").to(device)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, model_path)

        return pipe
    
    except Exception as e:
        logging.error(f'Error while loading pretrained model: {e}')