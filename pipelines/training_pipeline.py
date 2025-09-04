from zenml import pipeline

from steps.data_collect import collect_data
from steps.model_load import loading_base_model
from steps.model_tune import model_tuning

@pipeline
def training_pipeline(data_path: str, model_url: str, tokenizer_url: str, peft_save_dir: str):
    
    data = collect_data(data_path)
    base_pipe, tokenizer, text_encoder = loading_base_model(model_url, tokenizer_url)
    model_tuning(data, base_pipe, tokenizer, text_encoder, peft_save_dir)