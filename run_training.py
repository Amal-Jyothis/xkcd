from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    data_path = "olivierdehaene/xkcd"
    model_url = "sd-legacy/stable-diffusion-v1-5"
    tokenizer_url = "openai/clip-vit-large-patch14"
    peft_save_dir = "trainer_output/peft_model"
    training_pipeline(data_path, model_url, tokenizer_url, peft_save_dir)