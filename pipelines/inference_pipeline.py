from zenml import pipeline

from steps.model_load import load_model

@pipeline
def inference_model(model_path: str, prompt: str):
    pipe = load_model(model_path)
    image = pipe(prompt).images[0]
    image.save("image1.png")