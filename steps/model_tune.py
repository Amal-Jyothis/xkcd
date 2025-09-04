import torch
from peft import LoraConfig, get_peft_model
from diffusers import DiffusionPipeline
import torch
from zenml import step
import logging
from transformers import TrainingArguments, Trainer, CLIPTokenizer, CLIPTextModel
from torchvision import transforms

class UNetWithLoss(torch.nn.Module):
    def __init__(self, unet, noise_scheduler):
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler

    def forward(self, latents, encoder_hidden_states):
        
        bsz = latents.shape[0]
        device = latents.device

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
            device=device, dtype=torch.long
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample

        loss = torch.nn.functional.mse_loss(model_pred, noise)
        return {"loss": loss}

class DiffusionCollator:
    def __init__(self, tokenizer, vae, text_encoder, image_size=(256, 256)):
        self.tokenizer = tokenizer
        self.vae = vae
        self.text_encoder = text_encoder
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()

    @torch.no_grad()
    def __call__(self, examples):
        
        texts = [ex["text"] for ex in examples]
        tok = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tok.input_ids.to(self.text_encoder.device)
        encoder_hidden_states = self.text_encoder(input_ids)[0].cpu()

        images = torch.stack([self.image_transform(ex["image"]) for ex in examples]).to(self.vae.device)
        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        latents = latents.cpu()

        return {
            "latents": latents,
            "encoder_hidden_states": encoder_hidden_states,
        }

@step
def model_tuning(dataset, base_pipe, tokenizer, text_encoder, peft_save_dir):

    data_collator = DiffusionCollator(tokenizer, base_pipe.vae, text_encoder, base_pipe.scheduler)

    lora_config = LoraConfig(
        r=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_alpha=32,
        lora_dropout=0.05
    )

    lora_model = get_peft_model(base_pipe.unet, lora_config)
    train_model = UNetWithLoss(lora_model, base_pipe.scheduler)

    training_args = TrainingArguments(
        learning_rate=1e-3,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        report_to='none',
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset['train'],
        # eval_dataset=dataset['validation'],
        data_collator=data_collator
    )

    trainer.train()

    lora_model.save_pretrained(peft_save_dir)
    tokenizer.save_pretrained(peft_save_dir)
