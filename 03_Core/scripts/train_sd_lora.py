"""
Train LoRA for Stable Diffusion 1.5 Inpainting on PCB Patches

This script fine-tunes SD1.5 Inpainting with LoRA to learn PCB-specific textures.
Optimized for 8GB GPU using:
- FP16 mixed precision
- Gradient checkpointing
- xFormers memory-efficient attention
- Small batch size with gradient accumulation

Expected training time: 2-4 hours for 5000 steps

Author: PCB Vision - SD LoRA Enhancement
"""

import os
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# Diffusers and PEFT
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)


class PCBPatchDataset(Dataset):
    """
    Dataset for PCB patches with inpainting masks
    
    Structure:
        data_dir/
            Capacitor/
                00000_image.png
                00000_inpaint_mask.png
                00000_prompt.txt
    """
    
    def __init__(self, data_dir, classes=['Capacitor', 'IC', 'Connector'], resolution=256):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.samples = []
        
        # Load all samples
        for class_name in classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found, skipping")
                continue
            
            # Find all image files
            image_files = sorted(class_dir.glob("*_image.png"))
            
            for img_path in image_files:
                base_name = img_path.stem.replace('_image', '')
                mask_path = class_dir / f"{base_name}_inpaint_mask.png"
                prompt_path = class_dir / f"{base_name}_prompt.txt"
                
                if mask_path.exists() and prompt_path.exists():
                    with open(prompt_path, 'r') as f:
                        prompt = f.read().strip()
                    
                    self.samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path),
                        'prompt': prompt,
                        'class': class_name
                    })
        
        print(f"Loaded {len(self.samples)} training samples")
        
        # Print class distribution
        class_counts = {}
        for sample in self.samples:
            cls = sample['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (original with CutMix artifacts)
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        
        # Load inpainting mask
        mask = Image.open(sample['mask_path']).convert('L')
        mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        # Create masked image (set inpaint regions to gray)
        masked_image = image.clone()
        masked_image = masked_image * (1 - mask) + 0.5 * mask  # Gray out masked regions
        
        return {
            'pixel_values': image,  # Target (clean)
            'masked_image': masked_image,
            'mask': mask,
            'prompt': sample['prompt']
        }


def collate_fn(examples):
    """Collate batch"""
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    masked_images = torch.stack([example['masked_image'] for example in examples])
    masks = torch.stack([example['mask'] for example in examples])
    prompts = [example['prompt'] for example in examples]
    
    # Normalize to [-1, 1] for Stable Diffusion
    pixel_values = pixel_values * 2.0 - 1.0
    masked_images = masked_images * 2.0 - 1.0
    
    return {
        'pixel_values': pixel_values,
        'masked_images': masked_images,
        'masks': masks,
        'prompts': prompts
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', default='data/cutmix_patches')
    parser.add_argument('--output_dir', default='models/sd_lora_pcb')
    
    # Model
    parser.add_argument('--pretrained_model', default='runwayml/stable-diffusion-inpainting')
    parser.add_argument('--resolution', type=int, default=256)
    
    # LoRA
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    
    # Training
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_train_steps', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=500)
    
    # Optimization
    parser.add_argument('--mixed_precision', default='fp16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--enable_xformers', action='store_true', default=True)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    
    # Logging
    parser.add_argument('--checkpointing_steps', type=int, default=500)
    parser.add_argument('--validation_steps', type=int, default=250)
    parser.add_argument('--num_validation_images', type=int, default=4)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=output_dir / "logs"
    )
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Load models
    logger.info(f"Loading pretrained model: {args.pretrained_model}")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder"
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet"
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae"
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler"
    )
    
    # Freeze text encoder and VAE (we only train UNet)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Enable xFormers
    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers enabled")
        except Exception as e:
            logger.warning(f"Could not enable xFormers: {e}")
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Attention layers
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    
    # Add LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Create dataset
    train_dataset = PCBPatchDataset(
        data_dir=args.data_dir,
        resolution=args.resolution
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # LR Scheduler
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move to device
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num steps = {args.max_train_steps}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
    
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(1000):  # Large number, will stop by max_steps
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Encode text
                input_ids = tokenizer(
                    batch['prompts'],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Encode images to latent space using VAE
                pixel_values = batch['pixel_values'].to(accelerator.device)
                masked_images = batch['masked_images'].to(accelerator.device)
                masks = batch['masks'].to(accelerator.device)
                
                with torch.no_grad():
                    # Encode to latent space (reduces 256x256 to 32x32 with 4 channels)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    # Encode masked image to latent space
                    masked_image_latents = vae.encode(masked_images).latent_dist.sample()
                    masked_image_latents = masked_image_latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample timestep
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Resize mask to latent dimensions (256x256 -> 32x32)
                mask_latent = F.interpolate(masks, size=latents.shape[-2:], mode='nearest')
                
                # Concatenate for inpainting: [noisy_latents(4) + mask(1) + masked_image_latents(4)] = 9 channels
                latent_model_input = torch.cat([noisy_latents, mask_latent, masked_image_latents], dim=1)
                
                # Predict noise
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states
                ).sample
                
                # Compute loss (MSE between predicted and actual noise)
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = output_dir / f"checkpoint-{global_step}"
                        accelerator.unwrap_model(unet).save_pretrained(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation (generate sample images)
                if global_step % args.validation_steps == 0:
                    logger.info(f"Step {global_step}: Validation")
                    # TODO: Add validation image generation here
                
                # Stop if max steps reached
                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    if accelerator.is_main_process:
        final_save_path = output_dir / "final"
        accelerator.unwrap_model(unet).save_pretrained(final_save_path)
        logger.info(f"Saved final model to {final_save_path}")
        print(f"\nTraining complete! LoRA weights saved to: {final_save_path}")
        print(f"Total steps: {global_step}")


if __name__ == "__main__":
    main()
