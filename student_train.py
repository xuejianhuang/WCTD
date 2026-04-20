import json
import os
import random

import safetensors
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torch.nn as nn
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers, convert_all_state_dict_to_peft
from torch.utils.tensorboard import SummaryWriter
from pytorch_wavelets import DWTForward, DWTInverse
from PIL import Image
import argparse
# Custom utilities
from utils import *

# Global settings
Image.MAX_IMAGE_PIXELS = None


def main():
    # ======================== Configuration (TPAMI Standard) ========================


    def parse_args():
        parser = argparse.ArgumentParser()

        # Model & Paths
        parser.add_argument("--pretrained_model_name_or_path", type=str, default="downloaded_pretrain_models")
        parser.add_argument("--dataset_dir", type=str, default='')
        parser.add_argument("--output_root", type=str, default='')
        parser.add_argument("--label_to_caption_path", type=str, default="imagenet_class_index.json")
        parser.add_argument("--teacher_lora_path", type=str, default=None)
        parser.add_argument("--student_lora_path", type=str, default=None)

        # Training Settings
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_batch_size", type=int, default=4)
        parser.add_argument("--num_train_epochs", type=int, default=10)

        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--dataloader_num_workers", type=int, default=4)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.999)
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        parser.add_argument("--max_grad_norm", type=float, default=5.0)

        # Attack & Model Settings
        parser.add_argument("--rank", type=int, default=16)
        parser.add_argument("--eps", type=int, default=16)
        parser.add_argument("--model_type", type=str, default='res152') #incv3，res152，swin_tiny，mixer_b16，deit_b，cycle_mlp

        parser.add_argument("--attack_mode", type=str, default="multi_targeted")
        parser.add_argument("--s_for", type=int, default=6)
        parser.add_argument("--s_gen", type=int, default=6)
        parser.add_argument("--num_timesteps", type=int, default=20)
        parser.add_argument("--label_flag", type=str, default='N8')

        # Distillation
        parser.add_argument("--lambda_log", type=float, default=0.1)
        parser.add_argument("--distill_temperature", type=float, default=8.0)
        parser.add_argument("--w_distill", type=float, default=1.0)
        parser.add_argument("--lambda_img", type=float, default=1.0)
        parser.add_argument("--w_adv", type=float, default=1.0)

        args = parser.parse_args()

        args.log_dir = os.path.join(args.output_root, "logs")

        return args

    args = parse_args()
    eps = args.eps / 255.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_set = get_classes(args.label_flag)

    # Random Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Directory Initialization
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_hparams({
        "lr": args.learning_rate,
        "batch_size": args.train_batch_size,
        "lora_rank": args.rank,
        "epsilon": args.eps,
        "distill_temp": args.distill_temperature,
        "attack_mode": args.attack_mode,
    }, {})

    # Precision Settings
    torch.set_float32_matmul_precision('high')
    weight_dtype = torch.float32

    # ======================== Model Initialization ========================
    # Stable Diffusion Components
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    # Teacher-Student UNet for Distillation
    teacher_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    student_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_origin = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Schedulers
    ddim_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_inv_scheduler = DDIMInverseScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_inv_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps_inverse = ddim_inv_scheduler.timesteps[:args.s_for]
    ddim_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps = ddim_scheduler.timesteps[-args.s_gen:]
    scaling_factor = vae.config.scaling_factor

    print(f"[INFO] Loading target classifier: {args.model_type}")
    classifier, classifier_input_size = load_single_classifier(args.model_type, device, weight_dtype)
    criterion = nn.CrossEntropyLoss()

    # ======================== Freeze Pre-trained Models ========================
    teacher_unet.requires_grad_(False)
    student_unet.requires_grad_(False)
    unet_origin.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    classifier.requires_grad_(False)

    # Move Models to Device
    teacher_unet.to(device, dtype=weight_dtype)
    student_unet.to(device, dtype=weight_dtype)
    unet_origin.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    classifier.to(device, dtype=weight_dtype)

    # Teacher LoRA
    teacher_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    teacher_unet.add_adapter(teacher_lora_config)
    if args.teacher_lora_path is not None and os.path.exists(args.teacher_lora_path):
        lora_state_dict = safetensors.torch.load_file(args.teacher_lora_path)
        unet_lora_state_dict = {
            key[len("unet."):]: val for key, val in lora_state_dict.items() if key.startswith("unet.")
        }
        unet_lora_state_dict = convert_all_state_dict_to_peft(unet_lora_state_dict)
        set_peft_model_state_dict(teacher_unet, unet_lora_state_dict)

    # Student LoRA
    student_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    student_unet.add_adapter(student_lora_config)
    if args.student_lora_path is not None and os.path.exists(args.student_lora_path):
        lora_state_dict = safetensors.torch.load_file(args.student_lora_path)
        unet_lora_state_dict = {
            key[len("unet."):]: val for key, val in lora_state_dict.items() if key.startswith("unet.")
        }
        unet_lora_state_dict = convert_all_state_dict_to_peft(unet_lora_state_dict)
        set_peft_model_state_dict(student_unet, unet_lora_state_dict)

    # Enable Student LoRA Gradients
    for name, param in student_unet.named_parameters():
        param.requires_grad_("lora" in name)
    student_params = [p for p in student_unet.parameters() if p.requires_grad]

    # ======================== Helper Functions (Unified) ========================
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # ======================== Dataset & DataLoader (Unified) ========================
    # Image Transformation
    if args.model_type in ['res152', 'swin_tiny', 'mixer_b16', 'cycle_mlp', 'deit_b']:
        scale_size = 224
        unet_in_transform = transforms.Resize(352)
        unet_out_transform = transforms.Resize(224)
    elif args.model_type == 'incv3':
        scale_size = 299
        unet_in_transform = transforms.Resize(352)
        unet_out_transform = transforms.Resize(299)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not supported")

    train_transforms = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(scale_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_dataset = torchvision.datasets.ImageFolder(args.dataset_dir, train_transforms)

    # Label Mapping
    with open(args.label_to_caption_path, 'r') as f:
        label_to_caption = json.load(f)

    def get_label_and_caption(attack_mode):
        if attack_mode == "multi_targeted":
            label = int(random.choice(label_set).item())
        else:
            raise NotImplementedError("Attack mode not supported")
        caption = f"a photo of {label_to_caption[str(label)][1]}"
        return label, caption

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        pixel_values = pixel_values.contiguous().float()
        labels = []
        captions = []
        for _ in examples:
            label, caption = get_label_and_caption(args.attack_mode)
            labels.append(label)
            captions.append(caption)
        labels = torch.tensor(labels, dtype=torch.long)
        captions = tokenize_captions(captions)
        return {"imgs": pixel_values, "captions": captions, "labels": labels}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
    )
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ======================== Optimizer & Scheduler ========================
    optimizer = torch.optim.AdamW(
        student_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    # ======================== Training Loop (TPAMI Standard) ========================
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        student_unet.train()
        # Epoch Metrics
        epoch_distill_loss_sum = 0.0
        epoch_adv_loss_sum = 0.0
        epoch_total_loss_sum = 0.0
        epoch_correct_sum = 0
        epoch_total_sum = 0

        epoch_pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(train_dataloader):
            # Data Preparation
            imgs = batch["imgs"].to(device=device, dtype=weight_dtype)
            captions = batch["captions"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            clean_imgs = unnormalize_ddpm(imgs)

            with torch.no_grad():

                latents = vae.encode(unet_in_transform(imgs)).latent_dist.sample() * scaling_factor
                encoder_hidden_states = text_encoder(captions, return_dict=False)[0]
                empty_captions = tokenize_captions([""] * imgs.size(0)).to(device)
                encoder_hidden_states_empty = text_encoder(empty_captions, return_dict=False)[0]

                # 2. DDIM Inversion
                for t in timesteps_inverse:
                    model_pred = unet_origin(latents, t, encoder_hidden_states_empty, return_dict=False)[0]
                    latents = ddim_inv_scheduler.step(model_pred, t, latents, return_dict=False)[0]

                teacher_latents = latents.clone()
                trajectory = [teacher_latents.clone()]
                for t in timesteps:
                    model_pred = teacher_unet(teacher_latents, t, encoder_hidden_states, return_dict=False)[0]
                    teacher_latents, _ = ddim_scheduler.step(model_pred, t, teacher_latents, return_dict=False)
                    trajectory.append(teacher_latents.clone())

                teacher_final_latent = trajectory[-1].clone()

                teacher_adv_raw = vae.decode(teacher_final_latent / scaling_factor, return_dict=False)[0]
                teacher_adv_raw = unet_out_transform(unnormalize_ddpm(teacher_adv_raw))
                teacher_adv_budget = budget(teacher_adv_raw, clean_imgs, eps, args.attack_mode)

                if args.model_type in ['swin_tiny', 'mixer_b16', 'deit_b', 'cycle_mlp']:
                    teacher_logits = classifier(normalize(teacher_adv_budget)).logits.detach()
                else:
                    teacher_logits = classifier(normalize(teacher_adv_budget)).detach()

                k = random.randint(0, len(timesteps) - 2)
                t_k = timesteps[k]
                z_k = trajectory[k].clone()

            model_pred_student = student_unet(z_k, t_k, encoder_hidden_states, return_dict=False)[0]
            _, pred_student_final = ddim_scheduler.step(model_pred_student, t_k, z_k, return_dict=False)

            adv_imgs_raw = vae.decode(pred_student_final / scaling_factor, return_dict=False)[0]
            adv_imgs_raw = unet_out_transform(unnormalize_ddpm(adv_imgs_raw))
            adv_imgs_adv_loss = budget(adv_imgs_raw, clean_imgs, eps, args.attack_mode)

            loss_mse = F.mse_loss(pred_student_final, teacher_final_latent)

            loss_kl = torch.tensor(0.0, device=device, dtype=weight_dtype)
            if args.lambda_log > 0:
                if args.model_type in ['swin_tiny', 'mixer_b16', 'deit_b', 'cycle_mlp']:
                    student_logits = classifier(normalize(adv_imgs_adv_loss)).logits
                else:
                    student_logits = classifier(normalize(adv_imgs_adv_loss))

                T = args.distill_temperature
                teacher_logits_T = teacher_logits / T
                student_logits_T = student_logits / T

                teacher_max_val = teacher_logits_T.max(dim=-1, keepdim=True)[0]
                student_max_val = student_logits_T.max(dim=-1, keepdim=True)[0]
                teacher_logits_T_stable = teacher_logits_T - teacher_max_val
                student_logits_T_stable = student_logits_T - student_max_val

                teacher_probs = F.softmax(teacher_logits_T_stable, dim=-1)
                student_log_probs = F.log_softmax(student_logits_T_stable, dim=-1)

                epsilon = 1e-10
                teacher_probs_safe = teacher_probs + epsilon
                teacher_probs_safe = teacher_probs_safe / teacher_probs_safe.sum(dim=-1, keepdim=True)

                loss_kl = F.kl_div(student_log_probs.double(), teacher_probs_safe.double(),
                                   reduction='batchmean').float()
                loss_kl = loss_kl * (T ** 2)

            loss_distill = args.lambda_img * loss_mse + args.lambda_log * loss_kl

            if args.model_type in ['swin_tiny', 'mixer_b16', 'deit_b', 'cycle_mlp']:
                adv_out = classifier(normalize(adv_imgs_adv_loss)).logits
            else:
                adv_out = classifier(normalize(adv_imgs_adv_loss))
            loss_adv = criterion(adv_out, labels)

            total_loss = args.w_distill * loss_distill + args.w_adv * loss_adv

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            batch_size = imgs.size(0)
            epoch_distill_loss_sum += loss_distill.item() * batch_size
            epoch_adv_loss_sum += loss_adv.item() * batch_size
            epoch_total_loss_sum += total_loss.item() * batch_size
            epoch_correct_sum += (adv_out.argmax(1) == labels).sum().item()
            epoch_total_sum += batch_size

            # Progress Bars

            global_step += 1
            epoch_pbar.update(1)
            logs = {"distill_loss": loss_distill.item(), "adv_loss": loss_adv.item(), "total_loss": total_loss.item()}
            epoch_pbar.set_postfix(**logs)

            if global_step >= (max_train_steps or float('inf')):
                break

        epoch_pbar.close()

        # -------------------------- Epoch Summary & Logging --------------------------
        avg_distill = epoch_distill_loss_sum / epoch_total_sum
        avg_adv = epoch_adv_loss_sum / epoch_total_sum
        epoch_asr = epoch_correct_sum / epoch_total_sum

        print(f"Epoch {epoch} | Train ASR: {epoch_asr:.2%} | Distill Loss: {avg_distill:.4f} | Adv Loss: {avg_adv:.4f}")

        # TensorBoard Logging
        writer.add_scalar("Train/Distill_Loss", avg_distill, global_step)
        writer.add_scalar("Train/Adv_Loss", avg_adv, global_step)
        writer.add_scalar("Train/Attack_Success_Rate", epoch_asr, global_step)
        writer.flush()

        # ======================== Save LoRA Checkpoint Every Epoch ========================
        epoch_save_path = os.path.join(args.output_root, f"epoch_{epoch}")
        unet_lora_state_dict_epoch = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(student_unet)
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=epoch_save_path,
            unet_lora_layers=unet_lora_state_dict_epoch,
            text_encoder_lora_layers=None,
            safe_serialization=True,
        )
        print(f"[INFO] Saved Epoch {epoch} LoRA model to: {epoch_save_path}")

        if global_step >= max_train_steps:
            break

    # -------------------------- Final Save --------------------------

    writer.close()
    final_save_path = os.path.join(args.output_root, "final_model")
    final_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(student_unet))
    StableDiffusionPipeline.save_lora_weights(final_save_path, final_lora_state, None, True)
    print(f"[INFO] Training finished! Model saved to {args.output_root}")


if __name__ == "__main__":
    main()
