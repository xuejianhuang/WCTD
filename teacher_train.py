import json
import math
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
from utils import *


class ImageFolderWithPath(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


def main():
    # ======================== Configuration ========================

    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--pretrained_model_name_or_path", type=str, default="downloaded_pretrain_models")
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

        parser.add_argument("--rank", type=int, default=16)
        parser.add_argument("--eps", type=int, default=16)
        parser.add_argument("--label_flag", type=str, default="N8")

        parser.add_argument("--dataset_dir", type=str, required=True)
        parser.add_argument("--output_root", type=str, required=True)

        parser.add_argument("--model_type", type=str, default="res152")  # incv3，res152，swin_tiny，mixer_b16，deit_b，cycle_mlp
        parser.add_argument("--attack_mode", type=str, default="multi_targeted")
        parser.add_argument("--lora_path", type=str, default=None)
        parser.add_argument("--label_to_caption_path", type=str, default="imagenet_class_index.json")

        parser.add_argument("--s_for", type=int, default=6)
        parser.add_argument("--s_gen", type=int, default=6)
        parser.add_argument("--num_timesteps", type=int, default=20)
        parser.add_argument("--val_split_ratio", type=float, default=0.1)
        parser.add_argument("--early_stopping_patience", type=int, default=3)

        args = parser.parse_args()
        if not 0.0 < args.val_split_ratio < 1.0:
            parser.error("--val_split_ratio must be between 0 and 1")
        if args.early_stopping_patience < 1:
            parser.error("--early_stopping_patience must be at least 1")

        args.log_dir = os.path.join(args.output_root, "logs")

        return args

    args = parse_args()
    eps = args.eps / 255.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_set = get_classes(args.label_flag)

    # Random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create output directories
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_hparams({
        "lr": args.learning_rate,
        "batch_size": args.train_batch_size,
        "epochs": args.num_train_epochs,
        "lora_rank": args.rank,
        "epsilon": args.eps,
        "attack_mode": args.attack_mode,
        "wavelet": "haar"
    }, {})

    torch.set_float32_matmul_precision('high')
    weight_dtype = torch.float32

    # ======================== Wavelet Transforms ========================
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device, dtype=weight_dtype)
    idwt = DWTInverse(wave='haar', mode='zero').to(device, dtype=weight_dtype)
    dwt.requires_grad_(False)
    idwt.requires_grad_(False)

    # ======================== Model Initialization ========================
    # Stable Diffusion components
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_origin = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Schedulers
    ddim_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_inv_scheduler = DDIMInverseScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_inv_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps_inverse = ddim_inv_scheduler.timesteps[:args.s_for]
    ddim_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps = ddim_scheduler.timesteps[-args.s_gen:]

    print(f"Loading target classifier: {args.model_type}")
    classifier, classifier_input_size = load_single_classifier(args.model_type, device, weight_dtype)
    criterion = nn.CrossEntropyLoss()

    # Freeze pre-trained models
    unet.requires_grad_(False)
    unet_origin.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    classifier.requires_grad_(False)

    # Move models to device
    unet.to(device, dtype=weight_dtype)
    unet_origin.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    classifier.to(device, dtype=weight_dtype)

    # ======================== LoRA Configuration ========================
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet.add_adapter(unet_lora_config)

    # Load pre-trained LoRA weights if available
    if args.lora_path is not None and os.path.exists(args.lora_path):
        lora_state_dict = safetensors.torch.load_file(args.lora_path)
        unet_lora_state_dict = {
            key[len("unet."):]: val for key, val in lora_state_dict.items() if key.startswith("unet.")
        }
        unet_lora_state_dict = convert_all_state_dict_to_peft(unet_lora_state_dict)
        set_peft_model_state_dict(unet, unet_lora_state_dict)

    lora_layers = [p for p in unet.parameters() if p.requires_grad]

    # ======================== Helper Functions ========================
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids

    # ======================== Dataset & DataLoader ========================
    # Image transformation setup
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

    train_dataset = ImageFolderWithPath(args.dataset_dir, train_transforms)

    # Reproducible train/validation split.
    total_size = len(train_dataset)
    val_size = int(total_size * args.val_split_ratio)
    train_size = total_size - val_size
    if val_size == 0 or train_size == 0:
        raise ValueError("The dataset and --val_split_ratio must produce non-empty train and validation sets")
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    # Load label mapping
    with open(args.label_to_caption_path, 'r') as f:
        label_to_caption = json.load(f)

    wnid_to_label = {v[0]: int(k) for k, v in label_to_caption.items()}
    label_candidates = [int(x.item()) if torch.is_tensor(x) else int(x) for x in label_set]

    def get_true_label_from_path(img_path):
        file_name = os.path.basename(img_path)
        wnid = file_name.split("_")[0]   # n01440764_105 -> n01440764

        if wnid not in wnid_to_label:
           raise ValueError(f"Image prefix {wnid} not found in label_to_caption mapping, file path: {img_path}")

        return wnid_to_label[wnid]

    def get_label_and_caption(img_path, attack_mode):
        true_label = get_true_label_from_path(img_path)

        if attack_mode == "multi_targeted":
            valid_labels = [x for x in label_candidates if x != true_label]
            if len(valid_labels) == 0:
                raise ValueError(f"No valid target labels available, true label is: {true_label}")
            label = random.choice(valid_labels)
        else:
            raise NotImplementedError("Attack mode not supported")

        caption = f"a photo of {label_to_caption[str(label)][1]}"
        return label, caption

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        pixel_values = pixel_values.contiguous().float()

        img_paths = [example[2] for example in examples]

        labels = []
        captions = []
        for img_path in img_paths:
            label, caption = get_label_and_caption(img_path, args.attack_mode)
            labels.append(label)
            captions.append(caption)

        labels = torch.tensor(labels, dtype=torch.long)
        captions = tokenize_captions(captions)
        return {"imgs": pixel_values, "captions": captions, "labels": labels}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
    )
    print(f"[INFO] Dataset split: {train_size} training images, {val_size} validation images")
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ======================== Optimizer & Scheduler ========================
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    @torch.no_grad()
    def validate():
        # Keep validation labels and VAE samples repeatable without changing training RNGs.
        python_rng_state = random.getstate()
        was_training = unet.training
        unet.eval()
        adv_loss_sum = 0.0
        wavelet_loss_sum = 0.0
        total_loss_sum = 0.0
        correct_sum = 0
        sample_count = 0
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(args.seed)
                torch.manual_seed(args.seed)
                for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                    imgs = batch["imgs"].to(device=device, dtype=weight_dtype)
                    captions = batch["captions"].to(device)
                    labels = batch["labels"].to(device=device, dtype=torch.long)
                    clean_imgs = unnormalize_ddpm(imgs)

                    latents = vae.encode(unet_in_transform(imgs)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(captions, return_dict=False)[0]
                    empty_captions = tokenize_captions([""] * imgs.size(0)).to(device)
                    encoder_hidden_states_empty = text_encoder(empty_captions, return_dict=False)[0]
                    for t in timesteps_inverse:
                        model_pred = unet_origin(latents, t, encoder_hidden_states_empty, return_dict=False)[0]
                        latents = ddim_inv_scheduler.step(model_pred, t, latents, return_dict=False)[0]

                    avg_adv_loss = 0.0
                    avg_wavelet_loss = 0.0
                    avg_total_loss = 0.0
                    last_adv_out = None
                    for t in timesteps:
                        model_pred = unet(latents, t, encoder_hidden_states, return_dict=False)[0]
                        latents, pred_original_sample = ddim_scheduler.step(
                            model_pred, t, latents, return_dict=False
                        )
                        adv_imgs = vae.decode(
                            pred_original_sample / vae.config.scaling_factor, return_dict=False
                        )[0]
                        adv_imgs = unet_out_transform(unnormalize_ddpm(adv_imgs))
                        LL_clean, HF_clean = dwt(clean_imgs)
                        LL_adv, HF_adv = dwt(adv_imgs)
                        wavelet_loss = F.mse_loss(HF_adv[0], HF_clean[0])
                        adv_imgs = budget(adv_imgs, clean_imgs, eps, args.attack_mode)
                        if args.model_type in ['swin_tiny', 'mixer_b16', 'deit_b', 'cycle_mlp']:
                            adv_out = classifier(normalize(adv_imgs)).logits
                        else:
                            adv_out = classifier(normalize(adv_imgs))
                        adv_loss = criterion(adv_out, labels)
                        total_loss = lamda_adv * adv_loss + lamda_wavelet * wavelet_loss
                        avg_adv_loss += adv_loss.item() / len(timesteps)
                        avg_wavelet_loss += wavelet_loss.item() / len(timesteps)
                        avg_total_loss += total_loss.item() / len(timesteps)
                        last_adv_out = adv_out

                    batch_size = imgs.size(0)
                    adv_loss_sum += avg_adv_loss * batch_size
                    wavelet_loss_sum += avg_wavelet_loss * batch_size
                    total_loss_sum += avg_total_loss * batch_size
                    correct_sum += (last_adv_out.argmax(1) == labels).sum().item()
                    sample_count += batch_size
        finally:
            random.setstate(python_rng_state)
            unet.train(was_training)
        return {
            "adv_loss": adv_loss_sum / sample_count,
            "wavelet_loss": wavelet_loss_sum / sample_count,
            "total_loss": total_loss_sum / sample_count,
            "asr": correct_sum / sample_count,
        }

    # ======================== Training Loop ========================
    global_step = 0
    lamda_adv = 1.0
    lamda_wavelet = 10.0
    best_loss = float('inf')
    best_epoch = -1
    best_lora_state = None
    epochs_without_improvement = 0

    for epoch in range(args.num_train_epochs):
        unet.train()

        # Epoch metrics
        epoch_adv_loss_sum = 0.0
        epoch_wavelet_loss_sum = 0.0
        epoch_total_loss_sum = 0.0
        epoch_correct_sum = 0
        epoch_total_sum = 0

        epoch_pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(train_dataloader):
            imgs = batch["imgs"].to(device=device, dtype=weight_dtype)
            captions = batch["captions"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            clean_imgs = unnormalize_ddpm(imgs)

            # ---------------- DDIM Inversion ----------------
            with torch.no_grad():
                latents = vae.encode(unet_in_transform(imgs)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(captions, return_dict=False)[0]
                empty_captions = tokenize_captions([""] * imgs.size(0)).to(device)
                encoder_hidden_states_empty = text_encoder(empty_captions, return_dict=False)[0]

                for t in timesteps_inverse:
                    model_pred = unet_origin(latents, t, encoder_hidden_states_empty, return_dict=False)[0]
                    latents = ddim_inv_scheduler.step(model_pred, t, latents, return_dict=False)[0]

            # ---------------- Diffusion Generation ----------------
            avg_adv_loss = 0.0
            avg_wavelet_loss = 0.0
            avg_total_loss = 0.0
            last_adv_out = None

            for i, t in enumerate(timesteps):
                # Forward pass
                model_pred = unet(latents, t, encoder_hidden_states.detach(), return_dict=False)[0]
                latents, pred_original_sample = ddim_scheduler.step(model_pred, t, latents, return_dict=False)
                latents = latents.detach()

                # Image decoding
                adv_imgs = vae.decode(pred_original_sample / vae.config.scaling_factor, return_dict=False)[0]
                adv_imgs = unet_out_transform(unnormalize_ddpm(adv_imgs))

                # Wavelet constraint
                LL_clean, HF_clean = dwt(clean_imgs)
                LL_adv, HF_adv = dwt(adv_imgs)
                wavelet_loss = F.mse_loss(HF_adv[0], HF_clean[0])

                adv_imgs = budget(adv_imgs, clean_imgs, eps, args.attack_mode)

                # Adversarial loss
                if args.model_type in ['swin_tiny', 'mixer_b16', 'deit_b', 'cycle_mlp']:
                    adv_out = classifier(normalize(adv_imgs)).logits
                else:
                    adv_out = classifier(normalize(adv_imgs))

                adv_loss = criterion(adv_out, labels)
                total_loss = lamda_adv * adv_loss + lamda_wavelet * wavelet_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_layers, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

                # Accumulate losses
                avg_adv_loss += adv_loss.detach() / len(timesteps)
                avg_wavelet_loss += wavelet_loss.detach() / len(timesteps)
                avg_total_loss += total_loss.detach() / len(timesteps)

                if i == len(timesteps) - 1:
                    last_adv_out = adv_out.detach()

            # Update metrics
            batch_size = imgs.size(0)
            epoch_adv_loss_sum += avg_adv_loss.item() * batch_size
            epoch_wavelet_loss_sum += avg_wavelet_loss.item() * batch_size
            epoch_total_loss_sum += avg_total_loss.item() * batch_size

            if last_adv_out is not None:
                pred_labels = last_adv_out.argmax(dim=1)
                epoch_correct_sum += (pred_labels == labels).sum().item()
            epoch_total_sum += batch_size

            # Update progress bars
            global_step += 1
            epoch_pbar.update(1)

            logs = {
                "adv_loss": avg_adv_loss.item(),
                "wavelet_loss": avg_wavelet_loss.item(),
                "total_loss": avg_total_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            epoch_pbar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        # Epoch summary
        avg_epoch_adv_loss = epoch_adv_loss_sum / epoch_total_sum
        avg_epoch_wavelet_loss = epoch_wavelet_loss_sum / epoch_total_sum
        avg_epoch_total_loss = epoch_total_loss_sum / epoch_total_sum
        epoch_asr = epoch_correct_sum / epoch_total_sum

        print(
            f"Epoch {epoch} | Train ASR: {epoch_asr:.2%} | Adv Loss: {avg_epoch_adv_loss:.4f} | Wavelet Loss: {avg_epoch_wavelet_loss:.4f}"
        )

        # TensorBoard logging
        writer.add_scalar("Train/Adv_Loss", avg_epoch_adv_loss, global_step)
        writer.add_scalar("Train/Wavelet_Loss", avg_epoch_wavelet_loss, global_step)
        writer.add_scalar("Train/Attack_Success_Rate", epoch_asr, global_step)
        writer.add_scalar("Train/Epoch_loss", avg_epoch_total_loss, global_step)
        writer.flush()

        # Validate and select the checkpoint with the lowest validation total loss.
        val_metrics = validate()
        val_loss = val_metrics["total_loss"]
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}: {val_loss}")
        print(
            f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val ASR: {val_metrics['asr']:.2%} | "
            f"Adv Loss: {val_metrics['adv_loss']:.4f} | Wavelet Loss: {val_metrics['wavelet_loss']:.4f}"
        )
        writer.add_scalar("Val/Total_Loss", val_loss, global_step)
        writer.add_scalar("Val/Adv_Loss", val_metrics["adv_loss"], global_step)
        writer.add_scalar("Val/Wavelet_Loss", val_metrics["wavelet_loss"], global_step)
        writer.add_scalar("Val/Attack_Success_Rate", val_metrics["asr"], global_step)
        writer.flush()

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_lora_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in get_peft_model_state_dict(unet).items()
            }
            best_save_path = os.path.join(args.output_root, "best_model")
            StableDiffusionPipeline.save_lora_weights(
                save_directory=best_save_path,
                unet_lora_layers=convert_state_dict_to_diffusers(best_lora_state),
                text_encoder_lora_layers=None,
                safe_serialization=True,
            )
            print(f"[INFO] Saved best teacher checkpoint: epoch {best_epoch}, val_loss={best_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(f"[INFO] Validation loss did not improve: {epochs_without_improvement}/{args.early_stopping_patience}")

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"[INFO] Early stopping at epoch {epoch}; best epoch {best_epoch}, val_loss={best_loss:.6f}")
            break

        if global_step >= max_train_steps:
            break

    # Save final LoRA weights
    # Preserve the original output path, exporting the best validation checkpoint.
    if best_lora_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    set_peft_model_state_dict(unet, best_lora_state)
    writer.close()
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )
    StableDiffusionPipeline.save_lora_weights(
        save_directory=args.output_root,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=None,
        safe_serialization=True,
    )
    print(f" Final model saved to: {args.output_root}")


if __name__ == "__main__":
    main()
