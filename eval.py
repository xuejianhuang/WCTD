import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import safetensors
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from diffusers.utils import convert_all_state_dict_to_peft

from utils import *

from PIL import Image, ImageFile
import pywt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths"""
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        path = self.samples[idx][0]
        return img, path, label


def main():
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--pretrained_model_name_or_path", type=str, default="downloaded_pretrain_models")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--test_batch_size", type=int, default=8)
        parser.add_argument("--dataloader_num_workers", type=int, default=4)

        parser.add_argument("--rank", type=int, default=16)
        parser.add_argument("--eps", type=int, default=16)
        parser.add_argument("--label_flag", type=str, default='N8')

        parser.add_argument("--dataset_dir", type=str, default='')
        parser.add_argument("--model_type", type=str, default='res152')
        parser.add_argument("--n_class", type=int, default=1000)
        parser.add_argument("--attack_mode", type=str, default='multi_targeted')

        parser.add_argument("--student_lora_path", type=str, default='')
        parser.add_argument("--label_to_caption_path", type=str, default='imagenet_class_index.json')

        parser.add_argument("--s_for", type=int, default=6)
        parser.add_argument("--s_gen", type=int, default=6)
        parser.add_argument("--num_timesteps", type=int, default=20)

        parser.add_argument("--save_dir", type=str, default='')

        parser.add_argument("--save_clean", type=bool, default=True)
        parser.add_argument("--save_adv_noclip", type=bool, default=True)
        parser.add_argument("--save_adv_clip", type=bool, default=True)
        parser.add_argument("--save_pert", type=bool, default=True)
        parser.add_argument("--save_wavelet_diff", type=bool, default=True)
        parser.add_argument("--compute_avg_energy", type=bool, default=True)

        parser.add_argument("--WAVELET_BASE", type=str, default='haar')
        parser.add_argument("--UNIFIED_SCALE", type=bool, default=True)

        args = parser.parse_args()
        return args
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)

    eps = args.eps / 255.0
    label_set = get_classes(args.label_flag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    weight_dtype = torch.float32

    print(" Loading diffusion models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    unet_origin = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_origin.requires_grad_(False)

    unet_test = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet_test.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet_test.add_adapter(unet_lora_config)

    if args.student_lora_path and os.path.exists(args.student_lora_path):
        print(f"Loading LoRA from {args.student_lora_path}")
        lora_state_dict = safetensors.torch.load_file(args.student_lora_path)
        unet_lora_state_dict = {k[len("unet."):]: v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_lora_state_dict = convert_all_state_dict_to_peft(unet_lora_state_dict)
        set_peft_model_state_dict(unet_test, unet_lora_state_dict)
        print("LoRA loaded")
    else:
        print("LoRA weights not found")
        return

    ddim_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_inv_scheduler = DDIMInverseScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ddim_inv_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps_inverse = ddim_inv_scheduler.timesteps[:args.s_for]
    ddim_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps = ddim_scheduler.timesteps[-args.s_gen:]
    t_val = timesteps[0]


    if args.model_type in ['res152', 'swin_tiny', 'mixer_b16', 'cycle_mlp', 'deit_b']:
        scale_size = 224
        unet_in_transform = transforms.Resize(352)
        unet_out_transform = transforms.Resize(224)
    elif args.model_type == 'incv3':
        scale_size = 299
        unet_in_transform = transforms.Resize(352)
        unet_out_transform = transforms.Resize(299)
    else:
        raise NotImplementedError

    test_transforms = transforms.Compose([
        transforms.Resize(scale_size), transforms.CenterCrop(scale_size),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    test_dataset = ImageFolderWithPaths(args.dataset_dir, test_transforms)

    with open(args.label_to_caption_path, 'r') as f:
        label_to_caption = json.load(f)

    def tokenize_captions(captions):
        return tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncate=True, return_tensors="pt"
        ).input_ids

    def collate_fn(examples):
        pixel_values = torch.stack([ex[0] for ex in examples]).to(memory_format=torch.contiguous_format).float()
        img_paths = [ex[1] for ex in examples]
        labels = torch.tensor([ex[2] for ex in examples], dtype=torch.long)
        captions = [f"a photo of {label_to_caption[str(lbl.item())][1]}" for lbl in labels]
        captions = tokenize_captions(captions)
        return {"imgs": pixel_values, "captions": captions, "labels": labels, "img_paths": img_paths}

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn,
        batch_size=args.test_batch_size, num_workers=args.dataloader_num_workers,
    )

    unet_origin.to(device, dtype=weight_dtype)
    unet_test.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    energy_accumulator = None
    energy_count = 0

    unet_origin.eval()
    unet_test.eval()
    test_pbar = tqdm(total=len(test_dataloader), desc="Generating", leave=False)

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            imgs = batch["imgs"].to(device, dtype=weight_dtype)
            img_paths = batch["img_paths"]
            captions = batch["captions"].to(device)
            batch_size_current = imgs.size(0)

            latents = vae.encode(unet_in_transform(imgs)).latent_dist.sample() * vae.config.scaling_factor
            encoder_hidden_states = text_encoder(captions)[0]
            encoder_hidden_states_empty = text_encoder(tokenize_captions([""] * batch_size_current).to(device))[0]

            for t in timesteps_inverse:
                model_pred = unet_origin(latents, t, encoder_hidden_states=encoder_hidden_states_empty)[0]
                latents = ddim_inv_scheduler.step(model_pred, t, latents)[0]

            model_pred_student = unet_test(latents, t_val, encoder_hidden_states=encoder_hidden_states)[0]
            pred_student_final = ddim_scheduler.step(model_pred_student, t_val, latents)[1]

            adv_imgs_noclip = vae.decode(pred_student_final / vae.config.scaling_factor)[0]
            adv_imgs_noclip = unet_out_transform(unnormalize_ddpm(adv_imgs_noclip))
            clean_imgs = unnormalize_ddpm(imgs)
            adv_imgs_clip = budget(adv_imgs_noclip, clean_imgs, eps, args.attack_mode)

            diff = adv_imgs_clip - clean_imgs
            diff_clamped = torch.clamp(diff, -eps, eps)
            pert_tensor = torch.clamp((diff_clamped + eps) / (2 * eps), 0.0, 1.0)

            clean_imgs_cpu = clean_imgs.detach().cpu()
            adv_noclip_cpu = torch.clamp(adv_imgs_noclip, 0.0, 1.0).detach().cpu()
            adv_clip_cpu = torch.clamp(adv_imgs_clip, 0.0, 1.0).detach().cpu()
            pert_cpu = pert_tensor.detach().cpu()
            diff_for_energy = (adv_clip_cpu - clean_imgs_cpu).numpy()

            for j in range(batch_size_current):
                orig_path = img_paths[j]
                base_name = os.path.splitext(os.path.basename(orig_path))[0]
                sample_save_dir = os.path.join(args.save_dir, "images", base_name)
                os.makedirs(sample_save_dir, exist_ok=True)

                clean_pil = transforms.ToPILImage('RGB')(clean_imgs_cpu[j])
                adv_noclip_pil = transforms.ToPILImage('RGB')(adv_noclip_cpu[j])
                adv_clip_pil = transforms.ToPILImage('RGB')(adv_clip_cpu[j])
                pert_pil = transforms.ToPILImage('RGB')(pert_cpu[j])

                if args.save_clean:
                    clean_pil.save(os.path.join(sample_save_dir, "clean.png"))
                if args.save_adv_noclip:
                    adv_noclip_pil.save(os.path.join(sample_save_dir, "adv_noclip.png"))
                if args.save_adv_clip:
                    adv_clip_pil.save(os.path.join(sample_save_dir, "adv_clip.png"))
                if args.save_pert:
                    pert_pil.save(os.path.join(sample_save_dir, "pert.png"))

                if args.compute_avg_energy:
                    try:
                        fft_shift = np.fft.fftshift(np.fft.fft2(diff_for_energy[j], axes=(-2, -1)), axes=(-2, -1))
                        energy = np.log(np.abs(fft_shift) ** 2 + 1e-8)
                        energy_mean = energy.mean(axis=0)
                        if energy_accumulator is None:
                            energy_accumulator = energy_mean
                        else:
                            energy_accumulator += energy_mean
                        energy_count += 1
                    except Exception as e:
                        print(f" Energy failed {base_name}: {e}")

                if args.save_wavelet_diff :
                    try:
                        clean_gray = np.array(clean_pil.convert("L")) / 255.0
                        adv_gray = np.array(adv_clip_pil.convert("L")) / 255.0
                        coeffs_clean = pywt.dwt2(clean_gray, args.WAVELET_BASE)
                        coeffs_adv = pywt.dwt2(adv_gray, args.WAVELET_BASE)
                        LL_c, (LH_c, HL_c, HH_c) = coeffs_clean
                        LL_a, (LH_a, HL_a, HH_a) = coeffs_adv
                        diff_LL, diff_LH, diff_HL, diff_HH = np.abs(LL_c - LL_a), np.abs(LH_c - LH_a), np.abs(HL_c - HL_a), np.abs(HH_c - HH_a)
                        all_diffs = [diff_LL, diff_LH, diff_HL, diff_HH]
                        subplot_titles = ['LL', 'LH', 'HL', 'HH']

                        norm = Normalize(vmin=0.0, vmax=np.max([np.max(d) for d in all_diffs])) if args.UNIFIED_SCALE else None

                        fig, axes = plt.subplots(2, 2, figsize=(3.2, 3.0), dpi=300)
                        axes_flat = axes.flatten()
                        im = None
                        for idx, (ax, title, diff_data) in enumerate(zip(axes_flat, subplot_titles, all_diffs)):
                            im = ax.imshow(diff_data, cmap='jet', norm=norm)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.axis('off')
                            ax.set_title(title, fontsize=9, pad=2 if idx in [0, 1] else 2, y=-0.11 if idx in [2, 3] else None)

                        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.045, shrink=0.9)
                        cbar.ax.tick_params(labelsize=7)
                        plt.subplots_adjust(left=0.02, right=0.85, top=0.95, bottom=0.08, wspace=-0.02, hspace=0.01)
                        plt.savefig(os.path.join(sample_save_dir, "wavelet_diff.png"), bbox_inches='tight', pad_inches=0.02)
                        plt.close()
                    except Exception as e:
                        print(f"Failed to save wavelet diff: {e}")

            test_pbar.update(1)
    test_pbar.close()

    if args.compute_avg_energy and energy_count > 0:

        avg_energy = energy_accumulator / energy_count

        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        im = ax.imshow(avg_energy, cmap='jet', aspect='auto')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(os.path.join(args.save_dir, "average_frequency_energy.png"), dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Energy map saved")


if __name__ == "__main__":
    main()