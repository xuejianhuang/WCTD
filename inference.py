import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from diffusers.utils import convert_all_state_dict_to_peft

from utils import (
    fix_labels_nips,
    normalize,
    unnormalize_ddpm,
    budget,
    get_all_models,
    load_robust_model,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="downloaded_pretrain_models")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label_to_caption_path", type=str, default="imagenet_class_index.json")
    parser.add_argument("--lora_path", type=str, required=True)

    parser.add_argument("--source_model_name", type=str, default="student_lora")
    parser.add_argument("--eval_model_type", type=str, default="robust")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--eps", type=int, default=16)
    parser.add_argument("--attack_mode", type=str, default="multi_targeted")
    parser.add_argument("--s_for", type=int, default=6)
    parser.add_argument("--s_gen", type=int, default=6)
    parser.add_argument("--num_timesteps", type=int, default=20)
    parser.add_argument("--label_flag", type=str, default="N8")
    parser.add_argument("--base_size", type=int, default=224)

    return parser.parse_args()


def log(message):
    print(message, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlatImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        self.samples = []
        for name in sorted(os.listdir(image_dir)):
            path = os.path.join(image_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in self.extensions:
                self.samples.append((path, 0))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.targets = [0] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.input_ids


def resize_batch(x, size):
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def format_table(rows, headers):
    widths = []
    for i, h in enumerate(headers):
        max_len = len(str(h))
        for row in rows:
            max_len = max(max_len, len(str(row[i])))
        widths.append(max_len)

    def make_line():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def make_row(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)) + " |"

    lines = [make_line(), make_row(headers), make_line()]
    for row in rows:
        lines.append(make_row(row))
    lines.append(make_line())
    return "\n".join(lines)


def is_corrupted_checkpoint_error(error):
    msg = str(error).lower()
    return (
        "unexpected eof" in msg
        or "corrupted" in msg
        or "failed finding central directory" in msg
        or "invalid header" in msg
    )


def remove_latest_torch_checkpoint():
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    if not os.path.isdir(cache_dir):
        return None

    valid_exts = (".pth", ".pt", ".tar", ".pth.tar", ".zip")
    candidates = [
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.endswith(valid_exts)
    ]

    if len(candidates) == 0:
        return None

    latest = max(candidates, key=os.path.getmtime)
    os.remove(latest)
    return latest


def load_robust_model_with_retry(name, device):
    try:
        return load_robust_model(name, device)
    except RuntimeError as error:
        if not is_corrupted_checkpoint_error(error):
            raise

        removed = remove_latest_torch_checkpoint()
        if removed is not None:
            log(f"[WARN] Removed corrupted checkpoint: {removed}")
        else:
            log("[WARN] Corrupted checkpoint detected, but no cached checkpoint was found.")

        log(f"[INFO] Retrying robust model loading: {name}")
        return load_robust_model(name, device)


def load_eval_model_list(eval_model_type, weight_dtype):
    cpu_device = torch.device("cpu")
    log(f"[INFO] Loading evaluation models: {eval_model_type}")

    if eval_model_type == "all":
        model_list = get_all_models(cpu_device, weight_dtype)
        log(f"[INFO] Loaded {len(model_list)} evaluation models.")
        return model_list

    robust_names = [
        "adv_incv3",
        "ens_inc_res_v2",
        "res50_sin",
        "res50_sin_in",
        "res50_sin_fine_in",
    ]

    model_list = []

    for idx, name in enumerate(robust_names, start=1):
        log(f"[INFO] Loading robust model [{idx}/{len(robust_names)}]: {name}")
        model = load_robust_model_with_retry(name, cpu_device)
        input_size = 299 if name in ["adv_incv3", "ens_inc_res_v2"] else 224

        model_list.append({
            "name": name,
            "model": model,
            "input_size": input_size,
        })

        log(f"[INFO] Loaded robust model [{idx}/{len(robust_names)}]: {name}")

    log("[INFO] Finished loading robust models.")
    return model_list


def get_running_average_asr(eval_model_list, total_correct, total_count):
    values = []
    for item in eval_model_list:
        name = item["name"]
        if total_count[name] > 0:
            values.append(100.0 * total_correct[name] / total_count[name])
    return float(np.mean(values)) if len(values) > 0 else 0.0


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    eps = args.eps / 255.0

    torch.set_float32_matmul_precision("high")

    log(f"[INFO] Device: {device}")
    log("[INFO] Loading diffusion components...")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    )
    unet_origin = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    )

    ddim_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    ddim_inv_scheduler = DDIMInverseScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    ddim_inv_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps_inverse = ddim_inv_scheduler.timesteps[:args.s_for]

    ddim_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps = ddim_scheduler.timesteps[-args.s_gen:]

    scaling_factor = vae.config.scaling_factor

    log("[INFO] Loading LoRA weights...")

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet.add_adapter(unet_lora_config)

    if args.lora_path is None or not os.path.exists(args.lora_path):
        raise FileNotFoundError(f"LoRA not found: {args.lora_path}")

    lora_state_dict = load_file(args.lora_path)
    unet_lora_state_dict = {
        key[len("unet."):]: val
        for key, val in lora_state_dict.items()
        if key.startswith("unet.")
    }
    unet_lora_state_dict = convert_all_state_dict_to_peft(unet_lora_state_dict)
    set_peft_model_state_dict(unet, unet_lora_state_dict)

    def freeze_and_move(model):
        model.requires_grad_(False)
        return model.to(device=device, dtype=weight_dtype)

    text_encoder = freeze_and_move(text_encoder)
    vae = freeze_and_move(vae)
    unet_origin = freeze_and_move(unet_origin)
    unet = unet.to(device=device, dtype=weight_dtype)
    unet.eval()

    log("[INFO] Preparing dataset...")

    with open(args.label_to_caption_path, "r") as f:
        label_to_caption = json.load(f)

    test_transforms = transforms.Compose([
        transforms.Resize(args.base_size),
        transforms.CenterCrop(args.base_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    test_set = FlatImageDataset(args.dataset_dir, test_transforms)
    test_set = fix_labels_nips(
        test_set=test_set,
        data_dir=args.data_dir,
        label_flag=args.label_flag,
        pytorch=True,
        target_flag=True,
        seed=args.seed
    )

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    log(f"[INFO] Dataset size: {len(test_set)}")
    log(f"[INFO] Number of batches: {len(loader)}")

    eval_model_list = load_eval_model_list(args.eval_model_type, weight_dtype)

    total_correct = {item["name"]: 0 for item in eval_model_list}
    total_count = {item["name"]: 0 for item in eval_model_list}

    unet_in_transform = transforms.Resize(352)

    log("[INFO] Starting evaluation...")

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", ncols=120)

        for batch_idx, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(device=device, dtype=weight_dtype)
            labels = labels.to(device=device, dtype=torch.long)

            clean_imgs = unnormalize_ddpm(imgs)

            captions = [
                f"a photo of {label_to_caption[str(int(label.item()))][1]}"
                for label in labels
            ]
            captions = tokenize_captions(tokenizer, captions).to(device)
            empty_captions = tokenize_captions(tokenizer, [""] * imgs.size(0)).to(device)

            latents = vae.encode(unet_in_transform(imgs)).latent_dist.sample()
            latents = latents * scaling_factor

            encoder_hidden_states = text_encoder(captions, return_dict=False)[0]
            encoder_hidden_states_empty = text_encoder(empty_captions, return_dict=False)[0]

            for t in timesteps_inverse:
                model_pred = unet_origin(latents, t, encoder_hidden_states_empty, return_dict=False)[0]
                latents = ddim_inv_scheduler.step(model_pred, t, latents, return_dict=False)[0]

            pred_original_sample = None
            for t in timesteps:
                model_pred = unet(latents, t, encoder_hidden_states, return_dict=False)[0]
                latents, pred_original_sample = ddim_scheduler.step(
                    model_pred, t, latents, return_dict=False
                )

            adv_imgs = vae.decode(pred_original_sample / scaling_factor, return_dict=False)[0]
            adv_imgs = resize_batch(unnormalize_ddpm(adv_imgs), args.base_size)
            adv_imgs = budget(adv_imgs, clean_imgs, eps,args.attack_mode)

            for item in eval_model_list:
                name = item["name"]
                model = item["model"].to(device)
                model.eval()

                eval_imgs = resize_batch(adv_imgs, item["input_size"])
                out = model(normalize(eval_imgs))
                if hasattr(out, "logits"):
                    out = out.logits

                pred = out.argmax(-1)
                total_correct[name] += (pred == labels).sum().item()
                total_count[name] += labels.size(0)

                model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_asr = get_running_average_asr(eval_model_list, total_correct, total_count)

            pbar.set_postfix({
                "batch": f"{batch_idx}/{len(loader)}",
                "avg_asr": f"{avg_asr:.2f}",
            })

    rows = []
    asr_values = []

    for item in eval_model_list:
        name = item["name"]
        asr = 100.0 * total_correct[name] / total_count[name] if total_count[name] > 0 else 0.0
        asr_values.append(asr)
        rows.append([args.source_model_name, name, f"{asr:.2f}"])

    if len(asr_values) > 0:
        rows.append([args.source_model_name, "Average", f"{np.mean(asr_values):.2f}"])

    log("[INFO] Evaluation finished.")
    print(format_table(rows, headers=["Source Model", "Target Model", "ASR (%)"]))


if __name__ == "__main__":
    main()
