#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Implementation of Model Evaluation & Adversarial Attack
Core Components: Model Loading, Image Quality Metrics, Data Preprocessing
Compatible with PyTorch 2.x + torchvision 0.15+
For TPAMI Submission
"""
import math
import os
import sys
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
import lpips
from piq import fsim
from torch.utils import model_zoo
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from transformers import AutoModelForImageClassification
from cyclemlp import CycleMLP_B5
from transformers import AutoModelForImageClassification

# --------------------------- Global Constants (TPAMI Standard) ---------------------------
# ImageNet Normalization Parameters (Fixed for Academic Reproducibility)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# CycleMLP Model Configuration (Relative Path, Reproducible)
CYCLEMLP_FOLDER_NAME: str = "cycle_mlp"
CYCLEMLP_WEIGHT_NAME: str = "CycleMLP_B5.pth"
CYCLEMLP_CLASS_FILE: str = "cyclemlp"

# Gaussian Blur Kernel Parameters
BLUR_KERNEL_SIZE: int = 3
BLUR_PAD: int = 2
BLUR_SIGMA: float = 1.0
BLUR_CHANNELS: int = 3



# --------------------------- Utility Functions ---------------------------
def get_gaussian_kernel(
    kernel_size: int = BLUR_KERNEL_SIZE,
    pad: int = BLUR_PAD,
    sigma: float = BLUR_SIGMA,
    channels: int = BLUR_CHANNELS
) -> nn.Conv2d:
    """
    Generate a Gaussian blur kernel for image smoothing.
    Args:
        kernel_size: Size of the Gaussian kernel
        pad: Padding size for convolution
        sigma: Standard deviation of Gaussian distribution
        channels: Input image channels (default: 3 for RGB)
    Returns:
        Frozen convolutional layer with Gaussian kernel weights
    """
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Compute Gaussian kernel
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
    )
    gaussian_kernel /= torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    # Initialize frozen convolution layer
    gaussian_filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, padding=kernel_size - pad, bias=False
    )
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad_(False)
    return gaussian_filter

# Initialize Gaussian blur layer (global, lazy-loaded)
alf_layer: nn.Conv2d = get_gaussian_kernel()

# --------------------------- Robust Model Loader ---------------------------
def load_robust_model(model_name: str, device: torch.device) -> nn.Module:
    
    Load adversarially robust pre-trained models from official URLs/weights.
    Args:
        model_name: Name of the robust model
        device: Torch device (CPU/GPU) for model placement
    Returns:
        Pre-trained robust model in evaluation mode
    Raises:
        ValueError: If model name is not supported
    """
    model_urls = {
        'res50_sin': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'res50_sin_in': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'res50_sin_fine_in': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    }

    if model_name in model_urls.keys():
        model = torchvision.models.resnet50(weights=None)
        model = nn.DataParallel(model).to(device)
        checkpoint = model_zoo.load_url(model_urls[model_name], map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    elif model_name == 'res50_augmix':
        model = torchvision.models.resnet50(weights=None)
        model = nn.DataParallel(model).to(device)
        raise NotImplementedError(
            "Please download AugMix weights from official repo and replace the path here."
        )

    elif model_name == 'adv_incv3':
        model = timm.create_model('adv_inception_v3', pretrained=True)

    elif model_name == 'ens_inc_res_v2':
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)

    else:
        raise ValueError(f"Unsupported robust model: {model_name}")

    return model.eval().requires_grad_(False)


# --------------------------- Batch Model Loader (Full Test Suite) ---------------------------
def get_all_models(device: torch.device, weight_dtype: torch.dtype) -> List[Dict[str, Any]]:

    Load all benchmark models for comprehensive evaluation.
    Includes TorchVision, Timm, HuggingFace, and CycleMLP models.
    Args:
        device: Torch device (CPU/GPU)
        weight_dtype: Model precision (float16/float32/bfloat16)
    Returns:
        List of model dictionaries with name, model, input size
    """
    model_list: List[Dict[str, Any]] = []
    print("\n Initializing all benchmark models...")

    # 1. TorchVision Official Models
    tv_models = [
        ("Inc-v3", torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT, transform_input=False), 299),
        ("Res-152", torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT), 224),
        ("DN-121", torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT), 224),
        ("GoogleNet", torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT), 224),
        ("VGG-16", torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT), 224),
        ("Swin-B", torchvision.models.swin_b(weights=torchvision.models.Swin_B_Weights.DEFAULT), 224),
    ]
    for name, model, input_size in tv_models:
        model = model.to(device=device, dtype=weight_dtype)
        model_list.append({"name": name, "model": model, "input_size": input_size})
        print(f" {name} loaded successfully")

    # 2. Timm Models
    timm_models = [("inception_resnet_v2", "Inc-Res-v2", 299), ("inception_v4", "Inc-v4", 299)]
    for model_id, name, input_size in timm_models:
        try:
            model = timm.create_model(model_id, pretrained=True, num_classes=1000)
            model = model.to(device=device, dtype=weight_dtype)
            model_list.append({"name": name, "model": model, "input_size": input_size})
            print(f" {name} loaded successfully")
        except Exception as e:
            print(f" Failed to load {name}: {str(e)}")

    # 3. HuggingFace Models
    hf_models = [
        ("microsoft/swin-tiny-patch4-window7-224", "Swin-T"),
        ("google/mixer-base-patch16-224", "Mixer-B16"),
        ("facebook/deit-base-distilled-patch16-224", "DeiT-B"),
        ("google/vit-base-patch16-224", "ViT-B"),
        ("google/mixer-large-patch16-224", "Mixer-L"),
    ]
    for model_id, name in hf_models:
        try:
            model = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=weight_dtype)
            model = model.to(device=device, dtype=weight_dtype)
            model_list.append({"name": name, "model": model, "input_size": 224})
            print(f"{name} loaded successfully")
        except Exception as e:
            print(f" Failed to load {name}: {str(e)}")

    # 4. CycleMLP-B5 (Relative Path, Reproducible)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cyclemlp_path = os.path.join(current_dir, CYCLEMLP_FOLDER_NAME)
        sys.path.append(cyclemlp_path)


        model = CycleMLP_B5(pretrained=False, num_classes=1000)
        weight_path = os.path.join(cyclemlp_path, CYCLEMLP_WEIGHT_NAME)

        # Load weights safely
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        model = model.to(device=device, dtype=weight_dtype)
        model_list.append({"name": "CycleMLP-B5", "model": model, "input_size": 224})
        print(f" CycleMLP-B5 loaded successfully")
    except Exception as e:
        print(f" Failed to load CycleMLP-B5: {str(e)}")
        print(f"Place {CYCLEMLP_FOLDER_NAME} in the root directory with model files.")

    print(f"\n Total models loaded: {[m['name'] for m in model_list]}")
    return model_list

def load_single_classifier(model_type, device, weight_dtype):

    if model_type == "incv3":
        model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False
        input_size = 299
    elif model_type == "res152":
        model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
        input_size = 224
    elif model_type == "swin_tiny":
        model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        input_size = 224
    elif model_type == "mixer_b16":
        model = torchvision.models.mlp_mixer_b16(weights=torchvision.models.MLP_Mixer_B16_Weights.IMAGENET1K_V1)
        input_size = 224
    elif model_type == "deit_b":

        model = AutoModelForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
        input_size = 224
    elif model_type == "cycle_mlp":
        import sys
        from cyclemlp import CycleMLP_B5
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cyclemlp_path = os.path.join(current_dir, "cycle_mlp")
        sys.path.append(cyclemlp_path)
        model = CycleMLP_B5(pretrained=False, num_classes=1000)
        weight_path = os.path.join(cyclemlp_path, "CycleMLP_B5.pth")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        input_size = 224
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = model.to(device=device, dtype=weight_dtype)
    model.eval()
    return model, input_size

# --------------------------- Data Preprocessing ---------------------------
def fix_labels(test_set: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    """Fix ImageNet validation labels using val.txt mapping file."""
    val_dict = {}
    with open("val.txt", "r", encoding="utf-8") as f:
        for line in f:
            key, val = line.strip().split(',')
            val_dict[key.split('.')[0]] = int(val)

    new_samples = [(path, val_dict[path.split('/')[-1].split('.')[0]]) for path, _ in test_set.samples]
    test_set.samples = new_samples
    return test_set

def fix_labels_nips(
    test_set: torch.utils.data.Dataset,
    data_dir: str,
    label_flag: str = "N8",
    pytorch: bool = True,
    target_flag: bool = False,
    seed: int = 42
) -> torch.utils.data.Dataset:
    filenames = [os.path.basename(path) for path, _ in test_set.samples]

    def recover_image_id(fname):
        image_id = os.path.splitext(fname)[0]
        if image_id.endswith("-checkpoint"):
            image_id = image_id[:-len("-checkpoint")]
        return image_id
    
    image_classes = pd.read_csv(os.path.join(data_dir, "images.csv"))
    image_metadata = pd.DataFrame(
        {"ImageId": [recover_image_id(f) for f in filenames]}
    ).merge(image_classes[["ImageId", "TrueLabel"]], on="ImageId", how="left")
    if image_metadata["TrueLabel"].isnull().any():
        missing_ids = image_metadata[image_metadata["TrueLabel"].isnull()]["ImageId"].tolist()
        raise ValueError(f"Missing labels for: {missing_ids}")

    true_labels_raw = image_metadata["TrueLabel"].astype(int).tolist()

    base_target_set = get_classes(label_flag)
    base_target_set = [
        int(x.item()) if torch.is_tensor(x) else int(x)
        for x in base_target_set
    ]

    if pytorch:
        true_labels = [x - 1 for x in true_labels_raw]
        target_set = base_target_set
    else:
        true_labels = true_labels_raw
        target_set = [x + 1 for x in base_target_set]

    rng = random.Random(seed)

    label_dict = {}
    for fname, true_label in zip(filenames, true_labels):
        valid_targets = [x for x in target_set if x != true_label]
        if len(valid_targets) == 0:
            raise ValueError(f"No valid target for {fname}")
        target_label = rng.choice(valid_targets)
        label_dict[fname] = [true_label, target_label]

    new_samples = []
    for path, _ in test_set.samples:
        fname = os.path.basename(path)
        label = label_dict[fname][1] if target_flag else label_dict[fname][0]
        new_samples.append((path, label))

    test_set.samples = new_samples
    if hasattr(test_set, "targets"):
        test_set.targets = [label for _, label in new_samples]

    return test_set

def get_classes(label_flag: str) -> np.ndarray:
    """
    Get predefined class labels for targeted attacks.
    Args:
        label_flag: Label set identifier
    Returns:
        Numpy array of target class indices
    """
    if label_flag == 'N8':
        return np.array([150, 426, 843, 715, 952, 507, 590, 62])
    raise ValueError(f"Unsupported label set: {label_flag}")

# --------------------------- Image Normalization ---------------------------
def normalize_ddpm(x: torch.Tensor) -> torch.Tensor:
    """Normalize images to DDPM input range [-1, 1]."""
    return 2.0 * x - 1.0

def unnormalize_ddpm(x: torch.Tensor) -> torch.Tensor:
    """Reverse DDPM normalization to [0, 1]."""
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)

def normalize(x: torch.Tensor) -> torch.Tensor:
    """Standard ImageNet normalization for model inference."""
    out = torch.zeros_like(x)
    for i in range(3):
        out[:, i, :, :] = (x[:, i, :, :] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    return out

# --------------------------- Image Saving & Masking ---------------------------
def save_img(tensor: torch.Tensor, save_dir: str, name: str) -> None:
    """Save torch tensor as RGB image."""
    os.makedirs(save_dir, exist_ok=True)
    img = transforms.ToPILImage()(tensor.cpu().squeeze())
    img.save(os.path.join(save_dir, name))

def get_mask(
    batch_perturb: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
    patch_size: int = 32
) -> torch.Tensor:
    """Generate patch-based mask for adversarial perturbations."""
    n, c, h, w = batch_perturb.shape
    num_patch_h, num_patch_w = h // patch_size, w // patch_size
    mask_patch_num = int(num_patch_h * num_patch_w * mask_ratio)

    if mask_patch_num <= 0:
        return batch_perturb

    mask = torch.zeros(c, patch_size, patch_size, device=device)
    noise = torch.rand(n, num_patch_h * num_patch_w, device=device)
    mask_indices = torch.argsort(noise, dim=1)[:, :mask_patch_num]

    for i in range(n):
        for idx in mask_indices[i]:
            row, col = idx // num_patch_w, idx % num_patch_w
            batch_perturb[i, :, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] *= mask
    return batch_perturb

# --------------------------- Adversarial Constraint Functions ---------------------------
def clp(x: torch.Tensor) -> torch.Tensor:
    """Differentiable pixel value clipping to [-1, 1]."""
    return torch.clamp(x, -1.0, 1.0)

def budget(
    x: torch.Tensor,
    origin_img: torch.Tensor,
    eps: float,
    device: torch.device,
    attack_mode: str = "multi_targeted"
) -> torch.Tensor:
    """
    Differentiable L∞ perturbation budget constraint.
    Ensures adversarial perturbations stay within epsilon bound.
    """
    # Resize to match original image dimensions
    if x.shape[-2:] != origin_img.shape[-2:]:
        x = nn.functional.interpolate(x, size=origin_img.shape[-2:], mode='bilinear', align_corners=False)

    x = clp(x)
    delta = torch.clamp(x - origin_img, -eps, eps)
    return clp(origin_img + delta)

# --------------------------- Image Quality Metrics (For Evaluation) ---------------------------
def init_quality_metrics(device: torch.device) -> Dict[str, nn.Module]:
    """
    Initialize all image quality assessment metrics.
    Args:
        device: Torch device for metric computation
    Returns:
        Dictionary of initialized metric modules
    """
    return {
        "lpips": lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False),
        "ssim": SSIM(data_range=1.0).to(device),
        "psnr": PSNR(data_range=1.0).to(device),

    }



@torch.no_grad()
def calculate_image_quality(
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    metrics: Dict[str, nn.Module]
) -> Dict[str, float]:
    """
    Compute full set of image quality metrics between clean and adversarial images.
    Input images must be normalized to [0, 1].
    """
    return {
        "lpips": metrics["lpips"](clean_imgs, adv_imgs).mean().item(),
        "ssim": metrics["ssim"](adv_imgs, clean_imgs).item(),
        "psnr": metrics["psnr"](adv_imgs, clean_imgs).item(),
        "fsim": fsim(adv_imgs, clean_imgs, data_range=1.0).item(),

    }

def print_quality_metrics(metric_dict: Dict[str, float], count: int) -> None:
    """Print averaged image quality metrics in academic format."""
    print("\n" + "-" * 60)
    print(f"LPIPS:       {metric_dict['lpips'] / count:.6f}")
    print(f"SSIM:        {metric_dict['ssim'] / count:.6f}")
    print(f"PSNR:        {metric_dict['psnr'] / count:.2f}")
    print(f"FSIM:        {metric_dict['fsim'] / count:.6f}")

    print("-" * 60)
