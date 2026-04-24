# 📰 WCTD: Wavelet-Constrained Trajectory Distillation for Efficient Transferable Targeted Attacks

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/xuejianhuang/WCTD)](https://github.com/xuejianhuang/WCTD)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange)](https://pytorch.org/)

</div>

This repository contains the official implementation of the paper **Wavelet-Constrained Trajectory Distillation for Efficient Transferable Targeted Attacks**.

---

## 📌 Method Overview


<div align="center">
  <img src='./fig/model.png' width='80%' alt="WCTD Framework">
  <br>
  <em>Figure 1: Overall architecture of WCTD teacher-student distillation framework.</em>
</div>

---


## 📂 Datasets
<div align="center">

| Dataset | Usage | Link | Notes |
|---------|-------|------|-------|
| **ImageNet** | Training | [Download](https://image-net.org/download.php) | Full training set |
| **ImageNet-NeurIPS** | Testing | [Kaggle](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) | Standard adversarial benchmark |
| **MSCOCO val2017** | Testing   | [Download](http://images.cocodataset.org/zips/val2017.zip) | Cross-domain evaluation |

</div>

## 🧠 Victim Models

We evaluate WCTD against a diverse set of victim models spanning multiple architectures:

<details>
<summary><b>CNN-based Models (7)</b></summary>

- [Inception-v3](https://huggingface.co/litert-community/inception_v3/tree/main)
- [Inception-v4](https://huggingface.co/timm/inception_v4.tf_in1k/tree/main)
- [ResNet-152](https://huggingface.co/litert-community/resnet152)
- [Inception-ResNet-v2](https://huggingface.co/timm/inception_resnet_v2.tf_in1k/tree/main)
- [DenseNet-121](https://download.pytorch.org/models/densenet121-a639ec97.pth)
- [GoogleNet](https://download.pytorch.org/models/googlenet-1378be20.pth)
- [VGG-16](https://huggingface.co/timm/vgg16.tv_in1k/tree/main)

</details>

<details>
<summary><b>Transformer-based Models (4)</b></summary>

- [ViT-B/16](https://huggingface.co/google/vit-base-patch16-224/tree/main)
- [DeiT-B](https://huggingface.co/facebook/deit-base-distilled-patch16-224/tree/main)
- [Swin-T](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
- [Swin-B](https://huggingface.co/microsoft/swin-base-patch4-window7-224/tree/main)
</details>

<details>
<summary><b>MLP-based Models (3)</b></summary>

- [CycleMLP](https://github.com/ShoufaChen/CycleMLP?tab=readme-ov-file)
- [Mixer-B/16](https://huggingface.co/timm/mixer_b16_224.goog_in21k_ft_in1k/tree/main)
- [Mixer-L/16](https://huggingface.co/timm/mixer_l16_224.goog_in21k_ft_in1k)
</details>

## 🧪 Surrogate Models

Surrogate models used for attack generation:
- **CNN-based**
  - [Inception-v3](https://huggingface.co/litert-community/inception_v3/tree/main)
  - [ResNet-152](https://huggingface.co/litert-community/resnet152)
- **Transformer-based**
  - [Swin-T](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
  - [DeiT-B](https://huggingface.co/facebook/deit-base-distilled-patch16-224/tree/main)
- **MLP-based**
  - [CycleMLP](https://github.com/ShoufaChen/CycleMLP?tab=readme-ov-file)
  - [Mixer-B/16](https://huggingface.co/timm/mixer_b16_224.goog_in21k_ft_in1k/tree/main)

## ⚙️ Baseline Methods

We compare WCTD against state-of-the-art targeted attack methods:
- **Optimization-based**
  - [Logit](https://github.com/ZhengyuZhao/Targeted-Transfer)
  - [SU](https://github.com/zhipeng-wei/Self-Universality)
- **Generator-based**
  - [C-GSP](https://github.com/ShawnXYang/C-GSP)
  - [CGNC](https://github.com/ffhibnese/CGNC_Targeted_Adversarial_Attacks)
- **Diffusion / Flow-based**
   - [DiffAttack](https://github.com/WindVChen/DiffAttack)
   - [TGAF](https://github.com/TemenosMistral/TGAF)
   - [Dual_Flow](https://github.com/Chyxx/Dual-Flow)

## 📁 Project Structure

```
├── data/
│   ├── imagenet/
│   │   └── train/
│   │       ├── n01749939/
│   │       │   ├── n01749939_10126.JPEG
│   │       │   └── ...
│   │       └── ...
│   ├── imagenet-nips-val/
│   │   ├── categories.csv
│   │   ├── images.csv
│   │   └── images/
│   │       ├── 000b7d55b6184b08.png
│   │       └── ...
│   └── MSCOCO/ (optional)
│       └── val2017/
│           ├── 000000000139.jpg
│           └── ...
├── downloaded_pretrain_models/
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── unet/
│   └── vae/
├── imagenet_class_index.json
└── test_model/
    ├── Cycle_mlp/
    │   ├── CycleMLP_B5.pth
    │   └── cyclemlp.py
    └── other_models/
```

---

## 🚀 Training

### Teacher Training

```bash
python teacher_train.py 
  --pretrained_model_name_or_path="downloaded_pretrain_models" \ 
  --dataset_dir="/path/to/imagenet/train" \ 
  --output_root="/path/to/output" \ 
  --model_type=res152 \
  --attack_mode=multi_targeted \ 
  --train_batch_size=4 \
  --num_train_epochs=10 \
  --learning_rate=1e-04 \ 
  --seed=42 \
```

### Student Training

```bash
python student_train.py 
  --pretrained_model_name_or_path="downloaded_pretrain_models" \ 
  --dataset_dir="/path/to/imagenet/train" \
  --output_root="/path/to/output" \
  --model_type=res152 \
  --attack_mode=multi_targeted \ 
  --train_batch_size=4 \
  --num_train_epochs=10 \
  --learning_rate=1e-04 \
  --seed=42
```

## 🧪 Generating Adversarial Examples

```bash
python eval.py 
  --pretrained_model_name_or_path="downloaded_pretrain_models" \
  --dataset_dir="/data/imagenet-nips-val" \
  --student_lora_path="/output/final_model" \
  --save_dir="/output/eval_results" \
  --model_type=res152 \
  --attack_mode=multi_targeted \ 
  --test_batch_size=8 \
  --eps=16 \
  --seed=42 \
```
## 🛡️ Test  Models

### Robust Model Test

```bash
python inference.py \
  --dataset_dir /path/to/imagenet-nips-val/images \
  --data_dir /path/to/imagenet-nips-val \
  --pretrained_model_name_or_path downloaded_pretrain_models \
  --lora_path /path/to/your_modal.safetensors \
  --source_model_name res152 \
  --eval_model_type robust \
  --label_flag N8 \
  --batch_size 4
```
### All  Models Test

```bash
python inference.py 
  --dataset_dir /path/to/imagenet-nips-val/images \
  --data_dir /path/to/imagenet-nips-val \
  --pretrained_model_name_or_path downloaded_pretrain_models \
  --lora_path /path/to/your_modal.safetensors \
  --source_model_name res152 \
  --eval_model_type all \
  --label_flag N8 \
  --batch_size 4
```

## 🙏 Acknowledgements
We sincerely thank the following researchers for their valuable contributions:
*  Yixiao Chen (Tsinghua University, Beijing, China)
*  Shikun Sun (Tsinghua University, Beijing, China)
*  Hao Fang (Tsinghua University, Beijing, China)
*  Jianqi Chen (Beihang University, Beijing, China)
*  Zhenwei Shi (Beihang University, Beijing, China)
*  Hangyu Liu (Zhejiang University, Hangzhou, China)

## 📧 Contact
For questions or collaboration opportunities, please open an issue or contact the authors directly.
