import os
import numpy as np
import torch
from torchvision import datasets, transforms
import argparse
from utils import get_classes, normalize, get_all_models, load_robust_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, default='generated_final_results_11111')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_type", type=str, default='robust')  # 'all' or 'robust'

    args = parser.parse_args()
    return args

args = parse_args()
TEST_DIR = args.test_dir
BATCH_SIZE = args.batch_size
MODEL_TYPE = args.model_type
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weight_dtype = torch.float32
results_dict = {}


if MODEL_TYPE == 'all':
    model_list = get_all_models(device, weight_dtype)
else:
    robust_names = ['adv_incv3', 'ens_inc_res_v2', 'res50_sin', 'res50_sin_in', 'res50_sin_fine_in']
    model_list = []
    for name in robust_names:
        model = load_robust_model(name, device).to(device)
        input_size = 299 if name in ['adv_incv3', 'ens_inc_res_v2'] else 224
        model_list.append({"name": name, "model": model, "input_size": input_size})


class_ids = get_classes('N8')

for item in model_list:
    name, model, img_size = item["name"], item["model"], item["input_size"]
    print(f"\nEvaluating: {name}")
    model.eval()


    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    sr_list = []
    for cid in class_ids:
        test_dir = os.path.join(TEST_DIR, str(cid))
        if not os.path.exists(test_dir):
            sr_list.append(0.0)
            continue

        total_correct, total_samples = 0, 0
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(test_dir, transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        with torch.no_grad():
            for img, _ in loader:
                img = img.to(device)
                out = model(normalize(img))

                if any(m in name for m in
                       ['swin_tiny', 'mixer_b16', 'deit_b', 'cycle_mlp', 'Swin-T', 'Mixer-B16', 'DeiT-B',
                        'CycleMLP-B5']):
                    out = out.logits

                total_correct += (out.argmax(-1) == cid).sum().item()
                total_samples += img.size(0)

        sr = total_correct / total_samples if total_samples > 0 else 0.0
        sr_list.append(sr)
        print(f"  Class {cid}: {sr:.2%}")

    mean_sr = np.mean(sr_list)
    results_dict[name] = float(mean_sr * 100)
    print(f"Mean ASR: {mean_sr:.2%}")

    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("Final Results (ASR %):")
for k, v in results_dict.items():
    print(f"{k:<20} | {v:.2f}%")
print("=" * 50)
