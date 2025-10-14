import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from lib.models import build_model
from lib.config import cfg, update_config_from_file
import numpy as np

def visualize_attention(model, image_path, output_dir, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        top_indices = np.where(probs > threshold)[0]

    print(f"Detected labels ({len(top_indices)}): {top_indices}")

    plt.imshow(image)
    plt.title(f"Predicted labels: {len(top_indices)} found")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"vis_{img_name}"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pic_path", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--output_dir", default="output_maps")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    update_config_from_file(cfg, args.config_file)
    model = build_model(cfg, is_train=False)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    visualize_attention(model, args.pic_path, args.output_dir, args.threshold)
