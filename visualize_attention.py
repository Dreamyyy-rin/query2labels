import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import matplotlib.pyplot as plt

import _init_paths
from dataset.get_dataset import get_datasets
from models.query2label import build_q2l
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict

COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

attention_weights_ca = []

def get_attention_storage_hook(storage_list):
    def hook(model, input, output):
        storage_list.append(output[1])
    return hook

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main():
    parser = argparse.ArgumentParser(description='Visualisasi Attention Map Query2Label')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--pic_path', required=True)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--img_size', default=448, type=int)
    parser.add_argument('--output_dir', default='attention_maps_output')
    parser.add_argument('--threshold', default=0.5, type=float)
    
    # Argumen dummy yang diperlukan oleh fungsi internal
    parser.add_argument('--dataname', default='coco14')
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--orid_norm', action='store_true', default=False)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output akan disimpan di: {args.output_dir}")

    print("Memuat model...")
    with open(args.config_file, 'r') as f:
        cfg = json.load(f)
    model_args = cfg['model']
    model = build_q2l(model_args)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
    
    decoder_layer_index = model_args['dec_layers'] - 1
    hook_handle = model.transformer.decoder.layers[decoder_layer_index].multihead_attn.register_forward_hook(
        get_attention_storage_hook(attention_weights_ca)
    )
    for layer in model.transformer.decoder.layers:
        layer.multihead_attn.need_weights = True

    model.cuda()
    model.eval()

    print("Memproses gambar...")
    image = Image.open(args.pic_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img_tensor)
        preds = torch.sigmoid(output)[0]

    attn_map = attention_weights_ca[0][0].cpu() # Ambil atensi dari gambar pertama di batch

    predicted_indices = torch.where(preds > args.threshold)[0]
    
    if len(predicted_indices) == 0:
        print(f"Tidak ada label yang terdeteksi di atas threshold {args.threshold}")
        return

    num_labels = len(predicted_indices)
    fig, axes = plt.subplots(1, num_labels + 1, figsize=(5 * (num_labels + 1), 5))
    
    original_img_display = denormalize_image(img_tensor.squeeze(0).cpu())
    axes[0].imshow(original_img_display.permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    print(f"\\nMembuat visualisasi untuk label yang terdeteksi:")
    
    for j, label_idx in enumerate(predicted_indices):
        label_name = COCO_CLASS_NAMES[label_idx]
        score = preds[label_idx].item()
        print(f"- {label_name} (Score: {score:.2f})")
        
        head_mean_attn = attn_map[:, label_idx, :].mean(dim=0)
        
        feature_map_size = args.img_size // 32
        heatmap = head_mean_attn.reshape(feature_map_size, feature_map_size).numpy()
        
        heatmap = cv2.resize(heatmap, (args.img_size, args.img_size))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        img_np_for_blend = np.array(original_img_display.permute(1, 2, 0))
        superimposed_img = cv2.addWeighted(img_np_for_blend, 0.6, heatmap, 0.4, 0)
        
        ax = axes[j+1]
        ax.imshow(superimposed_img)
        ax.set_title(f"Attention for: {label_name}")
        ax.axis('off')

    plt.tight_layout()
    output_filename = os.path.basename(args.pic_path)
    output_path = os.path.join(args.output_dir, f"attention_{output_filename}")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Visualisasi disimpan ke {output_path}")

    hook_handle.remove()
    print("\\n--- Proses Visualisasi Selesai! ---")

if __name__ == '__main__':
    main()