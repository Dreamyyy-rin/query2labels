import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import _init_paths
from dataset.get_dataset import get_datasets
from models.query2label import build_q2l
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict
import torchvision.transforms as transforms


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

attention_weights_sa = []
attention_weights_ca = []

def get_attention_storage_hook(storage_list):
    def hook(model, input, output):
        storage_list.append(output[1])
    return hook

def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * std + mean) * 255
    return tensor.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Visualisasi Attention Map Query2Label')
    parser.add_argument('--model_path', required=True, help='Path ke file model .pkl')
    parser.add_argument('--config_file', required=True, help='Path ke file konfigurasi .json')
    parser.add_argument('--dataset_dir', required=True, help='Path ke direktori dataset COCO')
    parser.add_argument('--num_images', type=int, default=5, help='Jumlah gambar untuk divisualisasikan')
    parser.add_argument('--output_dir', default='attention_maps_output', help='Folder untuk menyimpan hasil')
    
    parser.add_argument('--img_size', default=448, type=int)
    parser.add_argument('--dataname', default='coco14')
    parser.add_argument('--orid_norm', action='store_true', default=False)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output akan disimpan di: {args.output_dir}")

    print("Memuat model...")
    cfg = get_raw_dict(args.config_file)
    model_args = cfg['model']
    model = build_q2l(model_args)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
    
    decoder_layer_index = model_args['dec_layers'] - 1
    
    hook_handle_ca = model.transformer.decoder.layers[decoder_layer_index].multihead_attn.register_forward_hook(
        get_attention_storage_hook(attention_weights_ca)
    )
    for layer in model.transformer.decoder.layers:
        layer.multihead_attn.need_weights = True

    model.cuda()
    model.eval()

    print("Memuat dataset...")
    _, val_dataset = get_datasets(args)
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.num_images,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    print("\nMemulai proses visualisasi...")
    with torch.no_grad():
        images, targets = next(iter(data_loader))
        images = images.cuda()
        
        output = model(images)
        
        attn_map = attention_weights_ca[0].cpu()

        for i in range(args.num_images):
            img_tensor = images[i]
            img_np = denormalize_image(img_tensor)
            true_labels_indices = torch.where(targets[i] == 1)[0]
            
            if len(true_labels_indices) == 0:
                print(f"Gambar {i+1} tidak memiliki label ground truth, dilewati.")
                continue

            num_labels = len(true_labels_indices)
            fig, axes = plt.subplots(1, num_labels + 1, figsize=(5 * (num_labels + 1), 5))
            
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            print(f"\nMemproses Gambar {i+1} dengan label asli:")
            
            for j, label_idx in enumerate(true_labels_indices):
                label_name = COCO_CLASS_NAMES[label_idx]
                print(f"- {label_name}")
                
                head_mean_attn = attn_map[i, :, label_idx, :].mean(dim=0)
                
                feature_map_size = args.img_size // 32
                heatmap = head_mean_attn.reshape(feature_map_size, feature_map_size).numpy()
                
                heatmap = cv2.resize(heatmap, (args.img_size, args.img_size))
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                
                ax = axes[j+1]
                ax.imshow(superimposed_img)
                ax.set_title(f"Attention for: {label_name}")
                ax.axis('off')

            plt.tight_layout()
            output_path = os.path.join(args.output_dir, f"attention_map_image_{i+1}.jpg")
            plt.savefig(output_path)
            plt.close(fig)
            print(f"Visualisasi disimpan ke {output_path}")

    hook_handle_ca.remove()
    print("\n--- Proses Visualisasi Selesai! ---")

if __name__ == '__main__':
    main()