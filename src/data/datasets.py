import re
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image, ImageDraw

# Parse segmentation string utility
def parse_segmentation_string(seg_str):
    polygons = []
    if seg_str is None:
        return polygons
    seg_blocks = re.findall(r"<seg>(.*?)</seg>", seg_str)
    for block in seg_blocks:
        coords = re.findall(r"\(([\d\.]+),\s*([\d\.]+)\)", block)
        if coords:
            poly = np.array([[float(x), float(y)] for x, y in coords], dtype=np.float32)
            polygons.append(poly)
    return polygons

class GRefCocoTorchDataset(data.Dataset):
    def __init__(self, hf_dataset, image_size=224, train=True):
        self.ds = hf_dataset
        self.image_size = image_size
        self.train = train
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        record = self.ds[idx]
        pil_img = record["images"][0].convert("RGB")
        orig_w, orig_h = pil_img.size
        original_image = pil_img.copy()
        img_t = self.img_transform(pil_img)
        txt = record.get("problem", "")
        seg_str = record.get("answer", "")
        polygons = parse_segmentation_string(seg_str)
        mask_pil = Image.new("L", (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask_pil)
        for poly in polygons:
            if len(poly) >= 3:
                draw.polygon([tuple(p) for p in poly], outline=1, fill=1)
        mask_resized = mask_pil.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_np = np.array(mask_resized, dtype=np.uint8)
        mask_t = torch.from_numpy(mask_np).float()
        gt_masks = mask_t.unsqueeze(0)
        return {
            "image": img_t,
            "text": txt,
            "gt_masks": gt_masks,
            "original_image": original_image,
            "orig_size": (orig_h, orig_w),
            "id": record.get("id", None),
        }

def grefcoco_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    gt_masks_list = [b["gt_masks"] for b in batch]
    original_images = [b["original_image"] for b in batch]

    return images, texts, gt_masks_list, original_images
