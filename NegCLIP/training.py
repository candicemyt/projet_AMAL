from COCO_Dataset import COCODataset
from torch.utils.data import DataLoader
import clip
import torch

device = "cuda" if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load("ViT-B/32", device=device)

# custom dataset
coco_dts = COCODataset(root='../COCO/val2014/', pairwise_sim_path="../COCO/pairwise_sim/val.csv",
                       annFile='../COCO/annotations/captions_val2014.json', transform=preprocess)

coco_loader = DataLoader(coco_dts, shuffle=False, batch_size=32)
with torch.no_grad():
    for i, data in enumerate(coco_loader):
        image, pos_captions, neg_captions, strong_alt_image, strong_alt_captions, neg_strong_alt_captions = data
        print("pos caption", pos_captions)
        print("neg caption", neg_captions)
        print("pos strong alt caption", strong_alt_captions)
        print("neg strong alt caption", neg_strong_alt_captions)
        break
