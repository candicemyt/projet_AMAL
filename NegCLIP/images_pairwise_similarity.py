import torch
import clip
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
from os import path
import csv
import numpy as np
# TODO : add tqdm and preprocess images by batch
# TODO clean (function, main , ...)
# TODO à changer dossier

device = "cuda" if torch.cuda.is_available() else "cpu"
set_type = 'val'
pairwise_sim_path = f"../COCO/pairwise_sim/{set_type}.csv"

if path.exists(pairwise_sim_path):
    dataset_pairwise_sim_file = open(pairwise_sim_path, 'w')
else:
    dataset_pairwise_sim_file = open(pairwise_sim_path, 'x')

writer = csv.writer(dataset_pairwise_sim_file)



model, preprocess = clip.load("ViT-B/32", device=device)

coco_dts = CocoCaptions(root='../COCO/val2014/',
                        annFile='../COCO/annotations/captions_val2014.json',
                        transform=preprocess)
coco_loader = DataLoader(coco_dts)

encoded_images = []
with torch.no_grad():
    # boucle de processing / encoding
    for image, _ in coco_loader:
        encoded_images.append(model.encode_image(image))

    # boucle de calcul de similarité
    for image_features in encoded_images:
        similarities = []
        for image2_features in encoded_images:
            similarities.append(image2_features.cpu().numpy() @ image_features.cpu().numpy().T)
        similarities_sorted = torch.argsort(torch.Tensor(np.array(similarities)).squeeze(), descending=True)
        writer.writerow([int(s) for s in similarities_sorted[0:4]])

