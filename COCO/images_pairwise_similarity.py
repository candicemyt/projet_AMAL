import torch
import clip
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from os import path
import csv
import numpy as np
from tqdm import tqdm


class COCODatasetImSim(CocoDetection):
    def __init__(self, root, annFile, transforms=None, transform=None, target_transform=None):
        super().__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)

        if self.transform is not None: 
           image = self.transform(image)
        return id, image


def compute_pairwise_sim(dataloader, file_writer, model, device):
    ids = torch.tensor([], device=device)
    encoded_images = torch.tensor([], device=device)

    with torch.no_grad():
        print("Boucle d'encoding")
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_ids, images = data
            ids = torch.cat((ids, image_ids.to(device)))
            encoded_images = torch.cat((encoded_images, model.encode_image(images.to(device))))

        nb_images = encoded_images.shape[0]
        beg = (part-1) * (nb_images//5)
        end = beg + (nb_images//5)
        if part == 5:
            end = nb_images
        print("Boucle de calcul de similarité")
        for i, image1_features in tqdm(enumerate(encoded_images[beg:end]), total=end-beg):
            similarities = torch.zeros(encoded_images.shape[0], device=device)
            image_id = ids[i]
            for j, image2_features in enumerate(encoded_images):
                similarities[j] = image2_features @ image1_features.t()
            index_sim_sorted = torch.argsort(similarities.squeeze(), descending=True)
            ids_sorted = ids[index_sim_sorted]
            if ids_sorted[0] == image_id:
                file_writer.writerow([int(i) for i in ids_sorted[0:4]])
            else:
                new_row = [image_id] + list(ids_sorted[0:3])
                file_writer.writerow([int(i) for i in new_row])



if __name__ == '__main__':
    # params
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_type = 'train'
    part = 4

    # init file
    pairwise_sim_path = f"pairwise_sim/part{part}_{set_type}.csv"
    if path.exists(pairwise_sim_path):
        dataset_pairwise_sim_file = open(pairwise_sim_path, 'w')
    else:
        dataset_pairwise_sim_file = open(pairwise_sim_path, 'x')

    writer = csv.writer(dataset_pairwise_sim_file)

    # loading model
    model, preprocess = clip.load("ViT-B/32", device=device)
    coco_dts = COCODatasetImSim(root=f'{set_type}2014/',
                                annFile=f'annotations/captions_{set_type}2014.json',
                                transform=preprocess)
    coco_loader = DataLoader(coco_dts, batch_size=128)
    compute_pairwise_sim(coco_loader, writer, model, device)
