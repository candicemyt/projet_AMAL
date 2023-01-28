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

        if self.transforms is not None:
            image, _ = self.transforms(image, "")
        return id, image


def compute_pairwise_sim(dataloader, file_writer, model, device):
    ids = torch.tensor([], device=device)
    #encoded_images = torch.tensor([], device=device)

    with torch.no_grad():
        print("Boucle d'encoding")
        for i, data in tqdm(enumerate(dataloader)):
            image_ids = (data[0]).to(device)
            images = (data[1]).to(device)
            ids = torch.cat((ids, image_ids))
            if i == 0:
                encoded_images = model.encode_image(images)
            else:
                encoded_images = torch.cat((encoded_images, model.encode_image(images)))

        print("Boucle de calcul de similarité")
        for i, image1_features in tqdm(enumerate(encoded_images)):
            similarities = []
            image_id = ids[i]
            for j, image2_features in enumerate(encoded_images):
                similarities.append(image2_features @ image1_features.T)
            index_sim_sorted = torch.argsort(torch.Tensor(np.array(similarities)).squeeze(), descending=True)
            ids_sorted = ids[index_sim_sorted]
            if ids_sorted[0] == image_id:
                file_writer.writerow([int(s) for s in ids_sorted[0:4]])
            else:
                file_writer.writerow([int(s) for s in [image_id] + [ids_sorted[0:3]]])



if __name__ == '__main__':
    # params
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_type = 'val'

    # init file
    pairwise_sim_path = f"pairwise_sim/{set_type}.csv"
    if path.exists(pairwise_sim_path):
        dataset_pairwise_sim_file = open(pairwise_sim_path, 'w')
    else:
        dataset_pairwise_sim_file = open(pairwise_sim_path, 'x')

    writer = csv.writer(dataset_pairwise_sim_file)

    # loading model
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(device)
    coco_dts = COCODatasetImSim(root='val2014/',
                                annFile='annotations/captions_val2014.json',
                                transform=preprocess)
    coco_loader = DataLoader(coco_dts, batch_size=8)
    compute_pairwise_sim(coco_loader, writer, model, device)
