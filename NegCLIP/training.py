from COCO_Dataset import COCODataset
from torch.utils.data import DataLoader
import clip
import torch
from tqdm import tqdm


def training(model, dataloader, dts_len):

    for i, data in tqdm(enumerate(dataloader), total=dts_len, leave=False):
        images, pos_captions, neg_captions, strong_alt_images, strong_alt_captions, neg_strong_alt_captions = data


if __name__ == "__main__":
    # params
    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
    SET_TYPE = "val"
    BATCH_SIZE = 32

    # load pretrain model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    # custom dataset
    coco_dts = COCODataset(root=f'../COCO{SET_TYPE}2014/', pairwise_sim_path=f"../COCO/pairwise_sim/{SET_TYPE}.csv",
                           annFile=f'../COCO/annotations/captions_negcaptions_{SET_TYPE}2014.json',
                           transform=preprocess, target_transform=clip.tokenize)

    coco_loader = DataLoader(coco_dts, shuffle=False, batch_size=BATCH_SIZE)