from COCO_Dataset import COCODataset
from torch.utils.data import DataLoader
import clip
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter


def training(model, optimizer, scheduler, train_loader, val_loader, dts_len, max_epochs):

    writer = SummaryWriter()
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

    for epoch in tqdm(range(max_epochs)):
        # TRAIN
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            images, pos_captions, neg_captions, strong_alt_images, strong_alt_captions, neg_strong_alt_captions = data

            optimizer.zero_grad()

            text_features = model.encode_text(torch.concatenate((pos_captions, strong_alt_captions, neg_captions, neg_strong_alt_captions)))
            image_features = model.encode_image(torch.concatenate((images, strong_alt_images)))
            mat_size = image_features.shape[1]
            print(image_features.shape)
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # loss
            labels = torch.arange(mat_size)
            # TODO : prendre que les pos captions
            print(logits_per_text.shape)
            print(logits_per_image.shape)
            break
            loss_text = CrossEntropyLoss(logits_per_text, labels)
            loss_image = CrossEntropyLoss(logits_per_image, labels)
            loss = (loss_text + loss_image) / 2
            loss.backward()
            optimizer.step()
            scheduler.step()

            # logs
            step = epoch * len(train_loader) + i
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("loss image/train", loss_image.item(), step)
            writer.add_scalar("loss text/train", loss_text.item(), step)
            #writer.add_scalar("accuracy/train", , step)

            # save weights
            torch.save(model.state_dict(), f"weights/epoch{epoch}_step{step}.pth")

            # VAL
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
                images, pos_captions, neg_captions, strong_alt_images, strong_alt_captions, neg_strong_alt_captions = data


                text_features = model.encode_text(torch.concatenate((pos_captions, strong_alt_captions, neg_captions, neg_strong_alt_captions)))
                image_features = model.encode_image(torch.concatenate((images, strong_alt_images)))
                mat_size = image_features.shape[1]

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                # loss
                labels = torch.arange(mat_size)
                # TODO : prendre que les pos captions
                loss_text = CrossEntropyLoss(logits_per_text, labels)
                loss_image = CrossEntropyLoss(logits_per_image, labels)
                loss = (loss_text + loss_image) / 2


                # logs
                step = epoch * len(train_loader) + i
                writer.add_scalar("loss/val", loss.item(), step)
                writer.add_scalar("loss image/val", loss_image.item(), step)
                writer.add_scalar("loss text/val", loss_text.item(), step)
                # writer.add_scalar("accuracy/train", , step)


if __name__ == "__main__":
    # hyper params
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SET_TYPE = "val"
    BATCH_SIZE = 1024
    OPTIM_BATCH_SIZE = 1024
    MAX_EPOCHS = 5
    WARMUP_STEPS = 50
    VALSET_SIZE = 0.15
    SHUFFLE_DTS = False
    LR = 1e-5

    # load pretrain model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    # setup dataset an dataloader
    coco_dts = COCODataset(root=f'../COCO/{SET_TYPE}2014/', pairwise_sim_path=f"../COCO/pairwise_sim/{SET_TYPE}.csv",
                           annFile=f'../COCO/annotations/captions_negcaptions_{SET_TYPE}2014.json',
                           transform=preprocess, target_transform=clip.tokenize)
    dts_size = len(coco_dts)

    split = int(np.floor(VALSET_SIZE * dts_size))
    if SHUFFLE_DTS:
        indices = torch.randperm(dts_size)
    else:
        indices = list(range(dts_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(coco_dts, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(coco_dts, batch_size=BATCH_SIZE, sampler=val_sampler)

    # optimizer
    optim = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optim, len(train_loader) * MAX_EPOCHS)

    training(model, optim, scheduler, train_loader, val_loader, dts_size, MAX_EPOCHS)