from COCO_Dataset import COCODataset
from torch.utils.data import DataLoader
import clip
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from ARO_benchmark.evaluation_ARO import VGDataset

# TODO : evaluate on COCO-order

def evaluate(dataloader, model, device):
    """
    Evaluate model on the sataset in the dataloader
    :param dataloader: pytorch dataloader
    :param model : model to evaluate
    :param device : cuda or cpu
    :return: accuracy
    """
    acc = 0
    for _, vgr_data in enumerate(dataloader):
        image, captions = vgr_data
        captions = captions.flatten(0, 1).to(device)
        image = image.to(device)
        logits_per_image, logits_per_text = model(image, captions)
        probs = logits_per_image.softmax(dim=-1).squeeze()
        if probs[0] > probs[1]:
            acc += 1
    return acc / len(dataloader)


def training(model, optimizer, scheduler, train_loader, val_loader, vgr_loader, vga_loader, max_epochs, device):
    writer = SummaryWriter()

    for epoch in tqdm(range(max_epochs)):
        # TRAIN
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            images, captions = data
            images = images.to(device)
            captions = captions.to(device)
            captions_pos = captions[:, 0:2, :].flatten(0, 1)
            captions_neg = captions[:, 2:, :].flatten(0, 1)
            images = images.flatten(0, 1)

            # encoding + cosine similarity as logits
            logits_per_image, logits_per_text = model(images, torch.cat((captions_pos, captions_neg)))
            logits_per_image = logits_per_image[:, 2 * BATCH_SIZE:]  # keeping only the true captions to compute loss
            logits_per_text = logits_per_image.t()

            # loss
            labels = torch.arange(2 * BATCH_SIZE)
            loss_text = cross_entropy(logits_per_text, labels)
            loss_image = cross_entropy(logits_per_image, labels)
            loss = (loss_text + loss_image) / 2
            loss.backward()
            optimizer.step()
            scheduler.step()

            # evaluate on ARO
            acc_vgr = evaluate(vgr_loader, model, device)
            acc_vga = evaluate(vga_loader, model, device)

            # logs
            step = epoch * len(train_loader) + i
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("loss/image/train", loss_image.item(), step)
            writer.add_scalar("loss/text/train", loss_text.item(), step)
            writer.add_scalar("evaluate/VGR/train", acc_vgr / len(vgr_loader), step)
            writer.add_scalar("evaluate/VGA/train", acc_vga / len(vga_loader), step)

            # save weights
            torch.save(model.state_dict(), f"weights/epoch{epoch}_step{step}.pth")

        # VAL
        for i, data in enumerate(val_loader):

            images, captions = data
            images = images.to(device)
            captions = captions.to(device)
            captions_pos = captions[:, 0:2, :].flatten(0, 1)
            captions_neg = captions[:, 2:, :].flatten(0, 1)
            images = images.flatten(0, 1)

            # encoding + cosine similarity as logits
            logits_per_image, logits_per_text = model(images, torch.cat((captions_pos, captions_neg)))
            logits_per_image = logits_per_image[:, 2 * BATCH_SIZE:]  # keeping only the true captions to compute loss
            logits_per_text = logits_per_image.t()

            # loss
            labels = torch.arange(2 * BATCH_SIZE)
            loss_text = cross_entropy(logits_per_text, labels)
            loss_image = cross_entropy(logits_per_image, labels)
            loss = (loss_text + loss_image) / 2

            # evaluate on ARO
            acc_vgr = evaluate(vgr_loader, model, device)
            acc_vga = evaluate(vga_loader, model, device)

            # logs
            step = epoch * len(val_loader) + i
            writer.add_scalar("loss/val", loss.item(), step)
            writer.add_scalar("loss/image/val", loss_image.item(), step)
            writer.add_scalar("loss/text/val", loss_text.item(), step)
            writer.add_scalar("evaluate/VGR/val", acc_vgr / len(vgr_loader), step)
            writer.add_scalar("evaluate/VGA/val", acc_vga / len(vga_loader), step)


if __name__ == "__main__":
    # hyper params
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SET_TYPE = "val"
    BATCH_SIZE = 2
    MAX_EPOCHS = 5
    WARMUP_STEPS = 50
    VALSET_SIZE = 0.15
    SHUFFLE_DTS = False
    LR = 1e-5  # TODO : to check
    VGA_VGR_PATH = "../ARO_benchmark/VGA_VGR/"

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

    # setup dataloader for evaluation
    vgr_dts = VGDataset(VGA_VGR_PATH + f"/train/dataset_relations.csv", VGA_VGR_PATH + "images",
                        20, image_tranform=preprocess, text_transform=clip.tokenize)
    vgr_loader = DataLoader(vgr_dts)

    vga_dts = VGDataset(VGA_VGR_PATH + f"/train/dataset_attributes.csv", VGA_VGR_PATH + "images",
                        20, image_tranform=preprocess, text_transform=clip.tokenize)
    vga_loader = DataLoader(vga_dts)

    # optimizer
    optim = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optim, WARMUP_STEPS)

    training(model, optim, scheduler, train_loader, val_loader, vgr_loader, vga_loader, MAX_EPOCHS, DEVICE)
