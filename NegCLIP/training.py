import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

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
from ARO_benchmark.evaluation_ARO import VGDataset, COCOOrderDataset
from torch.utils.tensorboard import SummaryWriter



def evaluate(dataloader, model, device):
    """
    Evaluate model on the dataset in the dataloader
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
        if max(probs) == probs[0]:
            acc += 1
    return acc / len(dataloader)


def training(model, optimizer, scheduler, train_loader, val_loader, vgr_loader, vga_loader, coco_order_loader,
             max_epochs, device):
    writer = SummaryWriter()

    for epoch in tqdm(range(max_epochs)):
        # TRAIN
        model.train()
        for i, data in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
            optimizer.zero_grad()

            images, captions = data
            b, s, c, h, w = images.shape #batch_size, image+strong altenative, channels, height, width
            images = images.to(device)
            captions = captions.to(device)
            captions_pos = captions[:, 0:2, :].flatten(0, 1)
            captions_neg = captions[:, 2:, :].flatten(0, 1)
            images = images.flatten(0, 1) # BSCHW -> (B*S)CHW


            ##### -- DEBUGGING FORWARD -- #####

            encoded_image = model.encode_image(images)
            encoded_text = model.encode_text(torch.cat((captions_pos, captions_neg)))

            if torch.isnan(encoded_image).any():
                print("image encoded: ",encoded_image)
                print("\n text encoded: ", encoded_text)
                assert False
            
            if torch.isnan(encoded_text).any():
                print("image encoded: ",encoded_image)
                print("\n text encoded: ", encoded_text)
                assert False

            #this part is completely from openAI
            # normalized features
            encoded_image = encoded_image / encoded_image.norm(dim=1, keepdim=True)
            encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)

            if torch.isnan(encoded_image).any():
                print("after normalization")
                print("image encoded: ",encoded_image)
                print("\n text encoded: ", encoded_text)
                assert False
            
            if torch.isnan(encoded_text).any():
                print("after normalization")
                print("image encoded: ",encoded_image)
                print("\n text encoded: ", encoded_text)
                assert False

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()

            if torch.isnan(logit_scale).any():
                print("logits scale: ", logit_scale)
                assert False
        
            logits_per_image = logit_scale * encoded_image @ encoded_text.t()
            logits_per_image = logits_per_image[:,:2 * b]
            logits_per_text = logits_per_image.t()

            if torch.isnan(logits_per_image).any():
                print("logits_per_image: ", logits_per_image)
                print("logits_scale: ", logit_scale)

                assert False


            ###################################

            #following code to uncomment after debugging

            # encoding + cosine similarity as logits
            # logits_per_image, logits_per_text = model(images, torch.cat((captions_pos, captions_neg)))
            # logits_per_image = logits_per_image[:,:2 * BATCH_SIZE]  # keeping only the true captions to compute loss
            # logits_per_text = logits_per_image.t()

            # loss
            labels = torch.arange(2 * b).to(device)

            loss_text = cross_entropy(logits_per_text, labels)
            loss_image = cross_entropy(logits_per_image, labels)

            if torch.isnan(loss_text).any() or torch.isnan(loss_image):
                print("logits_per_text: ", logits_per_text)
                print("logits_per_image: ", logits_per_image)
                print("logits scale: ", logit_scale)
                print("labels: ", labels)
                assert False

            loss = (loss_text + loss_image) / 2
            loss.backward()
            optimizer.step()
            scheduler.step()

            #Clip the logit scale to increase stability as suggested in CLIP original paper
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))


            # evaluate on ARO
            acc_vgr = evaluate(vgr_loader, model, device)
            acc_vga = evaluate(vga_loader, model, device)
            acc_coco_order = evaluate(coco_order_loader, model, device)

            # logs
            step = epoch * len(train_loader) + i
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("loss/image/train", loss_image.item(), step)
            writer.add_scalar("loss/text/train", loss_text.item(), step)
            writer.add_scalar("evaluate/VGR/train", acc_vgr, step)
            writer.add_scalar("evaluate/VGA/train", acc_vga, step)
            writer.add_scalar("evaluate/COCO_order/train", acc_coco_order, step)

            # save weights
            if i % len(train_loader) / 10 == 0:
                torch.save(model.state_dict(), f"weights/epoch{epoch}_step{step}.pth")

        # VAL
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                images, captions = data
                b, _, c, h, w = images.shape
                images = images.to(device)
                captions = captions.to(device)
                captions_pos = captions[:, 0:2, :].flatten(0, 1)
                captions_neg = captions[:, 2:, :].flatten(0, 1)
                images = images.flatten(0, 1)

                # encoding + cosine similarity as logits
                logits_per_image, logits_per_text = model(images, torch.cat((captions_pos, captions_neg)))
                logits_per_image = logits_per_image[:,:2 * b]  # keeping only true captions to compute loss
                logits_per_text = logits_per_image.t()
                print("logits per image: ",logits_per_image)

                # loss
                labels = torch.arange(2 * b).to(device)
                loss_text = cross_entropy(logits_per_text, labels)
                print("loss text: ", loss_text)
                loss_image = cross_entropy(logits_per_image, labels)
                print("loss image: ", loss_image)
                loss = (loss_text + loss_image) / 2

                # evaluate on ARO
                acc_vgr = evaluate(vgr_loader, model, device)
                acc_vga = evaluate(vga_loader, model, device)
                acc_coco_order = evaluate(coco_order_loader, model, device)

                # logs
                step = epoch * len(val_loader) + i
                writer.add_scalar("loss/val", loss.item(), step)
                writer.add_scalar("loss/image/val", loss_image.item(), step)
                writer.add_scalar("loss/text/val", loss_text.item(), step)
                writer.add_scalar("evaluate/VGR/val", acc_vgr, step)
                writer.add_scalar("evaluate/VGA/val", acc_vga, step)
                writer.add_scalar("evaluate/COCO_order/val", acc_coco_order, step)


if __name__ == "__main__":
    # hyper params
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SET_TYPE = "val"
    BATCH_SIZE = 4
    MAX_EPOCHS = 5
    WARMUP_STEPS = 50
    VALSET_SIZE = 0.15
    SHUFFLE_DTS = False
    LR = 5e-6  # picked one of the three proposed : {1e − 5, 5e − 6, 1e − 6}
    VGA_VGR_PATH = "../ARO_benchmark/VGA_VGR/"
    COCO_ORDER_PATH = "../ARO_benchmark/COCO_Order/captions_negcaptions.json"
    NB_TESTCASES = 50

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
    vgr_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_relations.csv", VGA_VGR_PATH + "images",
                        NB_TESTCASES, image_tranform=preprocess, text_transform=clip.tokenize)
    vgr_loader = DataLoader(vgr_dts)

    vga_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_attributes.csv", VGA_VGR_PATH + "images",
                        NB_TESTCASES, image_tranform=preprocess, text_transform=clip.tokenize)
    vga_loader = DataLoader(vga_dts)

    coco_order_dts = COCOOrderDataset(COCO_ORDER_PATH, f'../COCO/{SET_TYPE}2014/COCO_{SET_TYPE}2014_', SET_TYPE,
                                      NB_TESTCASES, image_tranform=preprocess, text_transform=clip.tokenize)
    coco_order_loader = DataLoader(coco_order_dts)

    # optimizer
    optim = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optim, WARMUP_STEPS)

    training(model, optim, scheduler, train_loader, val_loader,
             vgr_loader, vga_loader, coco_order_loader,
             MAX_EPOCHS, DEVICE)
