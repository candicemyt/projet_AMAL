import sys
import os
from torch.utils.data import DataLoader
import clip
import torch
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from ARO_benchmark.evaluation_ARO import VGDataset, COCOOrderDataset, evaluate
from NegCLIP.COCO_Dataset import COCODataset


def training(model, optimizer, scheduler, coco_loader, vgr_loader, vga_loader, coco_order_loader, max_epochs, device):
    run = f"negclip-textft-epoch{max_epochs}-lr{LR}"
    writer = SummaryWriter(log_dir="runs/" + run)

    for epoch in tqdm(range(max_epochs)):
        # TRAIN
        model.train()

        for i, data in tqdm(enumerate(coco_loader), leave=False, total=len(coco_loader)):
            optimizer.zero_grad()

            images, captions = data
            b, s, c, h, w = images.shape  # batch_size, image+strong altenative, channels, height, width
            images = images.to(device)
            captions = captions.to(device)
            captions_pos = captions[:, 0:2, :].flatten(0, 1)
            captions_neg = captions[:, 2:, :].flatten(0, 1)
            images = images.flatten(0, 1)  # BSCHW -> (B*S)CHW

            # forward pass + normalization
            encoded_text = model.encode_text(torch.cat((captions_pos, captions_neg)))
            encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)
            encoded_image = model.encode_image(images)
            encoded_image = encoded_image / encoded_image.norm(dim=1, keepdim=True)

            sim_text_text = model.logit_scale * encoded_text @ encoded_text.t()
            logits = model.logit_scale * encoded_image @ encoded_text.t()

            num_logits = logits.shape[0]
            # loss
            labels_text = torch.arange(2 * num_logits).to(device)
            labels_image = torch.arange(num_logits).to(device)
            loss_text_text = cross_entropy(sim_text_text, labels_text)
            loss_image_text = cross_entropy(logits, labels_image)
            loss = (loss_image_text + loss_text_text) / 2

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Clip the logit scale to increase stability as suggested in CLIP original paper
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))

            # logs
            step = epoch * len(coco_loader) + i

            # compute similarity between pos and neg caption
            sim = torch.tensor([logits[i, (i + b) % (2 * b)] for i in range(2 * b)]).mean()

            # add logs
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("similarity_pos_neg/train", sim.item(), step)

        # save neclip-ftxt-lr5e-06-epoch9.pth
        torch.save(model.state_dict(), f"weights/neclip-ftxt-lr5e-06-epoch9.pth/{run}_epoch{epoch}.pth")

        # VAL
        model.eval()
        with torch.no_grad():

            # evaluate on ARO
            acc_vgr = evaluate(vgr_loader, model, device)
            acc_vga = evaluate(vga_loader, model, device)
            acc_coco_order = evaluate(coco_order_loader, model, device)

            # logs
            writer.add_scalar("evaluate/VGR", acc_vgr, epoch)
            writer.add_scalar("evaluate/VGA", acc_vga, epoch)
            writer.add_scalar("evaluate/COCO_order", acc_coco_order, epoch)


if __name__ == "__main__":
    # hyper params
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SET_TYPE = "train"
    BATCH_SIZE = 20
    MAX_EPOCHS = 10
    WARMUP_STEPS = 50
    SHUFFLE_DTS = False
    LR = 5e-6  # picked one of the three proposed : {1e − 5, 5e − 6, 1e − 6}
    VGA_VGR_PATH = "../ARO_benchmark/VGA_VGR/"
    COCO_ORDER_PATH = "../ARO_benchmark/COCO_Order/captions_shuffled_captions.json"
    NB_TESTCASES = 1000

    # load pretrain model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model = model.float()

    # setup dataset an dataloader
    coco_dts = COCODataset(root=f'../COCO/{SET_TYPE}2014/', pairwise_sim_path=f"../COCO/pairwise_sim/{SET_TYPE}.csv",
                           annFile=f'../COCO/annotations/captions_negcaptions_{SET_TYPE}2014.json',
                           transform=preprocess, target_transform=clip.tokenize)
    dts_size = len(coco_dts)
    coco_loader = DataLoader(coco_dts, batch_size=BATCH_SIZE)

    # setup dataloader for evaluation
    vgr_dts = VGDataset(VGA_VGR_PATH + f"/val/dataset_relations.csv", VGA_VGR_PATH + "images",
                        NB_TESTCASES, image_tranform=preprocess, text_transform=clip.tokenize)
    vgr_loader = DataLoader(vgr_dts)

    vga_dts = VGDataset(VGA_VGR_PATH + f"/val/dataset_attributes.csv", VGA_VGR_PATH + "images",
                        NB_TESTCASES, image_tranform=preprocess, text_transform=clip.tokenize)
    vga_loader = DataLoader(vga_dts)

    coco_order_dts = COCOOrderDataset(COCO_ORDER_PATH, f'../COCO/val2014/COCO_val2014_', "val",
                                      NB_TESTCASES, image_tranform=preprocess, text_transform=clip.tokenize)
    coco_order_loader = DataLoader(coco_order_dts)

    # optimizer
    optim = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optim, WARMUP_STEPS)

    training(model, optim, scheduler, coco_loader,
             vgr_loader, vga_loader, coco_order_loader,
             MAX_EPOCHS, DEVICE)