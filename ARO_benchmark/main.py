from evaluation_ARO import VGDataset, COCOOrderDataset, evaluate
import clip
import torch
from torch.utils.data import DataLoader
from os import path

if __name__ == "__main__":
    #params
    VGA_VGR_PATH = "VGA_VGR/"
    COCO_ORDER_PATH = "COCO_Order/captions_negcaptions.json"
    SET_TYPE = "train"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # load pretrain model
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model = clip_model.float()

    # setup dataloader for evaluation
    vgr_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_relations.csv", VGA_VGR_PATH + "images",
                        image_tranform=preprocess, text_transform=clip.tokenize)
    vgr_loader = DataLoader(vgr_dts)

    vga_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_attributes.csv", VGA_VGR_PATH + "images",
                        image_tranform=preprocess, text_transform=clip.tokenize)
    vga_loader = DataLoader(vga_dts)

    coco_order_dts = COCOOrderDataset(COCO_ORDER_PATH, f'../COCO/val2014/COCO_val2014_', SET_TYPE,
                                      image_tranform=preprocess, text_transform=clip.tokenize)
    coco_order_loader = DataLoader(coco_order_dts)


    # evaluation
    acc_vgr = evaluate(vgr_loader, clip_model, DEVICE)
    print(acc_vgr)
    acc_vga = evaluate(vga_loader, clip_model, DEVICE)
    print(acc_vga)
    acc_coco_ord = evaluate(coco_order_loader, clip_model, DEVICE)
    print(acc_coco_ord)

    # save results
    print(acc_vgr)
    print(acc_vga)
    print(acc_coco_ord)

    res_clip_path = "clip_perf_aro.txt"
    if path.exists(res_clip_path):
        res_clip_file = open(res_clip_path , 'w')
    else:
        res_clip_file = open(res_clip_path, 'x')

    res_clip_file.write("vgr : " + str(acc_vgr))
    res_clip_file.write("vga : " + str(acc_vga))
    res_clip_file.write("coco-order : " + str(acc_coco_ord))
