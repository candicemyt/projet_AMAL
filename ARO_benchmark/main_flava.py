from evaluation_ARO import VGDataset, COCOOrderDataset, evaluate
import torch
from torch.utils.data import DataLoader
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from transformers import FlavaModel, FlavaImageProcessor, BertTokenizerFast 

if __name__ == "__main__":
    VGA_VGR_PATH = "VGA_VGR/"
    COCO_ORDER_PATH = "COCO_Order/captions_shuffled_captions.json"
    SET_TYPE = "train"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    flava_model = FlavaModel.from_pretrained("facebook/flava-full")
    processor = FlavaImageProcessor.from_pretrained("facebook/flava-full")
    tokenizer = BertTokenizerFast.from_pretrained("facebook/flava-full")

    vgr_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_relations.csv", VGA_VGR_PATH + "images",
                            image_tranform=processor, text_transform=tokenizer)
    vgr_loader = DataLoader(vgr_dts)

    vga_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_attributes.csv", VGA_VGR_PATH + "images",
                            image_tranform=processor, text_transform=tokenizer)
    vga_loader = DataLoader(vga_dts)

    coco_order_dts = COCOOrderDataset(COCO_ORDER_PATH, f'../COCO/val2014/COCO_val2014_', SET_TYPE,
                            image_tranform=processor, text_transform=tokenizer)
    coco_order_loader = DataLoader(coco_order_dts)

            
    acc_vgr = evaluate(vgr_loader, flava_model, DEVICE)
    print(acc_vgr)
    acc_vga = evaluate(vgr_loader, flava_model, DEVICE)
    print(acc_vga)
    acc_coco_ord = evaluate(coco_order_loader, flava_model, DEVICE)
    print(acc_coco_ord)

    # save results
    res_flava_path = "flava_perf_aro.txt"
    if os.path.exists(res_flava_path):
        res_flava_file = open(res_flava_path, 'w')
    else:
        res_flava_file = open(res_flava_path, 'x')

    res_flava_file.write("vgr : " + str(acc_vgr) + "\n")
    res_flava_file.write("vga : " + str(acc_vga) + "\n")
    res_flava_file.write("coco-order : " + str(acc_coco_ord))
