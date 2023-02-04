from evaluation_ARO import VGDataset, COCOOrderDataset, evaluate
import clip
import torch
from torch.utils.data import DataLoader
from os import path
import sys

if __name__ == "__main__":
    #params
    VGA_VGR_PATH = "VGA_VGR/"
    COCO_ORDER_PATH = "COCO_Order/captions_shuffled_captions.json"
    SET_TYPE = "train"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHTS_PATHS = ["../NegCLIP/weights/negclip-epoch10-lr1e-05_epoch9.pth",
                     "../NegCLIP/weights/negclip-epoch10-lr1e-06_epoch9.pth",
                     "../NegCLIP/weights/negclip-epoch10-lr5e-06_epoch9.pth"]

    models_to_test = [arg for arg in sys.argv]

    if "clip" in models_to_test or "negclip" in models_to_test:
        # load pretrain model
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        clip_model = clip_model.float()
        clip_model.eval()

        # setup dataloader for evaluation
        vgr_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_relations.csv", VGA_VGR_PATH + "images",
                            image_tranform=clip_preprocess, text_transform=clip.tokenize)
        vgr_loader = DataLoader(vgr_dts)

        vga_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_attributes.csv", VGA_VGR_PATH + "images",
                            image_tranform=clip_preprocess, text_transform=clip.tokenize)
        vga_loader = DataLoader(vga_dts)

        coco_order_dts = COCOOrderDataset(COCO_ORDER_PATH, f'../COCO/val2014/COCO_val2014_', SET_TYPE,
                                          image_tranform=clip_preprocess, text_transform=clip.tokenize)
        coco_order_loader = DataLoader(coco_order_dts)

        # evaluation
        if "clip" in models_to_test:
            print("Evaluation of clip on ARO")

            acc_vgr = evaluate(vgr_loader, clip_model, DEVICE)
            print(acc_vgr)
            acc_vga = evaluate(vga_loader, clip_model, DEVICE)
            print(acc_vga)
            acc_coco_ord = evaluate(coco_order_loader, clip_model, DEVICE)
            print(acc_coco_ord)

            # save results
            res_clip_path = "clip_perf_aro.txt"
            if path.exists(res_clip_path):
                res_clip_file = open(res_clip_path , 'w')
            else:
                res_clip_file = open(res_clip_path, 'x')

            res_clip_file.write("vgr : " + str(acc_vgr) + "\n")
            res_clip_file.write("vga : " + str(acc_vga) + "\n")
            res_clip_file.write("coco-order : " + str(acc_coco_ord))

        if "negclip" in models_to_test:
            # save results in file
            res_negclip_path = "negclip_perf_aro.txt"
            if path.exists(res_negclip_path):
                res_negclip_file = open(res_negclip_path, 'a')
            else:
                res_negclip_file = open(res_negclip_path, 'x')

            for weight_path in WEIGHTS_PATHS:
                clip_model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
                negclip_model = clip_model
                negclip_model.eval()

                print("Evaluation of NegClip on ARO")
                acc_vgr = evaluate(vgr_loader, negclip_model, DEVICE)
                print(acc_vgr)
                acc_vga = evaluate(vga_loader, negclip_model, DEVICE)
                print(acc_vga)
                acc_coco_ord = evaluate(coco_order_loader, negclip_model, DEVICE)
                print(acc_coco_ord)

                res_negclip_file.write(weight_path.split('/')[3] + "\n")
                res_negclip_file.write("vgr : " + str(acc_vgr) + "\n")
                res_negclip_file.write("vga : " + str(acc_vga) + "\n")
                res_negclip_file.write("coco-order : " + str(acc_coco_ord))
