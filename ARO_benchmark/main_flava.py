from evaluation_ARO import VGDataset, COCOOrderDataset, evaluate
import torch
from torch.utils.data import DataLoader
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from transformers import FlavaForPreTraining, FlavaImageProcessor, BertTokenizerFast 

def evaluate_flava(dataloader, model, device):
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
        image = image.to(device)
        captions = captions.to(device)

        caption = {
            "input_ids" : captions.input_ids[0].unsqueeze(0),
            "token_type_ids" : captions.token_type_ids[0].unsqueeze(0),
            "attention_mask" : captions.attention_mask[0].unsqueeze(0)
        }
        outputs = model(**image, **caption)
        value0 = outputs.contrastive_logits_per_image.item()
        probmax = outputs.contrastive_logits_per_image.item()
        for i in range(1, captions.input_ids.size()[0]) :
            caption = {
            "input_ids" : captions.input_ids[i].unsqueeze(0),
            "token_type_ids" : captions.token_type_ids[i].unsqueeze(0),
            "attention_mask" : captions.attention_mask[i].unsqueeze(0)
            }
            outputs = model(**image, **caption)
            probmax = max(probmax, outputs.contrastive_logits_per_image.item())
        
        if probmax == value0:
            acc += 1
    return acc / len(dataloader)

if __name__ == "__main__":
    VGA_VGR_PATH = "VGA_VGR/"
    COCO_ORDER_PATH = "COCO_Order/captions_shuffled_captions.json"
    SET_TYPE = "train"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    flava_model = FlavaForPreTraining.from_pretrained("facebook/flava-full", cache_dir = "FLAVA/")
    processor = FlavaImageProcessor.from_pretrained("facebook/flava-full", cache_dir = "FLAVA/")
    tokenizer = BertTokenizerFast.from_pretrained("facebook/flava-full", cache_dir = "FLAVA/")
    
    flava_model = flava_model.float()
    flava_model.eval()

    imageprocessing = lambda image : processor.preprocess(image, return_tensors='pt', return_image_mask=True, return_codebook_pixels=True)
    tokenizing = lambda text : tokenizer(text, return_tensors='pt',padding="max_length", max_length=77)
    
    vgr_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_relations.csv", VGA_VGR_PATH + "images",
                        image_tranform=imageprocessing, text_transform=tokenizing)
    vgr_loader = DataLoader(vgr_dts)

    vga_dts = VGDataset(VGA_VGR_PATH + f"/{SET_TYPE}/dataset_attributes.csv", VGA_VGR_PATH + "images",
                            image_tranform=imageprocessing, text_transform=tokenizing)
    vga_loader = DataLoader(vga_dts)

    coco_order_dts = COCOOrderDataset(COCO_ORDER_PATH, f'../COCO/val2014/COCO_val2014_', SET_TYPE,
                            image_tranform=imageprocessing, text_transform=tokenizing)
    coco_order_loader = DataLoader(coco_order_dts)

    print("evaluation starts")
    acc_vgr = evaluate_flava(vgr_loader, flava_model, DEVICE)
    print(acc_vgr)
    acc_vga = evaluate_flava(vga_loader, flava_model, DEVICE)
    print(acc_vga)
    acc_coco_ord = evaluate_flava(coco_order_loader, flava_model, DEVICE)
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
