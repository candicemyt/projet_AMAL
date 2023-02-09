import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from CLIP.my_clip import Clip as MyClip

def evaluate(dataset, model, device, preprocess):
    acc = 0
    # for i, (image, label) in tqdm(enumerate(dataloader), leave=False, total=len(loader)):
    #     image_input = image.to(device)
    #     label = label.to(device)
    #     text_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    #     # Calculate features
    #     with torch.no_grad():
    #         image_features = model.encode_image(image_input)
    #         text_features = model.encode_text(text_input)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #     pred = similarity.argmax() #pas sur du 0

    for i in tqdm(range(len(dataset)), leave=False, total=len(dataset)):
        image, class_id = dataset[i]
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred = similarity[0].argmax()
        if pred == class_id:
            acc += 1
    return acc / len(dataset)

if __name__ == "__main__":
    #hyperparameters 
    BATCH_SIZE = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_test = [arg for arg in sys.argv]
    
    #load model and preprocessing
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    loader = DataLoader(cifar100, batch_size=BATCH_SIZE, shuffle=True)

    if "negclip" in models_to_test:
        print("Evaluation of negclip on CIFAR")

        # Load the model
        weight_path = "../NegCLIP/weights/negclip-train-newloss-epoch10-lr5e-06_epoch9.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device))

        acc_cifar = evaluate(loader, model, device)
        print("accuracy cifar: ", acc_cifar)

        #save results
        res_negclip_path = "./NegCLIP_perf_CIFAR.txt"
        if os.path.exists(res_negclip_path):
            res_myclip_file = open(res_negclip_path, 'w')
        else:
            res_myclip_file = open(res_negclip_path, 'x')

        res_myclip_file.write("CIFAR100 : " + str(acc_cifar) + "\n")


    elif "myclip" in models_to_test:
        print("Evaluation of my_clip on CIFAR")

        # Load the model

        model = MyClip(embedding_size=512, vision_embedding=768, seq_length=77,
                             num_heads=8, vocab_size=49408, n_blocks=12,
                             output_dim=512, kernel_size=32, stride=32, input_resolution=224)

        weight_path = "../CLIP/my_clip_weights/clip.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)

        acc_cifar = evaluate(loader, model, device)
        print("accuracy myclip cifar: ", acc_cifar)

        #save results
        res_negclip_path = "./MyCLIP_perf_CIFAR.txt"
        if os.path.exists(res_negclip_path):
            res_myclip_file = open(res_negclip_path, 'w')
        else:
            res_myclip_file = open(res_negclip_path, 'x')

        res_myclip_file.write("CIFAR100 : " + str(acc_cifar) + "\n")
    
    elif "clip" in models_to_test:
        print("Evaluation of clip on CIFAR")


        acc_cifar = evaluate(cifar100, model, device, preprocess)
        print("accuracy cifar: ", acc_cifar)

        #save results
        res_negclip_path = "./NegCLIP_perf_CIFAR.txt"
        if os.path.exists(res_negclip_path):
            res_myclip_file = open(res_negclip_path, 'w')
        else:
            res_myclip_file = open(res_negclip_path, 'x')

        res_myclip_file.write("CIFAR100 : " + str(acc_cifar) + "\n")

    