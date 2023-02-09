import my_clip
import torch
from PIL import Image
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import requests
from transformers import CLIPProcessor, CLIPModel
from importlib import reload
reload(my_clip)
from my_clip import Clip
import clip


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load("ViT-B/32", device="cpu")
    model = Clip(embedding_size=512, #certain
                vision_embedding=768, #certain
                seq_length=77, #presque sur
                num_heads=8, #sur
                vocab_size=49408, #presque sur
                n_blocks=12, #certain
                output_dim=512, #certain
                kernel_size=32, #certain
                stride=32, #certain
                input_resolution=224,
                device=device).to(device)

    model.load_state_dict(torch.load("./weights/my_clip.pth"))

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = ["an image of a cat", "an image of a duck", "an image of a dog"]

    text_inputs = clip.tokenize(texts).to(device)
    image_input = preprocess(image).unsqueeze(0).to(device)

    print(model(image_input, text_inputs))


