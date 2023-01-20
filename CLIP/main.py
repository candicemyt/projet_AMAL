import clip
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from importlib import reload
reload(clip)
from clip import Clip


if __name__ == "__main__":
    

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    print(model)
    my_clip = Clip(embedding_size=512, #certain
                   vision_embedding=768, #certain
                   seq_length=77, #presque sur
                   num_heads=7, #pas sur
                   vocab_size=49408, #presque sur
                   n_blocks=12, #certain
                   output_dim=512, #certain
                   kernel_size=32, #certain
                   stride=32, #certain
                   input_resolution=224) #par calcul ça donne ça

    print(my_clip)
    #openai multihead attention != Transformer from scratch -> look at the doc of F.multihead_attention_forward