import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("2.jpg")).unsqueeze(0).to(device)
image2 = preprocess(Image.open("6.jpg")).unsqueeze(0).to(device)
#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
images = [image, image2]
with torch.no_grad():
    for i in range(0, len(images), 2) :
        image_features = model.encode_image(images[i])
        image2_features = model.encode_image(images[i+1])

        similarity = image2_features.cpu().numpy() @ image_features.cpu().numpy().T

print(similarity)