import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch


class VGDataset(Dataset):
    def __init__(self, file_path, image_root, n=None, image_tranform=None, text_transform=None):
        self.file_path = file_path
        self.image_root = image_root
        self.n = n
        self.image_transform = image_tranform
        self.text_transform = text_transform
        self.raw_data, self.indexes = self.read_VG_file(self.file_path, self.n)

    def read_VG_file(self, file_path, n):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.split(',')
                data.append(({"image_id": line[0], "caption": line[1], "label": int(line[2])}))

        data = np.array(data)
        if n is not None:
            indexes = np.random.randint(0, data.shape[0], size=n)
            indexes = 2 * (indexes // 2) + 1  # odd indexes = true caption
        else:
            indexes = np.arange(0, data.shape[0], step=2)  # odd indexes = true caption
        return data, indexes

    def _load_image(self, image_id):
        image = Image.open(self.image_root + '/' + image_id + '.jpg')
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, index):
        id = self.indexes[index-1]
        item = self.raw_data[id]
        item_neg = self.raw_data[id + 1]
        image = self._load_image(item['image_id'])

        captions = [item["caption"], item_neg["caption"]]
        if self.text_transform is not None:
            captions = self.text_transform(captions)

        return image, captions
