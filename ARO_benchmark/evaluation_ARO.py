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
        self.raw_data = self.read_VG_file(self.file_path, self.n)

    def read_VG_file(self, file_path, n):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.split(',')
                data.append(({"image_id": line[0], "caption": line[1], "label": int(line[2])}))

        data = np.array(data)
        if n is not None:
            indices = np.random.randint(0, data.shape[0], size=n)
            indices = 2*(indices//2) + 1
        else:
            indices = np.arange(0,  data.shape[0], step=2)

        return data[indices]

    def _load_image(self, image_id):
        image = Image.open(self.image_root+'/'+image_id+'.jpg')
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image

    def _load_text(self, t1, t2):
        if self.text_transform is not None:
            text = torch.cat((self.text_transform(t1), self.text_transform(t2)))
        else:
            text = torch.cat((t1, t2))
        return text

    def __len__(self):
        if self.n is not None:
            return self.n
        else:
            return self.raw_data.shape[0]

    def __getitem__(self, index):
        item = self.raw_data[index]
        item_neg = self.raw_data[index+1]
        image = self._load_image(item['image_id'])
        text = self._load_text(item["caption"], item_neg["caption"])
        return image, text

