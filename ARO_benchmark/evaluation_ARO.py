import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json

class VGDataset(Dataset):
    def __init__(self, file_path, image_root, n=None, image_tranform=None, text_transform=None):
        """
        :param file_path: file with captions
        :param image_root: path to images
        :param n: number of test cases wanted
        :param image_tranform: encoding image
        :param text_transform: encoding text
        """
        self.file_path = file_path
        self.image_root = image_root
        self.n = n
        self.image_transform = image_tranform
        self.text_transform = text_transform
        self.raw_data, self.indexes = self.read_VG_file()

    def read_VG_file(self):
        """
        Select the indexes + read the file with data
        """
        data = []
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.split(',')
                data.append(({"image_id": line[0], "caption": line[1], "label": int(line[2])}))

        data = np.array(data)
        if self.n is not None:
            indexes = np.random.randint(0, data.shape[0], size=self.n)
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
        """
        Get an image with its true and false caption
        :param index: int
        :return: image, list of two captions
        """
        id = self.indexes[index-1]
        item = self.raw_data[id]  # true caption
        item_neg = self.raw_data[id + 1]  # false caption
        image = self._load_image(item['image_id'])

        captions = [item["caption"], item_neg["caption"]]
        if self.text_transform is not None:
            captions = self.text_transform(captions)

        return image, captions


class COCOOrderDataset(Dataset):
    def __init__(self, file_path, image_root, set_type, n=None, image_tranform=None, text_transform=None):
        self.file_path = file_path
        self.image_root = image_root
        self.set_type = set_type
        self.n = n
        self.image_transform = image_tranform
        self.text_transform = text_transform
        self.raw_data, self.indexes = self.read_COCOOrder_file()

    def read_COCOOrder_file(self):

        with open(self.file_path, "r") as f:
            data = json.load(f)["annotations"]
        data = np.array(data)
        data_size = data.shape[0]

        if self.set_type == 'val':
            min_ind = 0
            max_ind = int(data_size * 0.15)
        else:
            min_ind = int(data_size * 0.15)
            max_ind = data_size

        if self.n is not None:
            indexes = np.random.randint(min_ind, max_ind, size=self.n)
        else:
            indexes = np.arange(min_ind, max_ind)

        return data, indexes


    def _load_image(self, image_id):
        image = Image.open(self.image_root + str(image_id).zfill(12) + '.jpg')
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        id = self.indexes[index-1]
        item = self.raw_data[id]

        captions = [item["caption"]] + item["neg_captions"]
        if self.text_transform is not None:
            captions = self.text_transform(captions)

        image = self._load_image(item["image_id"])
        return image, captions


def evaluate(dataloader, model, device):
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
        captions = captions.flatten(0, 1).to(device)
        image = image.to(device)
        logits_per_image, logits_per_text = model(image, captions)
        probs = logits_per_image.softmax(dim=-1).squeeze()
        if max(probs) == probs[0]:
            acc += 1
    return acc / len(dataloader)
