from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import torch
from torchvision.datasets import CocoDetection

# TODO : ignore strong alt for CLIP-FT




def choose_neg_caption_per_caption(all_neg_captions):
    """
    all_neg_captions is a list of list of caption.
    It chooses randomly one caption per list
    """
    neg_captions = []
    for list_neg_captions in all_neg_captions:
        rd_neg_caption = torch.randint(0, len(list_neg_captions), (1,))
        neg_captions.append(list_neg_captions[rd_neg_caption])
    return neg_captions




class COCODataset(CocoDetection):
    def __init__(self, root, pairwise_sim_path, negcaptions_path, annFile, transforms=None, transform=None, target_transform=None):
        super().__init__(root, annFile, transform, target_transform, transforms)

        # loading the 3 nearest neighbors of each image
        self.pairwise_sim_path = pairwise_sim_path
        nearest_neighbors = dict()
        with open(pairwise_sim_path, 'r') as f:
            for line in f:
                ids = line.split(',')
                nearest_neighbors[int(ids[0])] = [int(id) for id in ids[1::]]

        # dictionanry with key : image index, value : list of id of 3 nearest neighbors of each image
        self.nearest_neighbors = nearest_neighbors

        # loading the shuffled captions
        self.negcaptions_path = negcaptions_path
        with open(negcaptions_path, 'r') as f:
            negcaptions = json.load(f)["annotations"]
        self.negcaptions = negcaptions

    def _load_target(self, id: int):
        ids = self.coco.getAnnIds(id)

        return self.coco.loadAnns(ids)
    def __getitem__(self, index: int):
        image, pos_captions = super().__getitem__(index)

        # choose one strong alternative image (and its respective captions) between the k=3 nearest neighbors
        rd_image_neighbor = torch.randint(0, 3, (1,))
        strong_alt_index = self.nearest_neighbors[index][rd_image_neighbor]  # index in dataloader
        strong_alt_id = self.ids[strong_alt_index]  # id in root directory
        strong_alt_image = self._load_image(strong_alt_id)
        strong_alt_image = self.transform(strong_alt_image)
        strong_alt_captions = self._load_target(strong_alt_id)

        # compute the negative captions and choose only one for each positive caption
        id = self.ids[index] #id in root directory
        all_neg_captions = [data_neg_capt["neg_captions"] for data_neg_capt in self.negcaptions]
        #pos_captions, all_neg_captions = shuffle_captions(pos_captions)
        neg_captions = []  #choose_neg_caption_per_caption(all_neg_captions)
        #strong_alt_captions, all_neg_strong_alt_captions = shuffle_captions(strong_alt_captions)
        neg_strong_alt_captions = [] #choose_neg_caption_per_caption(all_neg_strong_alt_captions)

        return image, pos_captions, neg_captions, strong_alt_image, strong_alt_captions, neg_strong_alt_captions
