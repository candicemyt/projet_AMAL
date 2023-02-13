import torch
from torchvision.datasets import CocoDetection


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
    def __init__(self, root, pairwise_sim_path, annFile, transforms=None, transform=None, target_transform=None):
        super().__init__(root, annFile, transform, target_transform, transforms)

        # loading the 3 nearest neighbors of each image
        self.pairwise_sim_path = pairwise_sim_path
        nearest_neighbors = dict()
        with open(pairwise_sim_path, 'r') as f:
            for line in f:
                ids = line.split(',')
                nearest_neighbors[int(ids[0])] = [int(id) for id in ids[1::]]

        # dictionnary with key : image index, value : list of id of 3 nearest neighbors of each image
        self.nearest_neighbors = nearest_neighbors

    def _load_target(self, id: int):
        anns = super()._load_target(id)
        captions = []
        neg_captions = []
        for ann in anns:
            captions.append(ann["caption"])
            neg_captions.append(ann["neg_captions"])
        caption_id = torch.randint(0, len(captions), (1,))
        return captions[caption_id], neg_captions[caption_id]

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        pos_caption, neg_captions = self._load_target(id)

        # choose one strong alternative image (and its respective captions) between the k=3 nearest neighbors
        rd_image_neighbor = torch.randint(0, 3, (1,))
        strong_alt_id = self.nearest_neighbors[id][rd_image_neighbor]  # id in root directory
        strong_alt_image = self._load_image(strong_alt_id)
        strong_alt_caption, neg_strong_alt_captions = self._load_target(strong_alt_id)

        # choose only one negative caption for each positive caption
        neg_caption = neg_captions[torch.randint(0, len(neg_captions), (1,))]
        neg_strong_alt_caption = neg_strong_alt_captions[torch.randint(0, len(neg_strong_alt_captions), (1,))]

        # transforming the data
        image = self.transform(image).unsqueeze(0)
        strong_alt_image = self.transform(strong_alt_image).unsqueeze(0)
        pos_caption = self.target_transform(pos_caption)
        neg_caption = self.target_transform(neg_caption)
        strong_alt_caption = self.target_transform(strong_alt_caption)
        neg_strong_alt_caption = self.target_transform(neg_strong_alt_caption)

        captions = torch.cat((pos_caption, strong_alt_caption, neg_caption, neg_strong_alt_caption))
        images = torch.cat((image, strong_alt_image))

        return images, captions
