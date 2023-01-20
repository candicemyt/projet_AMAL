from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from torchvision.datasets import CocoCaptions
import spacy

# TODO : if none of the caption of an image has negative captions -> delete from the batch
# TODO : preprocess les negative captions
# TODO : ignore strong alt for CLIP-FT

def shuffle_one_caption(caption, first_token, second_token):
    """
    Shuffle the caption : replacing first_token by second_token and vice versa
    """
    sentence_beg = caption.index(first_token)
    sentence_mid_beg = caption.index(first_token) + len(first_token)
    sentence_mid_end = caption.index(second_token)
    sentence_end = caption.index(second_token) + len(second_token)
    shuffled_caption = caption[0:sentence_beg] + second_token + \
                       caption[sentence_mid_beg:sentence_mid_end] + \
                       first_token + caption[sentence_end::]
    return shuffled_caption


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


def shuffle_captions(captions):
    """
    Shuffle 5 times (maximum) each caption of captions
    :param captions: list of string (caption)
    :return: the positive captions that are keeped and a list of negative captions for each one
    """
    nlp = spacy.load("en_core_web_sm")
    all_pos_captions = []
    all_neg_captions = []
    for caption in captions:
        shuffled_captions = []
        doc = nlp(caption)  # pos tagging with spacy

        for word_type in ["NOUN", "ADJ", "ADV", "VERB"]:
            word_list = [token.text for token in doc if token.pos_ == word_type]
            if len(word_list) >= 2:  # shuffle possible only if we have at least two words of the same type
                shuffled_caption = shuffle_one_caption(caption, word_list[0], word_list[-1])
                shuffled_captions.append(shuffled_caption)

        noun_phrases = [noun_phrase for noun_phrase in doc.noun_chunks if
                        len(noun_phrase) > 3]  # noun phrases with spacy # TODO : len> 3 à modif

        if len(noun_phrases) >= 2:  # shuffle possible only if we have at least two noun phrases
            shuffled_caption = shuffle_one_caption(caption, str(noun_phrases[0]), str(noun_phrases[-1]))
            shuffled_captions.append(shuffled_caption)

        if len(shuffled_captions) > 0:  # if the cpation generates negative captions they are stored
            all_pos_captions.append(caption)
            all_neg_captions.append(shuffled_captions)

    return all_pos_captions, all_neg_captions


class COCODataset(CocoCaptions):
    def __init__(self, root, pairwise_sim_path, annFile, transforms=None, transform=None, target_transform=None):
        super().__init__(root, annFile, transform, target_transform, transforms)

        # path of the csv file with the 3 nearest neighbors of each image
        self.pairwise_sim_path = pairwise_sim_path
        nearest_neighbors = dict()
        with open(pairwise_sim_path, 'r') as f:
            for line in f:
                ids = line.split(',')
                nearest_neighbors[int(ids[0])] = [int(id) for id in ids[1::]]

        # dictionanry with key : image index, value : list of id of 3 nearest neighbors of each image
        self.nearest_neighbors = nearest_neighbors

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
        #pos_captions, all_neg_captions = shuffle_captions(pos_captions)
        neg_captions = []  #choose_neg_caption_per_caption(all_neg_captions)
        #strong_alt_captions, all_neg_strong_alt_captions = shuffle_captions(strong_alt_captions)
        neg_strong_alt_captions = [] #choose_neg_caption_per_caption(all_neg_strong_alt_captions)

        return image, pos_captions, neg_captions, strong_alt_image, strong_alt_captions, neg_strong_alt_captions
