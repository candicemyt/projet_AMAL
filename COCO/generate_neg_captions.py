import json
import spacy
from os import path
from tqdm import tqdm
from itertools import combinations

def shuffle(caption, first_token, second_token):
    """
    Shuffle the caption : replacing first_token by second_token and vice versa
    """
    sentence_beg = caption.index(first_token)
    sentence_mid_beg = caption.index(first_token) + len(first_token)
    sentence_mid_end = caption.index(second_token)
    sentence_end = caption.index(second_token) + len(second_token)
    #print("beg", caption[0:sentence_beg])
    #print("second", second_token)
    #print("mid", sentence_mid_beg, sentence_mid_end, caption[sentence_mid_beg:sentence_mid_end])
    #print("first", first_token)
    #print("end", caption[sentence_end::])
    shuffled_caption = caption[0:sentence_beg] + second_token + \
                       caption[sentence_mid_beg:sentence_mid_end] + \
                       first_token + caption[sentence_end::]
    return shuffled_caption


def shuffle_caption(caption):
    """
    Shuffle 5 times (maximum) each caption of captions
    :param caption: string (caption)
    :return: the positive captions that are keeped and a list of negative captions for each one
    """
    nlp = spacy.load("en_core_web_sm")

    shuffled_captions = []
    doc = nlp(caption)  # pos tagging with spacy

    for word_type in ["NOUN", "ADJ", "ADV", "VERB"]:
        word_list = [token.text for token in doc if token.pos_ == word_type]
        # shuffle only if we have at least two words of the same type and not identical
        for word1, word2 in combinations(word_list, 2):
            if word1 not in word2 and word2 not in word1:
                shuffled_caption = shuffle(caption, word1, word2)
                #print(word_type)
                shuffled_captions.append(shuffled_caption)
                break

    noun_phrases = [noun_phrase for noun_phrase in doc.noun_chunks if
                    len(noun_phrase) > 3]  # noun phrases with spacy

    if len(noun_phrases) >= 2:  # shuffle possible only if we have at least two noun phrases
        shuffled_caption = shuffle(caption, str(noun_phrases[0]), str(noun_phrases[-1]))
        shuffled_captions.append(shuffled_caption)

    return caption, shuffled_captions


def generate_neg_captions(set_type):

    #open file
    with open(f"annotations/captions_{set_type}2014.json", 'r') as f:
        data = json.load(f)
    raw_captions_list = data["annotations"]

    neg_captions_list = []
    pos_captions_list = []

    # shuffling the captions
    for caption_data in tqdm(raw_captions_list):
        pos_caption, neg_captions = shuffle_caption(caption_data["caption"])
        if len(neg_captions) > 0:
            neg_captions_list.append({'image_id' : caption_data['image_id'], 'id' : caption_data['id'], 'neg_captions' : neg_captions})
            pos_captions_list.append(caption_data) # we keep only the captions that have negative captions

    # write files
    path_neg_captions_file = f"annotations/neg_captions_{set_type}.json"
    if path.exists(path_neg_captions_file):
        neg_captions_file = open(path_neg_captions_file, 'w')
    else:
        neg_captions_file = open(path_neg_captions_file, 'x')
    json.dump({"annotations" : neg_captions_list}, neg_captions_file)

    data["annotations"] = pos_captions_list
    with open(f"annotations/captions_modif_{set_type}2014.json", 'w') as f:
        json.dump(data, f)


generate_neg_captions("val")