
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import json
import spacy

import random
from os import path
from tqdm import tqdm
from itertools import combinations, permutations, product

def shuffle(caption, first_token, second_token):
    """
    Shuffle the caption : replacing first_token by second_token and vice versa
    """

    shuffled = second_token.join(part.replace(second_token, first_token) for part in caption.split(first_token))
    return shuffled



def shuffle_caption(caption):
    """
    Shuffle 5 times (maximum) each caption of captions
    :param caption: string (caption)
    :return: the positive captions that are kept and a list of negative captions for each one
    """
    nlp = spacy.load("en_core_web_sm")

    shuffled_captions = []
    caption = "remarkable scene with a blue ball behind a green chair"
    doc = nlp(caption)  # pos tagging with spacy
    print(doc)

    #suffle noun and adjectives
    #TODO check if selected noun and selected adj are not together
    #IDEA Shuffle adjectives with adj and nouns with nouns as long as they're not linked in the first time
    #if n adj = 2 = n noun -> shuffle just adj
    #if n adj = 1 and n_noun > 1 shuffle nouns
    #if n noun = 1 and n adj > 1 shuffle adj
    #if n noun > 2 and n adj > 2 shuffle tout -> trouver paires exclusives valides
    
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
  
    adj_noun = [(doc[i-1].text,doc[i].text) for i in range(1, len(doc)) if doc[i].pos_ =="NOUN" and doc[i-1].pos_ == "ADJ"]
   
    
    if len(adjectives) > 2 and len(nouns) > 2:
        adj1 = random.choice(adjectives)
        adjectives.remove(adj1)
        adj2 = random.choice(adjectives)
        noun1 = random.choice([n for n in nouns if (adj1, n) not in adj_noun and (adj2, n) not in adj_noun])
        nouns.remove(noun1)
        noun2 = random.choice(nouns)
        caption = shuffle(caption, adj1, adj2)
        caption = shuffle(caption, noun1, noun2)
        print(caption)
    if len(adjectives) == 2 and len(nouns) == 2:
        caption = shuffle(caption, adjectives[0], adjectives[1])

    if len(adjectives) < 2 and len(nouns) > 1:
        noun1 = random.choice(nouns)
        nouns.remove(noun1)
        noun2 = random.choice(nouns)
        caption = shuffle(caption, noun1, noun2)
    
    if len(nouns) < 2 and len(adjectives) > 1:
        adj1 = random.choice(adjectives)
        adjectives.remove(adj1)
        adj2 = random.choice(adjectives)
        caption = shuffle(caption, adj1, adj2)

    if len(nouns) < 2 and len(nouns) < 2:
        pass

    
    for word_type in ["NOUN", "ADJ", "ADV", "VERB"]:
        word_list = [token.text for token in doc if token.pos_ == word_type]
        #print(word_list)
        assert False
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
    with open(f"COCO/annotations/captions_{set_type}2014.json", 'r') as f:
        data = json.load(f)
    raw_captions_list = data["annotations"]

    processed_captions_list = []

    # shuffling the captions
    for caption_data in tqdm(raw_captions_list):
        pos_caption, neg_captions = shuffle_caption(caption_data["caption"])
        if len(neg_captions) > 0: # we keep only the captions that have negative captions
            caption_data["neg_captions"] = neg_captions
            processed_captions_list.append(caption_data)

    # write files

    data["annotations"] = processed_captions_list
    with open(f"COCO_ORDER/perturbations_{set_type}2014.json", 'w') as f:
        json.dump(data, f)


generate_neg_captions("val")