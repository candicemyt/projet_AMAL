import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import json
import spacy
from copy import copy
import random
from tqdm import tqdm


def swap(caption, mot1, mot2):
    splitted = caption.split()
    idx1 = splitted.index(mot1)
    idx2 = splitted.index(mot2)
    ret = copy(splitted)
    ret[idx1] = mot2
    ret[idx2] = mot1
    return ' '.join(ret)


def shuffle(caption, first_token, second_token):
    """
    Shuffle the caption : replacing first_token by second_token and vice versa
    """
    shuffled = second_token.join(part.replace(second_token, first_token) for part in caption.split(first_token))
    return shuffled


def trigram(caption):
    splitted = caption.split()
    trigrams = []
    for i in range(0, len(splitted) - 2, 3):
        trigrams.append((splitted[i], splitted[i + 1], splitted[i + 2]))

    if len(trigrams) < 2:
        return None

    leftover = splitted[i + 3:]
    ret = copy(trigrams)

    sampled = random.sample(trigrams, len(trigrams))

    for i in range(0, len(sampled) - 1, 2):
        id1 = trigrams.index(sampled[i])
        id2 = trigrams.index(sampled[i + 1])
        ret[id1] = sampled[i + 1]
        ret[id2] = sampled[i]

    lst = []
    for t in ret:
        for w in t:
            lst.append(w)
    return ' '.join(lst + leftover)


def shuffle_within_trigram(caption):
    splitted = caption.split()
    trigrams = []
    for i in range(0, len(splitted) - 2, 3):
        trigrams.append([splitted[i], splitted[i + 1], splitted[i + 2]])

    leftover = splitted[i + 3:]
    # for each tri gram we randomly choose two words two swap
    for t in trigrams:
        t1, t2 = random.sample(t, k=2)
        id1 = t.index(t1)
        id2 = t.index(t2)
        t[id1] = t2
        t[id2] = t1

    caption = []
    for t in trigrams:
        for w in t:
            caption.append(w)
    return ' '.join(caption + leftover)


def shuffle_caption(caption):
    """
    Shuffle 5 times (maximum) each caption of captions
    :param caption: string (caption)
    :return: the positive captions that are kept and a list of negative captions for each one
    """
    nlp = spacy.load("en_core_web_sm")
    caption = caption.replace(".", '')
    caption = caption.replace(",", '')
    caption = caption.replace("-", ' ')
    caption = caption.replace("'", "")
    caption = caption.replace("\n", "")
    caption = caption.replace('"', "")
    for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+]":
        caption = caption.replace(c, ' ')

    shuffled_captions = []

    doc = nlp(caption)  # pos tagging with spacy

    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    adj_noun = [(doc[i - 1].text, doc[i].text) for i in range(1, len(doc)) if
                doc[i].pos_ == "NOUN" and doc[i - 1].pos_ == "ADJ"]

    try:
        if (len(adjectives) > 2 and len(nouns) > 2):
            adj1, adj2 = random.sample(adjectives, k=2)
            noun1 = random.choice([n for n in nouns if (adj1, n) not in adj_noun and (adj2, n) not in adj_noun])
            nouns.remove(noun1)
            noun2 = random.choice(nouns)
            tmp_caption = swap(caption, adj1, adj2)
            shuffled = swap(tmp_caption, noun1, noun2)
            shuffled_captions.append(shuffled)

        elif len(adjectives) > 2 and len(nouns) == 2:
            adj1 = random.choice(
                [adj for adj in adjectives if (adj, nouns[0]) not in adj_noun and (adj, nouns[1]) not in adj_noun])
            adjectives.remove(adj1)
            adj2 = random.choice(adjectives)
            shuffled = swap(caption, adj1, adj2)
            shuffled = swap(shuffled, nouns[0], nouns[1])

        elif len(adjectives) == 2 and len(nouns) > 2:
            noun1 = random.choice(
                [n for n in nouns if (adjectives[0], n) not in adj_noun and (adjectives[1], n) not in adj_noun])
            nouns.remove(noun1)
            noun2 = random.choice(nouns)
            shuffled = swap(caption, adjectives[0], adjectives[1])
            shuffled = swap(shuffled, noun1, noun2)

        elif len(adjectives) == 2 and len(nouns) == 2:
            shuffled = swap(caption, adjectives[0], adjectives[1])
            shuffled_captions.append(shuffled)

        elif len(adjectives) < 2 and len(nouns) > 1:
            noun1 = random.choice(nouns)
            nouns.remove(noun1)
            noun2 = random.choice(nouns)
            shuffled = swap(caption, noun1, noun2)
            shuffled_captions.append(shuffled)

        elif len(nouns) < 2 and len(adjectives) > 1:
            adj1 = random.choice(adjectives)
            adjectives.remove(adj1)
            adj2 = random.choice(adjectives)
            shuffled = swap(caption, adj1, adj2)
            shuffled_captions.append(shuffled)

        elif len(nouns) < 2 and len(nouns) < 2:
            shuffled = None
            pass
    except:
        shuffled = None

    # shuffle anything but nouns and adj
    others = [token.text for token in doc if
              token.pos_ != "NOUN" and token.pos_ != "ADJ" and token.pos_ != "DET" and token.pos_ != "SPACE"]
    words = random.sample(others, len(others))  # choose the order at random

    shuffled = copy(caption)
    for i in range(0, len(words) - 1, 1):
        try:
            shuffled = swap(shuffled, words[i], words[i + 1])
        except:
            shuffled = None

    if shuffled is not None:
        shuffled_captions.append(shuffled)

    # shuffle tri grams:
    try:
        shuffled = trigram(caption)
    except:
        shuffled = None
    if shuffled is not None:  # to ensure only negative captions are generated
        shuffled_captions.append(shuffled)

    # shuffle within tri grams:
    try:
        shuffled = shuffle_within_trigram(caption)
    except:
        shuffled = None

    if shuffled is not None:
        shuffled_captions.append(shuffled)

    return caption, shuffled_captions


def generate_neg_captions(set_type):
    # open file
    with open(f"COCO/annotations/captions_{set_type}2014.json", 'r') as f:
        data = json.load(f)
    raw_captions_list = data["annotations"]

    processed_captions_list = []

    cpt = {1: 0, 2: 0, 3: 0, 4: 0}
    # shuffling the captions
    for i in tqdm(range(0, len(raw_captions_list))):
        caption_data = raw_captions_list[i]
        pos_caption, neg_captions = shuffle_caption(caption_data["caption"])
        if len(neg_captions) > 0:  # we keep only the captions that have negative captions
            caption_data["neg_captions"] = neg_captions
            processed_captions_list.append(caption_data)
            cpt[len(neg_captions)] += 1

        if i % 2000 == 0 and i > 1:
            print(f"Statistics (1->4 captions) {[v / i for k, v in cpt.items()]}")

    # write files
    data["annotations"] = processed_captions_list
    with open(f"COCO_ORDER/perturbations_{set_type}2014.json", 'w') as f:
        json.dump(data, f)


generate_neg_captions("val")
