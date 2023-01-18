import json
import cv2
from os import path
import csv
from tqdm import tqdm
from itertools import product

RELATIONS_TO_DEL = ["standing_by", "standing next to", "touching", "connected to", "by", "sitting next to", 'leaning against', "around", "parked next to", "chained to", "walking by", "hugging", "between", "on the other side of", "near", "sitting beside", "next to", "with", "tied to", "pulled by", "beside", "on the side of", "playing", "standing near", "parked near", "surrounding"]
RELATIONS_TO_KEEP = ["above", "at", "behind", "below", "beneath", "in", "in front of", "inside", "on", "on top of", "to the left of", "to the right of", 'under', "carrying", "covered by", "covered in", "covered with", "covering", "cutting", "eating", "feeding", "grazing on", "hanging on", "holding", "leaning on", "looking at", "lying in", "lying on", "parked on", "reflected in", "resting on", "riding", "sitting at", "sitting in", "sitting on", "sitting on top of", "standing by", "standing in", "standing on", "surrounded by", "using", "walking in", "walking on", "watching", "wearing"]

RELATIONS = set()
ATTRIBUTES = set()

def init_files(set_type):
    path_dataset_relations_file = f"VGA_VGR/{set_type}/dataset_relations.csv"
    path_dataset_attributes_file = f"VGA_VGR/{set_type}/dataset_attributes.csv"

    if path.exists(path_dataset_relations_file):
        dataset_relations_file = open(path_dataset_relations_file , 'w')
    else:
        dataset_relations_file = open(path_dataset_relations_file, 'x')

    if path.exists(path_dataset_attributes_file):
        dataset_attributes_file = open(path_dataset_attributes_file, 'w')
    else:
        dataset_attributes_file = open(path_dataset_attributes_file, 'x')

    writer_rel = csv.writer(dataset_relations_file)
    writer_att = csv.writer(dataset_attributes_file)

    return writer_rel, writer_att, dataset_relations_file, dataset_attributes_file


def create_relations(object1, object2, relation, id, writer_rel):
    if relation in RELATIONS_TO_KEEP:
        writer_rel.writerow([id, f"the {object1} is {relation} the {object2}", 1])
        writer_rel.writerow([id, f"the {object2} is {relation} the {object1}", 0])
        RELATIONS.add(relation)

def create_attributes(object1, object2, attribute1, attribute2, id, writer_att):
    if (attribute2, attribute1) not in ATTRIBUTES:
        ATTRIBUTES.add((attribute1, attribute2))
    writer_att.writerow([id, f"the {attribute1} {object1} and the {attribute2} {object2}", 1])
    writer_att.writerow([id, f"the {attribute2} {object1} and the {attribute1} {object2}", 0])


def extract_image(obj1_data, obj2_data, image_file_name):
    x_list = []
    y_list = []
    for obj_data in [obj1_data, obj2_data]:
        x_list.append(obj_data["x"])
        x_list.append(obj_data["x"] + obj_data["w"])
        y_list.append(obj_data["y"])
        y_list.append(obj_data["y"] + obj_data["h"])

    min_y = min(y_list)
    max_y = max(y_list)
    min_x = min(x_list)
    max_x = max(x_list)

    image = cv2.imread(f"GQA_raw/images/{image_file_name.split('_')[0]}.jpg")
    extracted_image = image[min_y: min_y + (max_y - min_y), min_x: min_x + (max_x - min_x)]
    cv2.imwrite(f"VGA_VGR/images/{image_file_name}.jpg", extracted_image)


def is_good_size(object_id, image_data):
    object_data = image_data["objects"][object_id]
    return object_data["w"] > image_data["width"] / 4 and object_data["h"] > image_data["height"] / 4


def generate_dataset(set_type):

    writer_rel, writer_att, dataset_relations_file, dataset_attributes_file = init_files(set_type)

    with open(f"GQA_raw/{set_type}_sceneGraphs.json", "r") as f:
        data_raw = json.load(f)

    for id_image, image_data in tqdm(data_raw.items()):
        cpt = 0 #how many images extracted from this image

        if path.exists(f"GQA_raw/images/{id_image}.jpg"):
            obj_pairs = dict()
            for object_id, object_data in image_data["objects"].items():
                if is_good_size(object_id, image_data):
                    for relation_data in object_data["relations"]:
                        if (object_id, relation_data["object"]) not in obj_pairs.keys() and (relation_data["object"], object_id) not in obj_pairs.keys():
                            if is_good_size(relation_data["object"], image_data):
                                obj_pairs[(object_id, relation_data["object"])]= relation_data["name"]


            for (obj1, obj2), relation in obj_pairs.items():
                name_obj1 = image_data["objects"][obj1]["name"]
                name_obj2 = image_data["objects"][obj2]["name"]
                if name_obj1 != name_obj2:
                    process_image_id = id_image+'_'+str(cpt)

                    create_relations(name_obj1, name_obj2, relation, process_image_id, writer_rel)

                    obj1_data = image_data["objects"][obj1]
                    obj2_data = image_data["objects"][obj2]
                    cpt +=1
                    #extract_image(obj1_data, obj2_data, process_image_id)

                    attributes_obj1 = list(set(image_data["objects"][obj1]["attributes"]) - set(image_data["objects"][obj2]["attributes"]))
                    attributes_obj2 = list(set(image_data["objects"][obj2]["attributes"]) - set(image_data["objects"][obj1]["attributes"]))

                    att_pairs = []
                    if len(attributes_obj1) > 0 and len(attributes_obj2) > 0:
                        for (att_obj1, att_obj2) in product(attributes_obj1, attributes_obj2):
                            if att_obj1 != att_obj2 and (att_obj1, att_obj2) not in att_pairs and (att_obj2, att_obj1) not in att_pairs :
                                att_pairs.append((att_obj1, att_obj2))
                                create_attributes(name_obj1, name_obj2, att_obj1, att_obj2, process_image_id, writer_att)
        else:
            continue

    dataset_relations_file.close()
    dataset_attributes_file.close()

generate_dataset("train")
print("nb rel ", len(RELATIONS))
print("nb att ", len(ATTRIBUTES))