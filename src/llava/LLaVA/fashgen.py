import json
from pathlib import Path
import random

from tqdm import tqdm
import re
import jsonlines
from random import choice, sample

# with jsonlines.open('/home/data2/xiangyu/Code/SPRC/fashionGen/annotation_val_with_title.jsonl', mode="r") as writer:
#     dataset2 = [item for item in writer]
with jsonlines.open('/home/data2/xiangyu/Code/SPRC/fashionGen/annotation_train_only_title.jsonl', mode="r") as writer:
    dataset = [item for item in writer]

new_list = []
for item in dataset:
    image_name = json.loads(item['index'])
    captions = item['description']
    image_path = '/home/data2/xiangyu/Code/SPRC/fashionGen/image/' + f"{image_name}.jpg"
    prompt = "<image>\nPlease generate a caption to describe the image."
    conversation = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': captions}]
    item_dict = {'id': 'fashionGen' + str(image_name), 'image': image_path, 'conversations': conversation}
    new_list.append(item_dict)
    # else:
    #     image_name = choice(image_name)
    #     captions = item['description']
    #     image_path = '/home/data2/xiangyu/Code/SPRC/fashionGen/image/' + f"{image_name}.jpg"
    #
    #     prompt = "<image>\nPlease generate a caption to describe the image."
    #
    #     conversation = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': captions}]
    #
    #     item_dict = {'id': 'fashionGen'+str(image_name), 'image': image_path, 'conversations': conversation}
    #
    #     new_list.append(item_dict)

for item in dataset2:
    image_name = json.loads(item['index'])
    captions = item['description']
    image_path = '/home/data2/xiangyu/Code/SPRC/fashionGen/image_val/' + f"{image_name}.jpg"
    prompt = "<image>\nPlease generate a caption to describe the image."
    conversation = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': captions}]
    item_dict = {'id': 'fashionGen' + str(image_name), 'image': image_path, 'conversations': conversation}
    new_list.append(item_dict)
    # else:
    #     image_name = choice(image_name)
    #     captions = item['description']
    #     image_path = '/home/data2/xiangyu/Code/SPRC/fashionGen/image_val/' + f"{image_name}.jpg"
    #
    #     prompt = "<image>\nPlease generate a caption to describe the image."
    #
    #     conversation = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': captions}]
    #
    #     item_dict = {'id': 'fashionGen'+str(image_name), 'image': image_path, 'conversations': conversation}
    #
    #     new_list.append(item_dict)

json_file_path = '/home/data2/xiangyu/Code/SPRC/fashionGen/' + f'llava_caption_fashion_only_title.json'
json_file = open(json_file_path, mode='w')
json.dump(new_list, json_file, indent=4)
