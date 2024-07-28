import json

import torch
import json
from pathlib import Path
import random
import os

from tqdm import tqdm

splits=['train','val']
dress_types = ['dress', 'toptee', 'shirt']

for dress_type in dress_types:
    new_dict = {}
    for split in splits:
        new_list = []
        with open('/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/image_splits/' + f'split.{dress_type}.{split}.json') as f:
            data = json.load(f)
        with open('/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/captions/' + f'cap.{dress_type}.{split}.json') as f:
            caption_data = json.load(f)

        for item in data:
            for item_caption in caption_data:
                if item_caption['target']==item:
                    dict_new = {"image": item, "caption": item_caption['captions']}
                    break
                else:
                    dict_new = {"image": item, "caption": ''}
            new_list.append(dict_new)

        json_file_path_all = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/' + f'img_w_cap.{split}.{dress_type}.json'
        json_file_all = open(json_file_path_all, mode='w')
        json.dump(new_list, json_file_all, indent=1)