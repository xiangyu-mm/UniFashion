from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json

import torch
import json
from pathlib import Path
import random
import os

from tqdm import tqdm


def generate_randomized_fiq_caption(flattened_captions):
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = ''
    for i in range(0, len(flattened_captions), 2):
        # random_num = random.random()
        random_num=0
        if random_num < 0.25:
            captions = ''.join(
                f"{flattened_captions[i].strip('.?, ').capitalize()}, {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions = ''.join(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()}, {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions = ''.join(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions = ''.join(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


model_path = "/home/data2/xiangyu/llava/LLaVA/checkpoints/llava-v1.5-13B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "/home/data2/xiangyu/llava/LLaVA/checkpoints/llava-v1.5-13B"
splits=['train','val']
dress_types = ['dress', 'toptee', 'shirt']
task = 'generate'
for dress_type in dress_types:
    new_dict = {}
    for split in splits:
        new_list = []
        folder_path = f'/home/data2/xiangyu/Code/StableVITON/dataset/{split}/cloth'

        if task == 'VITON':

            data = os.listdir(folder_path)

            for image_name in data:

                prompt = "Please generate a caption to describe the cloth in this image, the caption describe the color, short or long, sleeve's style, pattern's style, design and other key points."

                image_file = f'/home/data2/xiangyu/Code/StableVITON/dataset/{split}/cloth/' + f"{image_name}"

                args1 = type('Args', (), {
                    "model": model,
                    "tokenizer": tokenizer,
                    "image_processor": image_processor,
                    "context_len": context_len,
                    "model_base": None,
                    "model_name": get_model_name_from_path(model_path),
                    "query": prompt,
                    "conv_mode": None,
                    "image_file": image_file,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512
                })()

                cap1 = eval_model(args1)
                print(cap1)

                dict_new = {"image": image_name, "caption": cap1}
                new_dict[image_name[:-7]]=cap1
                new_list.append(dict_new)

            json_file_path = '/home/data2/xiangyu/Code/StableVITON/dataset/' + f'cap.{split}.caption.json'
            json_file = open(json_file_path, mode='w')
            json.dump(new_list, json_file, indent=4)

    # json_file_path_all = '/home/data2/xiangyu/Code/StableVITON/dataset/' + f'all.{split}.caption.json'
    # json_file_all = open(json_file_path_all, mode='w')
    # json.dump(new_dict, json_file_all, indent=1)
        else:
            with open('/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/' + f'img_w_cap.{split}.{dress_type}.json') as f:
                data = json.load(f)
                for item in tqdm(data):
                    image_name = item['image']
                    if len(item['caption']) >1:
                        caption = generate_randomized_fiq_caption(item['caption'])
                        print(caption)
                    else:
                        caption = ''
                    if task == 'generate':
                        if dress_type == 'dress':
                            prompt = "Please generate a caption to describe the " + f'{dress_type}' + \
                                 " The caption describe the " + f'{dress_type}' + "'s" \
                      "color, short, long, sleeve's style, pattern/graphic/logo's style, design and other key points. " \
                      "You should select sufficient (at least 5) appropriate words from following: {revealing, conservative, western, " \
                      "eastern, sexy, modest, patterned, plain, frilly, simple, buttons, crochet, collar, " \
                      "floral, plain, elegant, casual, monochromatic, colorful, flowery, plain, shiny, matte, " \
                      "darker, lighter, fitted, loose, print, plain, flare, tighter, looser}. Reference: " + f'{caption}' +\
                    "."

                        elif dress_type == 'toptee':
                            prompt = "Please generate a caption to describe the " + f'{dress_type}' + \
                                 " The caption describe the " + f'{dress_type}' + "'s" \
                      "color, short or long, sleeve's style, pattern/graphic/logo's style, design and other key points. " \
                      "Please select sufficient appropriate words from: revealing or conservative, western or " \
                      "eastern, sexy or modest, patterned or plain, frilly or simple, buttons or not, crochet or not, collar or not, " \
                      "floral or plain, elegant or casual, monochromatic or colorful, flowery or plain, shiny or matte, " \
                      "darker or lighter, fitted or loose, print or plain, flare or not, tighter or looser..." + f'{caption}' +\
                    ". Do not use comparative words."

                        elif dress_type == 'shirt':
                            prompt = "Please generate a caption to describe the " + f'{dress_type}' + \
                                     " The caption describe the " + f'{dress_type}' + "'s" \
                      "color, short or long, sleeve's style, pattern/graphic/logo's style, design and other key points. " \
                      "Please select sufficient appropriate words from: revealing or conservative, western or " \
                      "eastern, sexy or modest, patterned or plain, frilly or simple, buttons or not, crochet or not, collar or not, " \
                      "floral or plain, elegant or casual, monochromatic or colorful, flowery or plain, shiny or matte, " \
                      "darker or lighter, fitted or loose, print or plain, flare or not, tighter or looser..." + f'{caption}' +\
                    ". Do not use comparative words."

                        image_file = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{image_name}.png"

                        args1 = type('Args', (), {
                            "model": model,
                            "tokenizer": tokenizer,
                            "image_processor": image_processor,
                            "context_len": context_len,
                            "model_base": None,
                            "model_name": get_model_name_from_path(model_path),
                            "query": prompt,
                            "conv_mode": None,
                            "image_file": image_file,
                            "sep": ",",
                            "temperature": 0,
                            "top_p": None,
                            "num_beams": 1,
                            "max_new_tokens": 512
                        })()

                        cap1 = eval_model(args1)
                        print(cap1)
        #
        #                 prompt2 = "Caption: "f'{cap1}'+"\n Please revision the caption according to the following point: 1. Removing the description about " \
        #                           "people appeared in this image, such as (woman, man, mannequin, model ...) 2. keeping the " \
        #                            "description about the " + f'{dress_type}'+". 3. Removing using phrases like 'The woman is wearing..."
        #
        #
        #                 args2 = type('Args', (), {
        #                     "model": model,
        #                     "tokenizer": tokenizer,
        #                     "image_processor": image_processor,
        #                     "context_len": context_len,
        #                     "model_base": None,
        #                     "model_name": get_model_name_from_path(model_path),
        #                     "query": prompt2,
        #                     "conv_mode": None,
        #                     "image_file": None,
        #                     "sep": ",",
        #                     "temperature": 0,
        #                     "top_p": None,
        #                     "num_beams": 1,
        #                     "max_new_tokens": 512
        #                 })()
        #
        #                 # cap2 = eval_model(args2)
        #                 # print(cap2)
        #
        #                 dict_new = {"image": item, "caption": cap1}
        #
        #                 new_list.append(dict_new)
        #
        #             elif task == 'VITON':
        #
        #                 prompt = "Please generate a short caption to describe the cloth."
        #
        #                 image_file = f'/home/data2/xiangyu/Code/StableVITON/dataset/{splits}/cloth' + f"{image_name}.jpg"
        #
        #                 args1 = type('Args', (), {
        #                     "model": model,
        #                     "tokenizer": tokenizer,
        #                     "image_processor": image_processor,
        #                     "context_len": context_len,
        #                     "model_base": None,
        #                     "model_name": get_model_name_from_path(model_path),
        #                     "query": prompt,
        #                     "conv_mode": None,
        #                     "image_file": image_file,
        #                     "sep": ",",
        #                     "temperature": 0,
        #                     "top_p": None,
        #                     "num_beams": 1,
        #                     "max_new_tokens": 512
        #                 })()
        #
        #                 cap1 = eval_model(args1)
        #                 print(cap1)
        #
        #                 dict_new = {"image": item, "caption": cap1}
        #
        #                 new_list.append(dict_new)
        #             else:
        #                 prompt = "Please generate a caption based on the given image. " \
        #                          "The caption describe the " + f'{dress_type}' + "'s style, " \
        #                          "color, and other key points. DONT describe the woman in image!"
        #
        #                 reference_name = item['candidate']
        #                 target_name = item['target']
        #                 caption = item['captions']
        #                 image_captions = generate_randomized_fiq_caption(caption)
        #
        #                 image_file = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{reference_name}.png"
        #                 image_file2 = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{target_name}.png"
        #
        #                 args1 = type('Args', (), {
        #                     "model": model,
        #                     "tokenizer": tokenizer,
        #                     "image_processor": image_processor,
        #                     "context_len": context_len,
        #                     "model_base": None,
        #                     "model_name": get_model_name_from_path(model_path),
        #                     "query": prompt,
        #                     "conv_mode": None,
        #                     "image_file": image_file,
        #                     "sep": ",",
        #                     "temperature": 0,
        #                     "top_p": None,
        #                     "num_beams": 1,
        #                     "max_new_tokens": 512
        #                 })()
        #
        #                 cap1 = eval_model(args1)
        #                 print(cap1)
        #                 print(image_captions)
        #
        #                 # prompt = "Please generate new captions based on the given image and the Guidance. " \
        #                 #          "The information in the picture and the guidance are inconsistent, please follow the " \
        #                 #          "Guidance. Please only briefly describe the " + f'{dress_type}' + "'s style, color, " \
        #                 #                                                                                   "and other key " \
        #                 #                                                                                   "points in the " \
        #                 #                                                                                   "picture. Example: " \
        #                 #                                                                                   "The clothes' color " \
        #                 #                                                                                   "is XXX, " \
        #                 #                                                                                   "stype is XXX, " \
        #                 #                                                                                   "designing is XXX. " \
        #                 #                                                                                   "Guidance: "+ \
        #                 #                                                                                   image_captions
        #
        #                 # prompt for llava_fashion_7B
        #                 # prompt = "Please generate a new caption based on the given image and the Guidance. " \
        #                 #          "If there is any inconsistency between the information in the picture and the guidance, " \
        #                 #          "please follow the Guidance. The new caption describe the " + f'{dress_type}' + "'s style, " \
        #                 #          "color, and other key points. Guidance: " + image_captions
        #
        #                 prompt = "Example: "+image_captions+"Please generate a new caption based on the given image and the Update Guidance. " \
        #                          "The caption must includes the " + f'{dress_type}' + "'s style, " \
        #                          "color, and other key points appeared in the Update Guidance: " + image_captions +\
        #                          " PLEASE DONT describe the people in image!"
        #
        #                 args2 = type('Args', (), {
        #                     "model": model,
        #                     "tokenizer": tokenizer,
        #                     "image_processor": image_processor,
        #                     "context_len": context_len,
        #                     "model_base": None,
        #                     "model_name": get_model_name_from_path(model_path),
        #                     "query": prompt,
        #                     "conv_mode": None,
        #                     "image_file": image_file2,
        #                     "sep": ",",
        #                     "temperature": 0,
        #                     "top_p": None,
        #                     "num_beams": 1,
        #                     "max_new_tokens": 512
        #                 })()
        #
        #                 cap2 = eval_model(args2)
        #                 cap2_list = cap2.split('.')
        #                 for i in range(len(cap2_list)):
        #                     # if 'in front of' in cap2_list[i]:
        #                     #     cap2_list[i] = ''
        #                     if 'is also wearing' in cap2_list[i]:
        #                         cap2_list[i] = ''
        #                     if 'is posing' in cap2_list[i]:
        #                         cap2_list[i] = ''
        #                     # candidate_new=candidate_new.join(i)
        #                 cap2='.'.join(cap2_list)
        #                 print(cap2)
        #
        #                 dict_new = {"target": item['target'], "candidate": item['candidate'], "captions": caption,
        #                             "target_caption": cap2, "candidate_caption": cap1}
        #
        #                 new_list.append(dict_new)
