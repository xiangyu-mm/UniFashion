from lavis.models import load_model_and_preprocess
import json

import torch
import json
from pathlib import Path
import random

import torch
from PIL import Image

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
        random_num = 0.1
        if random_num < 0.25:
            captions = ''.join(
                f"1.{flattened_captions[i].strip('.?, ').capitalize()} 2.{flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions = ''.join(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions = ''.join(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions = ''.join(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image

# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
# prepare the image


dress_types = ['dress', 'shirt', 'toptee']
splits = ['train']
task = 'generate'
for dress_type in dress_types:
    for split in splits:
        new_list = []
        with open('/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/captions/' + f'cap.{dress_type}.{split}.json') as f:
            data = json.load(f)
            for item in tqdm(data):
                if task == 'generate':
                    prompt = "Please generate a detailed caption to describe the " + f'{dress_type}' + "'s style, " \
                                                                                 "color, and other key points."


                    reference_name = item['candidate']
                    target_name = item['target']
                    caption = item['captions']
                    image_captions = generate_randomized_fiq_caption(caption)

                    image_file = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{reference_name}.png"
                    image_file2 = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{target_name}.png"

                    prompt2 = "Generating a detailed caption to the image but " + image_captions + " ."

                    raw_image2 = Image.open(image_file2).convert("RGB")
                    image2 = vis_processors["eval"](raw_image2).unsqueeze(0).to(device)

                    raw_image = Image.open(image_file).convert("RGB")
                    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                    a, _ = model.generate({"image": image, "prompt": prompt})
                    b, _ = model.generate({"image": image2, "prompt": prompt})
                    c, _ = model.generate({"image": image, "prompt": prompt2})

                    print(image_captions)
                    print(a)
                    print(b)
                    print(c)

                    # prompt = "Please generate new captions based on the given image and the Guidance. " \
                    #          "The information in the picture and the guidance are inconsistent, please follow the " \
                    #          "Guidance. Please only briefly describe the " + f'{dress_type}' + "'s style, color, " \
                    #                                                                                   "and other key " \
                    #                                                                                   "points in the " \
                    #                                                                                   "picture. Example: " \
                    #                                                                                   "The clothes' color " \
                    #                                                                                   "is XXX, " \
                    #                                                                                   "stype is XXX, " \
                    #                                                                                   "designing is XXX. " \
                    #                                                                                   "Guidance: "+ \
                    #                                                                                   image_captions

                    # prompt for llava_fashion_7B
                    # prompt = "Please generate a new caption based on the given image and the Guidance. " \
                    #          "If there is any inconsistency between the information in the picture and the guidance, " \
                    #          "please follow the Guidance. The new caption describe the " + f'{dress_type}' + "'s style, " \
                    #          "color, and other key points. Guidance: " + image_captions

                    prompt = "Please generate a new caption based on the given image and the Update Guidance " \
                             " The new caption must includes the " + f'{dress_type}' + "'s style, " \
                             "color, and other key points appeared in the Update Guidance: " + image_captions +\
                             "PLEASE DONT describe the people in image!"

                    # print(cap2)

                    dict_new = {"target": item['target'], "candidate": item['candidate'], "captions": caption,
                                "target_caption": a}

                    new_list.append(dict_new)
                else:
                    prompt = "Please generate a caption based on the given image. " \
                             "The caption describe the " + f'{dress_type}' + "'s style, " \
                             "color, and other key points. DONT describe the woman in image!"

                    reference_name = item['candidate']
                    target_name = item['target']
                    caption = item['captions']
                    image_captions = generate_randomized_fiq_caption(caption)

                    image_file = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{reference_name}.png"
                    image_file2 = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{target_name}.png"

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
                    cap1_list = cap1.split('.')
                    for i in range(len(cap1_list)):
                        # if 'in front of' in cap1_list[i]:
                        #     cap1_list[i] = ''
                        if 'is also wearing' in cap1_list[i]:
                            cap1_list[i] = ''
                        if 'is posing' in cap1_list[i]:
                            cap1_list[i] = ''
                        # candidate_new=candidate_new.join(i)
                    cap1='.'.join(cap1_list)
                    print(cap1)
                    print(image_captions)

                    # prompt = "Please generate new captions based on the given image and the Guidance. " \
                    #          "The information in the picture and the guidance are inconsistent, please follow the " \
                    #          "Guidance. Please only briefly describe the " + f'{dress_type}' + "'s style, color, " \
                    #                                                                                   "and other key " \
                    #                                                                                   "points in the " \
                    #                                                                                   "picture. Example: " \
                    #                                                                                   "The clothes' color " \
                    #                                                                                   "is XXX, " \
                    #                                                                                   "stype is XXX, " \
                    #                                                                                   "designing is XXX. " \
                    #                                                                                   "Guidance: "+ \
                    #                                                                                   image_captions

                    # prompt for llava_fashion_7B
                    # prompt = "Please generate a new caption based on the given image and the Guidance. " \
                    #          "If there is any inconsistency between the information in the picture and the guidance, " \
                    #          "please follow the Guidance. The new caption describe the " + f'{dress_type}' + "'s style, " \
                    #          "color, and other key points. Guidance: " + image_captions

                    prompt = "Example: "+image_captions+"Please generate a new caption based on the given image and the Update Guidance. " \
                             "The caption must includes the " + f'{dress_type}' + "'s style, " \
                             "color, and other key points appeared in the Update Guidance: " + image_captions +\
                             " PLEASE DONT describe the people in image!"

                    args2 = type('Args', (), {
                        "model": model,
                        "tokenizer": tokenizer,
                        "image_processor": image_processor,
                        "context_len": context_len,
                        "model_base": None,
                        "model_name": get_model_name_from_path(model_path),
                        "query": prompt,
                        "conv_mode": None,
                        "image_file": image_file2,
                        "sep": ",",
                        "temperature": 0,
                        "top_p": None,
                        "num_beams": 1,
                        "max_new_tokens": 512
                    })()

                    cap2 = eval_model(args2)
                    cap2_list = cap2.split('.')
                    for i in range(len(cap2_list)):
                        # if 'in front of' in cap2_list[i]:
                        #     cap2_list[i] = ''
                        if 'is also wearing' in cap2_list[i]:
                            cap2_list[i] = ''
                        if 'is posing' in cap2_list[i]:
                            cap2_list[i] = ''
                        # candidate_new=candidate_new.join(i)
                    cap2='.'.join(cap2_list)
                    print(cap2)

                    dict_new = {"target": item['target'], "candidate": item['candidate'], "captions": caption,
                                "target_caption": cap2, "candidate_caption": cap1}

                    new_list.append(dict_new)

        json_file_path = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/captions/' + f'cap.{dress_type}.{split}.llava_iblip.json'
        json_file = open(json_file_path, mode='w')

        json.dump(new_list, json_file, indent=4)
