from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import json

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

pretrained = "/home/data1/xiangyu/code/LLaVA-NeXT/ckeckpoint/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

splits=['train','val']
# dress_types = ['dress', 'toptee', 'shirt']
# dress_types = ['dress']
# dress_types = ['toptee']
dress_types = ['shirt']
for dress_type in dress_types:
    new_dict = {}
    for split in splits:
        new_list = []
        with open('/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/' + f'img_w_cap.{split}.{dress_type}.json') as f:
            data = json.load(f)
            for item in tqdm(data):
                image_name = item['image']
                if len(item['caption']) > 1:
                    caption = generate_randomized_fiq_caption(item['caption'])
                    print('caption:', caption)
                else:
                    caption = ''

                image_file = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{image_name}.png"
                image = Image.open(image_file)

                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

                conv_template = "llava_llama_3" # Make sure you use correct chat template for different models

                prompt = "Please generate a caption to describe the " + f'{dress_type}' + "'s " \
              "color, short or long, sleeve's style, pattern/graphic/logo's style, design and other key points. " \
              "You should select sufficient (at least 5) appropriate words from following: {revealing, conservative, western, " \
              "eastern, sexy, modest, patterned, plain, frilly, simple, buttons, crochet, collar, " \
              "floral, plain, elegant, casual, monochromatic, colorful, flowery, plain, shiny, matte, " \
              "darker, lighter, fitted, loose, print, plain, flare, tighter, looser}. The reference caption is: " + f'{caption}' + \
              ". Do not use comparative words. Please use the original form."

                question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                image_sizes = [image.size]

                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                )
                text_outputs = tokenizer.decode(cont[0], skip_special_tokens=True)
                dict_new = {"image": image_name, "caption": text_outputs, "text_guidance": caption}
                new_list.append(dict_new)
                print(text_outputs)
        json_file_path_all = '/home/data1/xiangyu/code/LLaVA-NeXT/caption_fashioniq/' + f'next_llava.{split}.{dress_type}.caption.json'
        json_file_all = open(json_file_path_all, mode='w')
        json.dump(new_list, json_file_all, indent=1)
