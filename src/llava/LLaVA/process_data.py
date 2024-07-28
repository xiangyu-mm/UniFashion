import json
from pathlib import Path
import random

from tqdm import tqdm
import re


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


# model_path = "/home/data2/xiangyu/llava/LLaVA/checkpoints/llava-v1.5-13B"
#
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )
#
# model_path = "/home/data2/xiangyu/llava/LLaVA/checkpoints/llava-v1.5-13B"
annotation_file = open('/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/captions/annotation_target.json', "w")
dress_types = ['toptee','dress','shirt']
splits = ['train','val']
new_list = []
task = 'after_process'
for dress_type in dress_types:
    for split in splits:
        with open(
                '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/captions/' + f'cap.{dress_type}.{split}.process.json') as f:
            data = json.load(f)
            if task == 'shirt' or task == 'after_process':
                new_list = []
            for item in tqdm(data):

                reference_name = item['candidate']
                target_name = item['target']
                target_caption = item['target_caption']
                caption = item['captions']
                candidate_caption = item['candidate_caption']

                if task == 'shirt':

                    candidate_caption_list = re.split(r'[,.]', candidate_caption.lower())
                    target_caption_list = re.split(r'[,.]', target_caption.lower())
                    candidate_new = ''
                    for i in range(len(candidate_caption_list)):
                        # if 'she is' in candidate_caption_list[i]:
                        #     candidate_caption_list[i] = ''
                        if 'which complement' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'the man is wearing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'posing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'is worn by' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'he is also wearing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'showcasing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        # candidate_new=candidate_new.join(i)
                    candidate_new = '.'.join(candidate_caption_list)

                    target_new = ''
                    for i in range(len(target_caption_list)):
                        if 'which complement' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'the man is wearing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'posing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'is worn by' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'he is also wearing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'showcasing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                    target_new = '.'.join(target_caption_list)


                    # target_new = re.sub(
                    #     r"a (woman|mannequin) (is )?wearing|is worn by a woman|is displayed on a mannequin"
                    #     r"|is hanging on a mannequin|is standing in (front of )?a white background|"
                    #     r"is displayed on a white background", '',
                    #     target_new)
                    # candidate_new = re.sub(
                    #     r"a (woman|mannequin) (is )?wearing|is worn by a woman|is displayed on a mannequin"
                    #     r"|is hanging on a mannequin|is standing in (front of )?a white background|"
                    #     r"is displayed on a white background", '',
                    #     candidate_new)

                    dict_new = {"target": item['target'], "candidate": item['candidate'], "captions": caption,
                                "target_caption": target_new, "candidate_caption": candidate_new}

                    new_list.append(dict_new)

                elif task == 'after_process':

                    # candidate_caption = candidate_caption.split('She is also wearing')[0].split('The woman is wearing high')[0].split('She is standing in front of a')[0].split('The woman is wearing black heels')[0].split('The woman is wearing a necklace')[0]
                    # target_caption = target_caption.split('She is also wearing')[0].split('The woman is wearing high')[0].split('She is standing in front of a')[0].split('The woman is wearing black heels')[0].split('The woman is wearing a necklace')[0]

                    candidate_caption_list = re.split(r'[,.]', candidate_caption.lower())
                    target_caption_list = re.split(r'[,.]', target_caption.lower())
                    candidate_new = ''
                    for i in range(len(candidate_caption_list)):
                        # if 'she is' in candidate_caption_list[i]:
                        #     candidate_caption_list[i] = ''
                        if 'the woman is standing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'the mannequin is standing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'showing off' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'showcasing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'camera' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'mannequin ' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'mannequin.' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'making it' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'the woman is wearing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'woman is also wearing' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'which complement' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        if 'she is' in candidate_caption_list[i] and i != 0:
                            candidate_caption_list[i] = ''
                        # candidate_new=candidate_new.join(i)
                    candidate_new='.'.join(candidate_caption_list)

                    # candidate_caption_list = re.split(r'the woman is standing|the mannequin is standing|she is standing',
                    #                                   candidate_caption.lower())
                    # if len(candidate_caption_list) > 1:
                    #     if len(candidate_caption_list) > 2:
                    #         print(candidate_caption_list)
                    #     candidate_caption_list[1] = ''
                    # candidate_new = '.'.join(candidate_caption_list)

                    target_new = ''
                    for i in range(len(target_caption_list)):
                        # if 'she is' in target_caption_list[i] and i != 0:
                        #     target_caption_list[i] = ''
                        if 'the woman is standing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'the mannequin is standing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'showing off' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'showcasing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'camera' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'mannequin ' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'mannequin.' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'making it' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'the woman is wearing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'woman is also wearing' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'which complement' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                        if 'she is' in target_caption_list[i] and i != 0:
                            target_caption_list[i] = ''
                    target_new = '.'.join(target_caption_list)

                    # target_caption_list = target_new.lower().split('the woman is standing')
                    # target_caption_list = re.split(r'the woman is standing|the mannequin is standing|she is standing',
                    #                                target_caption.lower())
                    # if len(target_caption_list) > 1:
                    #     if len(target_caption_list) > 2:
                    #         print(target_caption_list)
                    #     target_caption_list[1] = ''
                    # target_new = '.'.join(target_caption_list)

                    pattern = re.compile(r'(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])')

                    target_new = re.sub(
                        r"a (woman|mannequin) (is )?wearing|is worn by a woman|is displayed on a mannequin"
                        r"|is hanging on a mannequin|is standing in (front of )?a white background|"
                        r"is displayed on a white background", '',
                        target_new)
                    candidate_new = re.sub(
                        r"a (woman|mannequin) (is )?wearing|is worn by a woman|is displayed on a mannequin"
                        r"|is hanging on a mannequin|is standing in (front of )?a white background|"
                        r"is displayed on a white background", '',
                        candidate_new)

                    dict_new = {"target": item['target'], "candidate": item['candidate'], "captions": caption,
                                "target_caption": target_new, "candidate_caption": candidate_new}

                    new_list.append(dict_new)

                else:

                    image_captions = generate_randomized_fiq_caption(caption)

                    # prompt = "<image>\nPlease generate a new caption based on the given image and the Update Guidance. " \
                    #          "The new caption describe the " + f'{dress_type}' + "'s style, " \
                    #          "color, and other key points. Update Guidance: " + image_captions

                    prompt = "<image>\nPlease generate a new caption to describe the " + f'{dress_type}' +\
                             " based on the given image and the Update Guidance: " + image_captions + " Caption: "

                    prompt_original = "<image>\nPlease generate a caption to describe the " + f'{dress_type}' \
                                      + " based on the given image. " + "Caption: "

                    image_file = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{reference_name}.png"

                    target_file = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/images/' + f"{target_name}.png"

                    conversation = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': target_caption}]

                    conversation_reference = [{'from': 'human', 'value': prompt_original},
                                              {'from': 'gpt', 'value': candidate_caption}]

                    conversation_target = [{'from': 'human', 'value': prompt_original},
                                           {'from': 'gpt', 'value': target_caption}]
                    # conversation_target = [{'from': 'human', 'value': prompt_original},{'from': 'gpt', 'value': target_caption}]

                    item_dict = {'id': reference_name, 'image': image_file, 'conversations': conversation}

                    item_target = {'id': target_name + '_target', 'image': target_file, 'conversations': conversation_target}

                    item_reference = {'id': reference_name + '_ref', 'image': image_file,
                                      'conversations': conversation_reference}

                    # new_list.append(item_reference)
                    # new_list.append(item_target)
                    new_list.append(item_dict)



                    # json.dump(new_list, annotation_file, indent=4)

        # json_file_path = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/captions/' + f'cap.{dress_type}.{split}.process1.json'
        # json_file = open(json_file_path, mode='w')
        #
        # json.dump(new_list, json_file, indent=4)

json_file_path = '/home/data2/xiangyu/Code/SPRC/fashionIQ_dataset/captions/' + f'train_composed_fashion.json'
json_file = open(json_file_path, mode='w')

json.dump(new_list, json_file, indent=4)
