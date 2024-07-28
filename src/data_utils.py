import json
from pathlib import Path
from typing import Union, List, Dict, Literal

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import os
import numpy as np

from torchvision import transforms

import matplotlib.pyplot as plt

import jsonlines
from random import sample
import random

from torchvision.transforms.functional import InterpolationMode

base_path = Path(__file__).absolute().parents[1].absolute()


def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> str:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = ''
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions = ''.join(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions = ''.join(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions = ''.join(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions = ''.join(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def diffusion_transform():
    return Compose(
        [
            TargetPad(1.25, 224),
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def get_code(image: str):
    image_name = image + '.npy'
    code_path = os.path.join('/home/data2/xiangyu/Data/image2code/4X8/fashion', image_name)
    code = np.load(code_path).tolist()
    code_string = ''
    for id, item in enumerate(code):
        lst1 = list(map(lambda x: str(x), item))
        str1 = ' '.join(lst1)
        code_string = code_string + '<c>' + str1
    return code_string


class Fashion200KDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, preprocess: callable, usev=False, mode='pretrain'):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.split = split
        self.usev = usev
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")

        self.preprocess = preprocess
        self.mode = mode

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        if self.mode == 'pretrain':
            with jsonlines.open(base_path / 'fashion200K' / f'annotation_pairs_{split}.jsonl', mode="r") as writer:
                self.dataset = [item for item in writer]
        else:
            with jsonlines.open(base_path / 'fashion200K' / f'annotation_{split}_object.jsonl', mode="r") as writer:
                self.dataset = [item for item in writer]

        print(f"FashionGen {split} dataset initialized")

    def __getitem__(self, index):

        if self.mode == 'pretrain':
            item = self.dataset[index]
            reference_name = item['image_path']
            image_captions = item['caption']
            image_path = base_path / 'fashion200K' / f"{reference_name}"
            image = self.preprocess(PIL.Image.open(image_path))

            return image, image_captions

        else:

            if self.split == 'train':
                item = self.dataset[index]
                image_name_list = item['all_images']
                image_captions = item['caption']
                if len(image_name_list) > 1:
                    ref_target = sample(image_name_list, 2)
                    reference_name = ref_target[0]
                    target_name = ref_target[1]
                else:
                    reference_name = target_name = image_name_list[0]
                reference_image_path = base_path / 'fashion200K' / f"{reference_name}"
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                # toPIL = transforms.ToPILImage()
                # img = toPIL(reference_image)
                # img.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test1.png')

                # target_name = self.triplets[index]['target']
                target_image_path = base_path / 'fashion200K' / f"{target_name}"
                target_image = self.preprocess(PIL.Image.open(target_image_path))
                # img = toPIL(reference_image)
                # img.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test2.png')

                return reference_image, target_image, image_captions

            elif self.split == 'val':
                item = self.dataset[index]
                image_name_list = json.loads(item['index'])
                image_captions = item['description']
                if len(image_name_list) > 1:
                    ref_target = sample(image_name_list, 2)
                    reference_name = ref_target[0]
                    target_name = ref_target[1]
                else:
                    reference_name = target_name = image_name_list[0]
                reference_image_path = base_path / 'fashionGen' / 'image_val' / f"{reference_name}.jpg"
                # print(reference_image_path)
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                # toPIL = transforms.ToPILImage()
                # img = toPIL(reference_image)
                # img.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test1.png')

                # target_name = self.triplets[index]['target']
                target_image_path = base_path / 'fashionGen' / 'image_val' / f"{target_name}.jpg"
                # print(target_image_path)
                target_image = self.preprocess(PIL.Image.open(target_image_path))
                # img2 = toPIL(target_image)
                # img2.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test2.png')

                return reference_image, target_image, image_captions

    def __len__(self):

        return len(self.dataset)


class FashionGenDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, preprocess: callable, usev=False, mode='pretrain'):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.split = split
        self.usev = usev
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")

        self.preprocess = preprocess

        self.mode = mode

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        if self.mode == 'pretrain':
            if split == "val":
                with jsonlines.open(base_path / 'fashionGen' / f'{split}100_object.jsonl', mode="r") as writer:
                    dataset = [item for item in writer]
                    self.dataset = dataset
            else:
                with jsonlines.open(base_path / 'fashionGen' / f'annotation_{split}.jsonl', mode="r") as writer:
                    dataset = [item for item in writer]
                    self.dataset = dataset
        elif self.mode == 'classic':
            with jsonlines.open(base_path / 'fashionGen' / f'annotation_{split}.jsonl', mode="r") as writer:
                dataset = [item for item in writer]
                self.dataset = dataset
        else:
            with jsonlines.open(base_path / 'fashionGen' / f'annotation_{split}_object.jsonl', mode="r") as writer:
                self.dataset = [item for item in writer]

        print(f"FashionGen {split} dataset initialized")

    def __getitem__(self, index):

        if self.mode == 'pretrain':

            if self.split=='train':
                item = self.dataset[index]
                reference_name = json.loads(item['index'])
                image_captions = item['description']
                image_path = base_path / 'fashionGen' / 'image' / f"{reference_name}.jpg"
                image = self.preprocess(PIL.Image.open(image_path))
                return image, image_captions
            else:
                item = self.dataset[index]
                reference_name = json.loads(item['index'])
                image_captions = item['description']
                image_path = base_path / 'fashionGen' / 'image_val' / f"{reference_name}.jpg"
                image = self.preprocess(PIL.Image.open(image_path))
                return reference_name, image, image_captions

        elif self.mode == 'classic':
            item = self.dataset[index]
            image_name = json.loads(item['index'])
            image_path = base_path / 'fashionGen' / 'image_val' / f"{image_name}.jpg"
            image = self.preprocess(PIL.Image.open(image_path))
            return image_name, image

        else:

            if self.split == 'train':
                item = self.dataset[index]
                image_name_list = json.loads(item['index'])
                image_captions = item['description']
                if len(image_name_list) > 1:
                    ref_target = sample(image_name_list, 2)
                    reference_name = ref_target[0]
                    target_name = ref_target[1]
                else:
                    reference_name = target_name = image_name_list[0]
                reference_image_path = base_path / 'fashionGen' / 'image' / f"{reference_name}.jpg"
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                # toPIL = transforms.ToPILImage()
                # img = toPIL(reference_image)
                # img.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test1.png')

                # target_name = self.triplets[index]['target']
                target_image_path = base_path / 'fashionGen' / 'image' / f"{target_name}.jpg"
                target_image = self.preprocess(PIL.Image.open(target_image_path))
                # img = toPIL(reference_image)
                # img.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test2.png')

                return reference_image, target_image, image_captions

            elif self.split == 'val':
                item = self.dataset[index]
                image_name_list = json.loads(item['index'])
                image_captions = item['description']
                if len(image_name_list) > 1:
                    ref_target = sample(image_name_list, 2)
                    reference_name = ref_target[0]
                    target_name = ref_target[1]
                else:
                    reference_name = target_name = image_name_list[0]
                reference_image_path = base_path / 'fashionGen' / 'image_val' / f"{reference_name}.jpg"
                # print(reference_image_path)
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                # toPIL = transforms.ToPILImage()
                # img = toPIL(reference_image)
                # img.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test1.png')

                # target_name = self.triplets[index]['target']
                target_image_path = base_path / 'fashionGen' / 'image_val' / f"{target_name}.jpg"
                # print(target_image_path)
                target_image = self.preprocess(PIL.Image.open(target_image_path))
                # img2 = toPIL(target_image)
                # img2.save('/home/data2/xiangyu/Code/SPRC/fashionGen/test2.png')

                return reference_image, target_image, image_captions

    def __len__(self):

        return len(self.dataset)


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable, usev=False, llava='llava'):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.usev = usev
        # if mode not in ['relative', 'classic', 'diffusion', 'pretrain']:
        #     raise ValueError("mode should be in ['relative', 'classic', 'diffusion']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess
        self.diffusion_preprocess = diffusion_transform()

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.{llava}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        self.image_name_with_captions: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.llava_caption.{dress_type}.json') as f:
                self.image_name_with_captions.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                reference_caption = self.triplets[index]['candidate_caption']

                if self.split == 'train':

                    if self.usev:
                        target_name = self.triplets[index]['target']
                        # image_captions = generate_randomized_fiq_caption(image_captions)
                        target_caption = self.triplets[index]['target_caption']

                        return reference_name, target_name, image_captions, reference_caption, target_caption

                    else:
                        reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                        target_name = self.triplets[index]['target']
                        target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                        target_image = self.preprocess(PIL.Image.open(target_image_path))
                        image_captions = generate_randomized_fiq_caption(image_captions)
                        target_caption = self.triplets[index]['target_caption']

                        return reference_image, target_image, image_captions, reference_caption, target_caption

                elif self.split == 'val':
                    if self.usev:
                        reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                        target_name = self.triplets[index]['target']
                        target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                        target_image = self.preprocess(PIL.Image.open(target_image_path))
                        return reference_image, target_image, image_captions
                    else:
                        target_name = self.triplets[index]['target']
                        target_caption = self.triplets[index]['target_caption']
                        return reference_name, target_name, image_captions, reference_caption, target_caption

                elif self.split == 'val_train':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_captions

                elif self.split == 'test':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                if self.usev:
                    image_name = self.image_names[index]
                    image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                    image = self.preprocess(PIL.Image.open(image_path))
                    return image_name, image
                else:
                    image_name = self.image_name_with_captions[index]['image']
                    caption = self.image_name_with_captions[index]['caption']
                    image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                    image = self.preprocess(PIL.Image.open(image_path))
                    return image_name, image, caption

            elif self.mode == 'pretrain_ref':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                reference_caption = self.triplets[index]['candidate_caption']

                reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                return reference_image, reference_caption

            elif self.mode == 'pretrain_target':
                image_captions = self.triplets[index]['captions']
                target_name = self.triplets[index]['target']
                target_caption = self.triplets[index]['target_caption']

                target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                target_image = self.preprocess(PIL.Image.open(target_image_path))
                return target_image, target_caption

            elif self.mode == 'diffusion':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                reference_caption = self.triplets[index]['candidate_caption']

                if self.split == 'train':

                    if self.usev:

                        target_name = self.triplets[index]['target']
                        # image_captions = generate_randomized_fiq_caption(image_captions)
                        target_caption = self.triplets[index]['target_caption']

                        return reference_name, target_name, image_captions, reference_caption, target_caption

                    else:
                        reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                        target_name = self.triplets[index]['target']
                        target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                        target_image = self.preprocess(PIL.Image.open(target_image_path))
                        image_captions = generate_randomized_fiq_caption(image_captions)
                        target_caption = self.triplets[index]['target_caption']

                        target_diffusion = self.diffusion_preprocess(PIL.Image.open(target_image_path))

                        return reference_image, target_image, image_captions, reference_caption, target_caption, \
                            target_diffusion

                elif self.split == 'val':
                    if self.usev:
                        reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                        target_name = self.triplets[index]['target']
                        target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                        target_image = self.preprocess(PIL.Image.open(target_image_path))
                        return reference_image, target_image, image_captions
                    else:
                        target_name = self.triplets[index]['target']
                        target_caption = self.triplets[index]['target_caption']
                        return reference_name, target_name, image_captions, reference_caption, target_caption

                elif self.split == 'val_train':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_captions

                elif self.split == 'test':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative' or self.mode == 'diffusion' or self.mode == 'pretrain':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_name_with_captions)
        else:
            return len(self.triplets)


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
