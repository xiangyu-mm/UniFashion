import json
from pathlib import Path

stage="ciyun"

if stage=='generation_caption':
    dress_types=['dress', 'shirt', 'toptee']
    base_path = Path(__file__).absolute().parents[0].absolute()
    for dress_type in dress_types:
        image_names = []
        triplets = []
        new_image_names = []
        doubles = []
        no_list = []
        all_list=[]
        with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.val.json') as f:
            image_names.extend(json.load(f))
        with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.val.llava.json') as f:
            triplets.extend(json.load(f))
        for item in triplets:
            a = item['target']
            b = item['candidate']
            if a not in new_image_names:
                new_image_names.append(a)
                doubles.append({"image": a, "caption": item["target_caption"]})
            if b not in new_image_names:
                new_image_names.append(b)
                doubles.append({"image": b, "caption": item["candidate_caption"]})
        for item in image_names:
            name = item['image']
            if name in new_image_names:
                for dict_item in doubles:
                    if name == dict_item['image']:
                        caption = dict_item['caption']
                        break
                all_list.append({"image": name, "caption": caption})
            else:
                all_list.append({"image": name, "caption": item["caption"]})

        json_file_path = base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.llava_caption_all.{dress_type}.json'
        json_file = open(json_file_path, mode='w')
        json.dump(all_list, json_file, indent=4)

        print(len(doubles))
        print(len(image_names))
if stage == "merge":
    dress_types = ['dress', 'shirt', 'toptee']
    base_path = Path(__file__).absolute().parents[0].absolute()
    for dress_type in dress_types:
        image_names = []
        triplets = []
        new_image_names = []
        doubles = []
        no_list = []
        with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.llava_caption.{dress_type}.json') as f:
            image_names.extend(json.load(f))
        with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'cap.{dress_type}.llava1.5_4_no.json') as f:
            triplets.extend(json.load(f))
        with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.no_caption.{dress_type}.json') as f:
            no_list.extend(json.load(f))
        for item in no_list:
            # result = [item[key] for item in triplets]
            for dd in triplets:
                if item == dd['image']:
                    doubles.append({"image": item, "caption": dd["caption"]})
                    break
        print(len(image_names))
        print(len(doubles))
        c=image_names+doubles

        json_file_path = base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.all_caption.{dress_type}.json'
        json_file = open(json_file_path, mode='w')
        json.dump(c, json_file, indent=4)

        print(len(c))

if stage=='llava':
    dress_types=['dress', 'shirt', 'toptee']
    base_path = Path(__file__).absolute().parents[0].absolute()
    for dress_type in dress_types:
        image_names = []
        triplets = []
        new_image_names = []
        doubles = []
        no_list = []
        all_list=[]
        with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'cap.{dress_type}.llava_no_person.all.json') as f:
            image_names.extend(json.load(f))
        with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.val.llava.json') as f:
            triplets.extend(json.load(f))
        # for item in triplets:
        #     a = item['target']
        #     if a not in new_image_names:
        #         new_image_names.append(a)
        #         doubles.append({"image": a, "caption": item["captions"]})
        for item in image_names:
            # if item in new_image_names:
            #     for dict_item in doubles:
            #         if item == dict_item['image']:
            #             caption = dict_item['caption']
            #             break
            #     all_list.append({"image": item, "caption": caption})
            # else:
            all_list.append({"image": item['image']['image'], "caption": item['caption']})

        json_file_path = base_path / 'fashionIQ_dataset' / f'split.llava_caption.{dress_type}.json'
        json_file = open(json_file_path, mode='w')
        json.dump(all_list, json_file, indent=4)

        print(len(image_names))

if stage=='ciyun':
    dress_types=['dress', 'shirt', 'toptee']
    splits=['train','val']
    base_path = Path(__file__).absolute().parents[0].absolute()
    for dress_type in dress_types:
        image_names = []
        triplets = []
        new_image_names = []
        doubles = []
        no_list = []
        all_list=[]
        for split in splits:
            with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                triplets.extend(json.load(f))
            for item in triplets:
                a = item['captions']
                captions = ', '.join(a).replace('is', '').replace('more','').replace(" a ", '').replace('has','')
                all_list.append(captions)

        json_file_path = base_path / 'fashionIQ_dataset' / 'image_splits' / f'ciyun.{dress_type}.json'
        json_file = open(json_file_path, mode='w')
        json.dump(all_list, json_file, indent=4)

