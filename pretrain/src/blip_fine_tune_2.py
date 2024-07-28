# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR
import accelerate

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset, \
    FashionGenDataset, Fashion200KDataset
from utils import collate_fn, update_train_running_results, update_train_running_results_dict, \
    set_train_bar_description_dict, set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics, compute_fgen_val_metrics

from rq_vae_transformer.rqvae.models import create_model
from rq_vae_transformer.rqvae.utils.setup import setup

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

from transformers import LlamaTokenizer
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def pretrain(train_dress_types: List[str], val_dress_types: List[str],
             num_epochs: int, blip_model_name: str, learning_rate: float, batch_size: int,
             validation_frequency: int, transform: str, save_training: bool, save_best: bool,
             **kwargs):
    # config, logger, writer = setup(args)
    # rqvae_model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    # rqvae_model = rqvae_model.to(device)

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/Pretrain_retrieval_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type="pretrain",
                                                                           is_eval=False, device=device)
    model_path = '/home/data2/xiangyu/Code/SPRC/models/only_code2code_blip2_cir_cls_rqvae8_2024-04-19_21:28:44' \
                 '/saved_models/tuned_clip_best2.pt'
    model_path = None
    if model_path:
        checkpoint_path = model_path

        checkpoint = torch.load(checkpoint_path, map_location=device)
        msg = blip_model.load_state_dict(checkpoint['Blip2QformerCirCls'], strict=False)
        # blip_model.to(dtype=torch.bfloat16)
        print("Missing keys {}".format(msg.missing_keys))

    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, llava='process')
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)

    fashion_val_classic = FashionGenDataset('val', preprocess, mode='classic')
    fashion_val = FashionGenDataset('val', preprocess)

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'pretrain_ref', preprocess, llava='process') + \
                             FashionIQDataset('train', train_dress_types, 'pretrain_target', preprocess,
                                              llava='process') + \
                             FashionIQDataset('val', train_dress_types, 'pretrain_ref', preprocess, llava='process') + \
                             FashionIQDataset('val', train_dress_types, 'pretrain_target', preprocess, llava='process')
    gen_train_dataset = FashionGenDataset('train', preprocess)
    fashion200k_train_dataset = Fashion200KDataset('train', preprocess)
    print("===========================================")
    print(relative_train_dataset.__len__())
    print(gen_train_dataset.__len__())
    print(fashion200k_train_dataset.__len__())
    relative_train_dataset = relative_train_dataset
    # relative_train_dataset = gen_train_dataset
    print(relative_train_dataset.__len__())
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    accelerator = accelerate.Accelerator()
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          #   'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay':0.05}])
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}])
    # scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5 / num_epochs, div_factor=100.,
                           steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    blip_model, optimizer, relative_train_loader = accelerator.prepare(blip_model, optimizer,
                                                                       relative_train_loader)

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (image, captions) in enumerate(train_bar):
            images_in_batch = image.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            image = image.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]

            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image": image, "text_input": captions})
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # Backpropagate and update the weights
            accelerator.backward(loss.mean())
            optimizer.step()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
             pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []
            recalls_at10_rq = []
            recalls_at50_rq = []
            recalls_at1 = []
            recalls_at5 = []
            recalls_at1_rq = []
            recalls_at5_rq = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            # index_features, index_names = extract_index_blip_features(fashion_val_classic, blip_model)
            recall_at10, recall_at50, recall_at10_rq, recall_at50_rq, recall_at1, recall_at5, \
                recall_at1_rq, recall_at5_rq = compute_fgen_val_metrics(fashion_val, blip_model, txt_processors)

            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)
            recalls_at10_rq.append(recall_at10_rq)
            recalls_at50_rq.append(recall_at50_rq)
            recalls_at1.append(recall_at1)
            recalls_at5.append(recall_at5)
            recalls_at1_rq.append(recall_at1_rq)
            recalls_at5_rq.append(recall_at5_rq)
            torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10_rq'] = recalls_at10_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50_rq'] = recalls_at50_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at1'] = recalls_at1[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at5'] = recalls_at5[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at1_rq'] = recalls_at1_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at5_rq'] = recalls_at5_rq[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall_at10_rq': mean(recalls_at10_rq),
                f'average_recall_at50_rq': mean(recalls_at50_rq),
                f'average_recall_at1': mean(recalls_at1),
                f'average_recall_at5': mean(recalls_at5),
                f'average_recall_at1_rq': mean(recalls_at1_rq),
                f'average_recall_at5_rq': mean(recalls_at5_rq),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2,
                f'average_recall_rq': (mean(recalls_at50_rq) + mean(recalls_at10_rq)) / 2,
                f'average_recall_10': (mean(recalls_at5) + mean(recalls_at1) + + mean(recalls_at10)) / 3,
                f'average_recall_rq_10': (mean(recalls_at5_rq) + mean(recalls_at1_rq) + mean(recalls_at10_rq)) / 3
            })

            print(json.dumps(results_dict, indent=4))

            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    print('Saving better checkpoints')
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)
            model_name = 'tuned_clip_best' + str(epoch)
            save_model(model_name, epoch, blip_model, training_path)


def discriminator(train_dress_types: List[str], val_dress_types: List[str],
                  num_epochs: int, blip_model_name: str, learning_rate: float, batch_size: int,
                  validation_frequency: int, transform: str, save_training: bool, save_best: bool,
                  **kwargs):
    # config, logger, writer = setup(args)
    # rqvae_model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    # rqvae_model = rqvae_model.to(device)

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/Pretrain_retrieval_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type="stage2",
                                                                           is_eval=False, device=device, )

    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, llava='process')
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)

    fashion_val_classic = FashionGenDataset('val', preprocess, mode='classic')
    fashion_val = FashionGenDataset('val', preprocess)

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'pretrain_ref', preprocess, llava='process') + \
                             FashionIQDataset('train', train_dress_types, 'pretrain_target', preprocess,
                                              llava='process') + \
                             FashionIQDataset('val', train_dress_types, 'pretrain_ref', preprocess, llava='process') + \
                             FashionIQDataset('val', train_dress_types, 'pretrain_target', preprocess, llava='process')
    gen_train_dataset = FashionGenDataset('train', preprocess)
    fashion200k_train_dataset = Fashion200KDataset('train', preprocess)
    print("===========================================")
    print(relative_train_dataset.__len__())
    print(gen_train_dataset.__len__())
    print(fashion200k_train_dataset.__len__())
    relative_train_dataset = gen_train_dataset+relative_train_dataset
    print(relative_train_dataset.__len__())
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    accelerator = accelerate.Accelerator()
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          #   'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay':0.05}])
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}])
    # scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5 / num_epochs, div_factor=100.,
                           steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    blip_model, optimizer, relative_train_loader = accelerator.prepare(blip_model, optimizer,
                                                                       relative_train_loader)

    # model_path = "/home/data2/xiangyu/Code/SPRC/models/Pretrain_retrieval_blip2_cir_cls/saved_models/tuned_clip_best.pt"
    model_path = None
    if model_path:
        checkpoint_path = model_path

        checkpoint = torch.load(checkpoint_path, map_location=device)
        msg = blip_model.load_state_dict(checkpoint['Blip2QformerCirCls'], strict=False)
        # blip_model.to(dtype=torch.bfloat16)
        print("Missing keys {}".format(msg.missing_keys))

    # When save_best == True initialize the best result to zero
    print("Frozen parameters:")
    for name, param in blip_model.named_parameters():
        if not param.requires_grad:
            print(name)
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []
            recalls_at10_rq = []
            recalls_at50_rq = []
            recalls_at1 = []
            recalls_at5 = []
            recalls_at1_rq = []
            recalls_at5_rq = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            # index_features, index_names = extract_index_blip_features(fashion_val_classic, blip_model)
            recall_at10, recall_at50, recall_at10_rq, recall_at50_rq, recall_at1, recall_at5, \
                recall_at1_rq, recall_at5_rq = compute_fgen_val_metrics(fashion_val, blip_model, txt_processors)

            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)
            recalls_at10_rq.append(recall_at10_rq)
            recalls_at50_rq.append(recall_at50_rq)
            recalls_at1.append(recall_at1)
            recalls_at5.append(recall_at5)
            recalls_at1_rq.append(recall_at1_rq)
            recalls_at5_rq.append(recall_at5_rq)
            torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10_rq'] = recalls_at10_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50_rq'] = recalls_at50_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at1'] = recalls_at1[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at5'] = recalls_at5[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at1_rq'] = recalls_at1_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at5_rq'] = recalls_at5_rq[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall_at10_rq': mean(recalls_at10_rq),
                f'average_recall_at50_rq': mean(recalls_at50_rq),
                f'average_recall_at1': mean(recalls_at1),
                f'average_recall_at5': mean(recalls_at5),
                f'average_recall_at1_rq': mean(recalls_at1_rq),
                f'average_recall_at5_rq': mean(recalls_at5_rq),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2,
                f'average_recall_rq': (mean(recalls_at50_rq) + mean(recalls_at10_rq)) / 2,
                f'average_recall_10': (mean(recalls_at5) + mean(recalls_at1) + + mean(recalls_at10)) / 3,
                f'average_recall_rq_10': (mean(recalls_at5_rq) + mean(recalls_at1_rq) + mean(recalls_at10_rq)) / 3
            })

            print(json.dumps(results_dict, indent=4))
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (image, captions) in enumerate(train_bar):
            images_in_batch = image.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            image = image.to(device, non_blocking=True)
            captions = captions

            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image": image, "text_input": captions})
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # Backpropagate and update the weights
            accelerator.backward(loss.mean())
            optimizer.step()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
             pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []
            recalls_at10_rq = []
            recalls_at50_rq = []
            recalls_at1 = []
            recalls_at5 = []
            recalls_at1_rq = []
            recalls_at5_rq = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            # index_features, index_names = extract_index_blip_features(fashion_val_classic, blip_model)
            recall_at10, recall_at50, recall_at10_rq, recall_at50_rq, recall_at1, recall_at5, \
                recall_at1_rq, recall_at5_rq = compute_fgen_val_metrics(fashion_val, blip_model, txt_processors)

            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)
            recalls_at10_rq.append(recall_at10_rq)
            recalls_at50_rq.append(recall_at50_rq)
            recalls_at1.append(recall_at1)
            recalls_at5.append(recall_at5)
            recalls_at1_rq.append(recall_at1_rq)
            recalls_at5_rq.append(recall_at5_rq)
            torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10_rq'] = recalls_at10_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50_rq'] = recalls_at50_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at1'] = recalls_at1[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at5'] = recalls_at5[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at1_rq'] = recalls_at1_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at5_rq'] = recalls_at5_rq[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall_at10_rq': mean(recalls_at10_rq),
                f'average_recall_at50_rq': mean(recalls_at50_rq),
                f'average_recall_at1': mean(recalls_at1),
                f'average_recall_at5': mean(recalls_at5),
                f'average_recall_at1_rq': mean(recalls_at1_rq),
                f'average_recall_at5_rq': mean(recalls_at5_rq),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2,
                f'average_recall_rq': (mean(recalls_at50_rq) + mean(recalls_at10_rq)) / 2,
                f'average_recall_10': (mean(recalls_at5) + mean(recalls_at1) + + mean(recalls_at10)) / 3,
                f'average_recall_rq_10': (mean(recalls_at5_rq) + mean(recalls_at1_rq) + mean(recalls_at10_rq)) / 3
            })

            print(json.dumps(results_dict, indent=4))

            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                # if save_best and results_dict['average_recall'] > best_avg_recall:
                print('Saving better checkpoints')
                best_avg_recall = results_dict['average_recall']
                    # save_model('tuned_clip_best', epoch, blip_model, training_path)
                models_path = training_path / "saved_models"
                models_path.mkdir(exist_ok=True, parents=True)
                lora_bias = "none"

                state_dict = get_peft_state_maybe_zero_3(
                    blip_model.named_parameters(), lora_bias
                )

                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    blip_model.named_parameters()
                )

                print(list(non_lora_state_dict.keys()))
                name = 'none_lora'
                blip_model.get_llm().save_pretrained(training_path, state_dict=state_dict)
                torch.save(non_lora_state_dict, str(models_path / f'{name}_{epoch}.pt'))
            # model_name = 'tuned_clip_best' + str(epoch)
            # save_model(model_name, epoch, blip_model, training_path)


def rqvae_finetune(train_dress_types: List[str], val_dress_types: List[str],
                   num_epochs: int, blip_model_name: str, learning_rate: float, batch_size: int,
                   validation_frequency: int, transform: str, save_training: bool, save_best: bool,
                   **kwargs):
    # config, logger, writer = setup(args)
    # rqvae_model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    # rqvae_model = rqvae_model.to(device)

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/only_code2code_{blip_model_name}_rqvae8_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type="pretrain",
                                                                           is_eval=False, device=device)
    model_path = "/home/data2/xiangyu/Code/SPRC/models/only_code2code_blip2_cir_rerank_rqvae8_2024-06-11_17:21:29" \
                 "/saved_models/tuned_clip_best.pt"
    # model_path = None
    if model_path:
        checkpoint_path = model_path

        checkpoint = torch.load(checkpoint_path, map_location=device)
        msg = blip_model.load_state_dict(checkpoint['Blip2QformerCirRerank'], strict=False)
        # blip_model.to(dtype=torch.bfloat16)
        print("Missing keys {}".format(msg.missing_keys))

    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, llava='llava_blip')
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess, llava='llava_blip')
    gen_train_dataset = FashionGenDataset('train', preprocess) + FashionGenDataset('val', preprocess)
    # fashion200k_train_dataset = Fashion200KDataset('train', preprocess) + Fashion200KDataset('val', preprocess)
    print("===========================================")
    print(relative_train_dataset.__len__())
    print(gen_train_dataset.__len__())
    # print(fashion200k_train_dataset.__len__())
    # relative_train_dataset = gen_train_dataset + relative_train_dataset
    relative_train_dataset = relative_train_dataset
    print(relative_train_dataset.__len__())
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=False)

    # Define the optimizer, the loss and the grad scaler
    accelerator = accelerate.Accelerator()
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          #   'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay':0.05}])
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}])
    # scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5 / num_epochs, div_factor=100.,
                           steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    blip_model, optimizer, relative_train_loader = accelerator.prepare(blip_model, optimizer,
                                                                       relative_train_loader)

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []
            recalls_at10_rq = []
            recalls_at50_rq = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                      idx_to_dress_mapping):
                index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model)
                recall_at10, recall_at50, recall_at10_rq, recall_at50_rq = compute_fiq_val_metrics(relative_val_dataset,
                                                                                                   blip_model,
                                                                                                   index_features,
                                                                                                   index_names,
                                                                                                   txt_processors)

                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)
                recalls_at10_rq.append(recall_at10_rq)
                recalls_at50_rq.append(recall_at50_rq)
                torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10_rq'] = recalls_at10_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50_rq'] = recalls_at50_rq[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall_at10_rq': mean(recalls_at10_rq),
                f'average_recall_at50_rq': mean(recalls_at50_rq),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2,
                f'average_recall_rq': (mean(recalls_at50_rq) + mean(recalls_at10_rq)) / 2
            })

            print(json.dumps(results_dict, indent=4))
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (
                reference_images, target_images, captions, reference_caption, target_caption) in enumerate(
            train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
            # flattened_captions: list = np.array(captions).T.flatten().tolist()
            # captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors["eval"](caption) for caption in captions]
            reference_caption = [txt_processors["eval"](t) for t in reference_caption]
            target_caption = [txt_processors["eval"](t) for t in target_caption]
            # print(codes)
            # codes = [txt_processors["eval"](code) for code in codes]
            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image": reference_images, "target": target_images, "text_input": captions,
                                        "reference_caption": reference_caption, "target_caption": target_caption})
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # Backpropagate and update the weights
            accelerator.backward(loss.mean())
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
             pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []
            recalls_at10_rq = []
            recalls_at50_rq = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                      idx_to_dress_mapping):
                index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model)
                recall_at10, recall_at50, recall_at10_rq, recall_at50_rq = compute_fiq_val_metrics(relative_val_dataset,
                                                                                                   blip_model,
                                                                                                   index_features,
                                                                                                   index_names,
                                                                                                   txt_processors)

                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)
                recalls_at10_rq.append(recall_at10_rq)
                recalls_at50_rq.append(recall_at50_rq)
                torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10_rq'] = recalls_at10_rq[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50_rq'] = recalls_at50_rq[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall_at10_rq': mean(recalls_at10_rq),
                f'average_recall_at50_rq': mean(recalls_at50_rq),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2,
                f'average_recall_rq': (mean(recalls_at50_rq) + mean(recalls_at10_rq)) / 2
            })

            print(json.dumps(results_dict, indent=4))

            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    print('Saving better checkpoints')
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)
            model_name = 'tuned_clip_best' + str(epoch)
            save_model(model_name, epoch, blip_model, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model-config', type=str, default='./configs/c10-igpt.yaml')
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--data-path", type=str, default="./cirr_dataset")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str, help="[blip2_cir_cat, blip2_cir]")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--loss-align", default=0.6, type=float)
    parser.add_argument("--loss-rtc", default=0.6, type=float)
    parser.add_argument("--loss-itm", default=1, type=float)
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument('-r', '--result-path', type=str, default='./results.tmp')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('-p', '--postfix', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "data_path": args.data_path,
        "loss_rtc": args.loss_rtc,
        "loss_align": args.loss_align,
        "loss_itm": args.loss_itm
    }

    if args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        pretrain(**training_hyper_params)

    if args.dataset.lower() == 'discriminator':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        discriminator(**training_hyper_params)

    if args.dataset.lower() == 'cir':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        rqvae_finetune(**training_hyper_params)
