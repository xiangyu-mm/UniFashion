# UniFashion
The official code for paper "UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation"

# Abstract
The fashion domain encompasses a variety of real-world multimodal tasks, including multimodal retrieval and multimodal generation. The rapid advancements in artificial intelligence generated content, particularly in technologies like large language models for text generation and diffusion models for visual generation, have sparked widespread research interest in applying these multimodal models in the fashion domain. However, tasks involving embeddings, such as image-to-text or text-to-image retrieval, have been largely overlooked from this perspective due to the diverse nature of the multimodal fashion domain. And current research on multi-task single models lack focus on image generation. In this work, we present UniFashion, a unified framework that  simultaneously tackles the challenges of multimodal generation and retrieval tasks within the fashion domain, integrating image generation with retrieval tasks and text generation tasks. \modelname{} unifies embedding and generative tasks by integrating a diffusion model and LLM, enabling controllable and high-fidelity generation. Our model significantly outperforms previous single-task state-of-the-art models across diverse fashion tasks, and can be readily adapted to manage complex vision-language tasks. This work demonstrates the potential learning synergy between multimodal generation and retrieval, offering a promising direction for future research in the fashion domain.


# Data preparation

1. Download data from [FashionGen](https://pan.baidu.com/s/1amJvPYeRXYP-uKv8Cl1fyQ)(code: pz7a), [Fashion200K](https://github.com/xthan/fashion-200k) and [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)
2. Processing the FashionIQ dataset to Fashion-IQ-Cap:
   ```Shell
   cd src/llava/LLaVA
   python inference.py
   ```
We have uploaded the processed Fashion-IQ-Cap by LLaVA v1.6 on dataset directory, you can just download it!

# Training

1. Phase 1 - Cross-modal Pre-training

   b. run this command:

   ```Shell
    bash pretrain.sh
   ```

2. Phase 2 - Composed Multimodal Fine-tuning
   run this command:
   ```Shell
    bash cir_ft.sh
   ```
3. MGD finetuning:
    run this command:
   ```Shell
    cd src/UNIStableVITON
    bash train.sh
   ```

# Vaildation

1. FashionGen dataset for cross-modal retrieval tasks:
   During the training process, we vaild evey epoch and save the result in a csv file. Or you can just vaild the saved checkpoint by run this command:
   ```Shell
    bash vail.sh
   ```
   
   ```
   CUDA_VISIBLE_DEVICES=0 python src/blip_validate.py \
   --dataset 'fashiongen' \
   --blip-model-name 'blip2_cir_cls' \
   --model-path 
   ```
3. Image captioning task performance:
   run this command:
   ```Shell
    cd src
    python metrics.py
   ```
4. on the Fashion-IQ dataset for composed image retrieval task:
   run this command:
   ```Shell
    bash vail.sh
   ```
   ```
   CUDA_VISIBLE_DEVICES=0 python src/blip_validate.py \
   --dataset 'fashioniq' \
   --blip-model-name 'blip2_cir_rerank' \
   --model-path 
   ```
5. VITON-HD and MGD datasets for try-on task:
   run this command:
   ```Shell
    cd src/UNIStableVITON
    bash inference.sh
   ```
