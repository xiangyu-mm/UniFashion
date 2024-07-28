"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from torch.nn import functional as F

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTokenizer
from transformers.activations import QuickGELUActivation as QuickGELU
from lavis.models.blip_diffusion_models.modeling_ctx_clip import CtxCLIPTextModel
from lavis.models.blip_diffusion_models.ptp_utils import (
    LocalBlend,
    P2PCrossAttnProcessor,
    AttentionRefine,
)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def get_rand_des():
    text = ['Describe the image concisely.',
            'Provide a brief description of the given image.',
            'Offer a succinct explanation of the picture presented.',
            'Can you describe this image briefly?',
            'Summarize the visual content of the image.',
            'Give a short and clear explanation of the subsequent image.',
            'Share a concise interpretation of the image provided.',
            'Present a compact description of the photo’s key features.',
            'Relay a brief, clear account of the picture shown.',
            'Render a clear and concise summary of the photo.',
            'Write a terse but informative summary of the picture.',
            'Create a compact narrative representing the image presented.']

    return text[random.randint(0, 11)]


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x


@registry.register_model("blip2_cir_full")
class Blip2QformerCirFull(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "stage2": "configs/models/blip2/blip2_stage2.yaml",
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            llm_model="lmsys/vicuna-7b-v1.1",
            prompt="",
            max_txt_len=256,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            qformer_text_input=True,
            embed_dim=256,
            cross_attention_freq=2,
            lora=False,
            sd_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            vae_half_precision=True,
            qformer_train=False,
            proj_train=True,
            use_diffusion=False,
            use_llm=True
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.lora = lora
        self.use_diffusion = use_diffusion
        self.use_llm = use_llm

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            print("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )

        # projection layer
        self.proj_layer = ProjLayer(
            in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
        )

        self.qformer_train = qformer_train

        self.llm_query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, 4096)
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        if self.use_llm:
            model_path = "/home/data2/xiangyu/llava/LLaVA/checkpoints/llava-v1.5-13B"

            self.llm_tokenizer, self.llm_model, self.llm_image_processor, self.llm_context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path)
            )

            if self.lora:
                # https://github.com/lxe/llama-peft-tuner/blob/main/finetune_peft.py
                self.llm_model = LlamaForCausalLM.from_pretrained(
                    llm_model,
                    # load_in_8bit=True,
                    torch_dtype=torch.float16
                )
                self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=128,
                    lora_alpha=256, lora_dropout=0.05,
                    target_modules=find_all_linear_names(self.llm_model)
                )
                self.llm_model.gradient_checkpointing_enable()
                self.llm_model.enable_input_require_grads()
                # self.llm_model.lm_head = CastOutputToFloat(self.llm_model.lm_head)
                self.llm_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
                self.llm_model.print_trainable_parameters()
                # self.lora_model = lora_model
            else:
                self.llm_model = LlamaForCausalLM.from_pretrained(
                    llm_model, torch_dtype=torch.float16
                )

                self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

            self.llm_proj = nn.Linear(
                self.Qformer.config.hidden_size, 4096
            )

            self.freeze_modules()

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.proj_train = proj_train

        # stable diffusion
        if self.use_diffusion:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(
                sd_pretrained_model_name_or_path, subfolder="tokenizer"
            )
            self.text_encoder = CtxCLIPTextModel.from_pretrained(
                sd_pretrained_model_name_or_path, subfolder="text_encoder"
            )
            self.vae = AutoencoderKL.from_pretrained(
                sd_pretrained_model_name_or_path, subfolder="vae"
            )
            if vae_half_precision:
                self.vae.half()

            self.unet = UNet2DConditionModel.from_pretrained(
                sd_pretrained_model_name_or_path, subfolder="unet"
            )
            # self.unet.enable_xformers_memory_efficient_attention()

            self.noise_scheduler = DDPMScheduler.from_config(
                sd_pretrained_model_name_or_path, subfolder="scheduler"
            )

            self.qformer_train = qformer_train

            self.freeze_modules()

            self.ctx_embeddings_cache = nn.Parameter(
                torch.zeros(1, num_query_token, 768), requires_grad=False
            )
            self._use_embeddings_cache = False

            # inference-related
            self._CTX_BEGIN_POS = 2

    def get_llm(self):
        return self.llm_model

    def freeze_modules(self):
        if self.use_diffusion:
            to_freeze = [self.vae, self.vision_proj, self.text_proj, self.itm_head, self.ln_vision, self.text_encoder]
        else:
            to_freeze = [self.vision_proj, self.text_proj, self.itm_head, self.ln_vision]

        if not self.qformer_train:
            to_freeze.append(self.Qformer)

        if not self.proj_train:
            to_freeze.append(self.proj_layer)

        for module in to_freeze:
            module.eval()
            module.train = self.disabled_train
            module.requires_grad_(False)

    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    @property
    def pndm_scheduler(self):
        if not hasattr(self, "_pndm_scheduler"):
            self._pndm_scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                set_alpha_to_one=False,
                skip_prk_steps=True,
            )
        return self._pndm_scheduler

    @property
    def ddim_scheduler(self):
        if not hasattr(self, "_ddim_scheduler"):
            self._ddim_scheduler = DDIMScheduler.from_config(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )
        return self._ddim_scheduler

    def forward(self, samples):

        use_diffuer = self.use_diffusion
        use_llm = self.use_llm

        image = samples["image"]
        text = samples["text_input"]
        target = samples["target"]
        reference_caption = samples["reference_caption"]
        target_caption = samples["target_caption"]

        ###============== reference text fusion ===================###
        # reference image feature
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        self.tokenizer.truncation_side = 'right'

        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(query_output.last_hidden_state[:, 32, :]), dim=-1
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :]), dim=-1
        )

        text_only_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=text_Qformer.attention_mask,
            return_dict=True,
        )

        text_only_feat = F.normalize(
            self.text_proj(text_only_output.last_hidden_state[:, 0, :]), dim=-1
        )

        target_output = self.Qformer.bert(
            query_embeds=prompt_tokens,
            encoder_hidden_states=target_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )

        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )

        ###============== Fusion-target Contrastive ===================###
        # reference image feature

        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_q2t = torch.matmul(
            target_feats.unsqueeze(1), fusion_feats.unsqueeze(-1)
        ).squeeze()

        bs = image.size(0)
        sim_t2i, _ = sim_t2q.max(-1)
        sim_i2t, _ = sim_q2t.max(-1)

        sim_t2i = sim_t2i / self.temp
        sim_i2t = sim_i2t / self.temp

        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(
            image.device
        )

        loss_itc = F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)

        # image_feats = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )
        if use_llm:
            inputs_llm = self.llm_proj(query_output.last_hidden_state)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'
            instruct = [get_rand_des()]*len(text)
            text_input_tokens = self.llm_tokenizer(
                instruct,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            self.llm_tokenizer.truncation_side = 'right'
            text_output_tokens = self.llm_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(image.device)

            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )

            # do not apply loss to the padding
            targets_llm = llm_tokens['input_ids'].masked_fill(
                llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
            )

            # do not apply loss to the text input (i.e., instruction)
            for i, l in enumerate(input_part_targets_len):
                targets_llm[i][:l] = -100

            # do not apply loss to the query tokens
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
            )

            targets_llm = torch.cat([empty_targets, targets_llm], dim=1)
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

            with self.maybe_autocast():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                    labels=targets_llm,
                )
            # outputs.output_hidden_states[-1]: bs * seq_len * 4096

            loss_lm = outputs.loss
        # loss_diffusion = loss_itc
        if use_diffuer:
            latents = self.vae.encode(samples["image"].half()).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            # ctx_embeddings = self.forward_ctx_embeddings(
            #     input_image=samples["inp_image"], text_input=samples["subject_text"]
            # )
            blip_embeddings = query_output.last_hidden_state[:, : query_tokens.size(1), :]
            ref_feats = self.proj_layer(blip_embeddings)

            ctx_embeddings = ref_feats

            # Get the text embedding for conditioning
            input_ids = self.clip_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.clip_tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(image.device)
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids,
                ctx_embeddings=ctx_embeddings,
                ctx_begin_pos=[self._CTX_BEGIN_POS] * input_ids.shape[0],
            )[0]

            # Predict the noise residual
            noise_pred = self.unet(
                noisy_latents.float(), timesteps, encoder_hidden_states
            ).sample

            loss_diffusion = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        if use_diffuer:
            return {'loss_lm': loss_lm, 'diffusion_loss': loss_diffusion}
        elif use_llm:
            return {'loss_lm': loss_lm}
        else:
            return {'loss_itc': loss_itc}

    @torch.no_grad()
    def extract_for_diff(self, text):
        text = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self.device
        )

        text_output = self.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        text_features = self.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        return text_embeds

    def forward_ctx_embeddings(self, input_image, text_input, ratio=None):
        def compute_ctx_embeddings(input_image, text_input):
            # blip_embeddings = self.blip(image=input_image, text=text_input)
            blip_embeddings = self.blip.extract_features(
                {"image": input_image, "text_input": text_input}, mode="multimodal"
            ).multimodal_embeds
            ctx_embeddings = self.proj_layer(blip_embeddings)

            return ctx_embeddings

        if isinstance(text_input, str):
            text_input = [text_input]

        if self._use_embeddings_cache:
            # expand to batch size
            ctx_embeddings = self.ctx_embeddings_cache.expand(len(text_input), -1, -1)
        else:
            if isinstance(text_input[0], str):
                text_input, input_image = [text_input], [input_image]

            all_ctx_embeddings = []

            for inp_image, inp_text in zip(input_image, text_input):
                ctx_embeddings = compute_ctx_embeddings(inp_image, inp_text)
                all_ctx_embeddings.append(ctx_embeddings)

            if ratio is not None:
                assert len(ratio) == len(all_ctx_embeddings)
                assert sum(ratio) == 1
            else:
                ratio = [1 / len(all_ctx_embeddings)] * len(all_ctx_embeddings)

            ctx_embeddings = torch.zeros_like(all_ctx_embeddings[0])

            for ratio, ctx_embeddings_ in zip(ratio, all_ctx_embeddings):
                ctx_embeddings += ratio * ctx_embeddings_

        return ctx_embeddings

    @torch.no_grad()
    def inference(self, images, text):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(images))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)

        text_only_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_only_feat = F.normalize(
            self.text_proj(text_only_output.last_hidden_state[:, 0, :]), dim=-1
        )

        sim_t2q = torch.matmul(
            text_only_feat.unsqueeze(1).unsqueeze(1), image_features.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)

        sim_q2t = torch.matmul(
            image_features.unsqueeze(1), text_only_feat.unsqueeze(-1)
        ).squeeze()

        sim_t2i, _ = sim_q2t.max(-1)

        return sim_i2t, sim_t2i

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                    image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                    caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            num_captions=1,
            temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        image = samples["image"]

        bs = image.size(0)

        prompt = get_rand_des()

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        query_tokens = self.query_tokens.expand(bs, -1, -1)

        # For video data

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
            self.llm_model.config.output_scores = True
            self.llm_model.config.return_dict_in_generate = True
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        sequences_scores = outputs.sequences_scores
        outputs = outputs.sequences
        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text, sequences_scores

    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in
                                        enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id,
                                                              -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            cross_attention_freq=cross_attention_freq,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model
