import argparse
import deepspeed
import json
from safetensors import safe_open
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn.functional as F
from safetensors import safe_open

from eagle.model.cnets import Model
from eagle.model.configs import EConfig, Qwen2VLConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
import random


def get_embed(path):
    try:
        with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            embed_path = index_json["weight_map"]["model.embed_tokens.weight"]
        with safe_open(os.path.join(path, embed_path),
                    framework="pt",
                    device="cpu") as f:
            tensor_slice = f.get_slice("model.embed_tokens.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        try:
            with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            weights = torch.load(os.path.join(path, emb_path))
            tensor = weights["model.embed_tokens.weight"].float()
        except:
            with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["transformer.wte.weight"]
            weights = torch.load(os.path.join(path, emb_path))
            tensor = weights["transformer.wte.weight"].float()
    return tensor


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


def compute_distillation_loss(
        draft_logits,
        target_logits,
        target_probs,
        loss_mask,
        mode="topk_kl",
        topk=10,
):
    """Compute the frozen-target alignment term used alongside VSD.
    """
    token_mask = loss_mask.squeeze(-1).float()
    denominator = token_mask.sum().clamp_min(1e-5)

    if mode == "topk_kl":
        k = min(max(int(topk), 1), target_logits.shape[-1])
        _, top_indices = torch.topk(target_logits, k=k, dim=-1)
        target_top_probs = torch.gather(target_probs, -1, top_indices)
        target_top_probs = target_top_probs / target_top_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        draft_top_logits = torch.gather(draft_logits, -1, top_indices)
        draft_top_log_probs = F.log_softmax(draft_top_logits.float(), dim=-1)
        per_token_loss = -(target_top_probs.float() * draft_top_log_probs).sum(dim=-1)
    elif mode == "full_ce":
        draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1)
        per_token_loss = -(target_probs.float() * draft_log_probs).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported distillation mode: {mode}")

    return (per_token_loss * token_mask).sum() / denominator


def compute_vsd_loss(
        draft_logits,
        target_logits,
        loss_mask,
        num_samples=40,
        target_topk=3,
        temperature=0.2,
        use_arw=True,
        use_car=True,
):
    """Monte-Carlo, token-level VSD surrogate for cached-feature training."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if target_topk < 1:
        raise ValueError("target_topk must be at least 1")
    if use_car and temperature <= 0:
        raise ValueError("temperature must be positive when CAR is enabled")

    batch_size, seq_length, vocab_size = draft_logits.shape
    k = min(int(target_topk), target_logits.shape[-1])
    token_mask = loss_mask.squeeze(-1).float()
    denominator = token_mask.sum().clamp_min(1e-6)

    with torch.no_grad():
        draft_probs = F.softmax(draft_logits.float(), dim=-1)
        draft_samples = torch.multinomial(
            draft_probs.reshape(-1, vocab_size),
            num_samples=num_samples,
            replacement=True,
        ).view(batch_size, seq_length, num_samples)

        _, target_topk_indices = torch.topk(target_logits, k=k, dim=-1)
        target_top1 = target_topk_indices[..., :1]
        valid_samples = (
            draft_samples.unsqueeze(-1) == target_topk_indices.unsqueeze(-2)
        ).any(dim=-1)

        random_scores = torch.rand_like(draft_samples, dtype=torch.float32)
        random_scores.masked_fill_(~valid_samples, -1.0)
        selected_index = random_scores.argmax(dim=-1, keepdim=True)
        selected_sample = torch.gather(draft_samples, -1, selected_index)
        has_valid_sample = valid_samples.any(dim=-1, keepdim=True)
        corrected_sample = torch.where(
            has_valid_sample, selected_sample, target_top1
        )

        local_accept_rate = valid_samples.float().mean(dim=-1)
        beta = local_accept_rate if use_arw else torch.ones_like(local_accept_rate)

        rejected_samples = ~valid_samples
        if use_car:
            proposal_confidence = torch.gather(draft_probs, -1, draft_samples)
            rejection_weights = F.softmax(
                proposal_confidence / temperature, dim=-1
            )
        else:
            rejection_weights = torch.ones_like(draft_samples, dtype=torch.float32)

        rejection_weights = rejection_weights * rejected_samples.float()
        rejection_weights = rejection_weights / rejection_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

    draft_log_probs = F.log_softmax(draft_logits.float(), dim=-1)
    positive_nll = -torch.gather(
        draft_log_probs, -1, corrected_sample
    ).squeeze(-1)
    proposal_nll = -torch.gather(draft_log_probs, -1, draft_samples)
    negative_nll = (proposal_nll * rejection_weights).sum(dim=-1)

    per_token_loss = positive_nll - beta * negative_nll
    vsd_loss = (per_token_loss * token_mask).sum() / denominator

    return vsd_loss

def main():

    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--basepath', type=str, default=None)
    parser.add_argument('--tmpdir_v', type=str,
                        default='/home/VSD_train/eagle_data_image', help="visual instruction-tuning dataset")
    parser.add_argument('--tmpdir_t', type=str,
                        default='/home/VSD_train/eagle_data_text', help="text_only instruction-tuning dataset")
    parser.add_argument('--config', type=str, default='llava_v15_7B_config.json')
    parser.add_argument('--cpdir', type=str, default="", help="The path to save the model")
    parser.add_argument('--name', type=str, default="try")
    parser.add_argument(
        '--distill_loss', choices=('topk_kl', 'full_ce'), default='topk_kl',
        help='Frozen-target alignment loss; topk_kl is the VSD(KL) setting.',
    )
    parser.add_argument('--distill_topk', type=int, default=10)
    parser.add_argument('--prediction_weight', type=float, default=1.0)
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--vsd_weight', type=float, default=0.1)
    parser.add_argument('--vsd_num_samples', type=int, default=40)
    parser.add_argument('--vsd_target_topk', type=int, default=3)
    parser.add_argument('--vsd_temperature', type=float, default=0.2)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("-debug", "--debug", action="store_true", help="debug mode")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    train_config = {
        # "lr": 2e-4,
        "bs": 8,
        "gradient_accumulation_steps": 1,
        # "datapath": f"{args.tmpdir}",
        "datapath_v": f"{args.tmpdir_v}",
        "datapath_t": f"{args.tmpdir_t}",
        "is_warmup": True,
        "num_epochs": 40,
        "num_warmup_steps": 1500,
        "total_steps": 85000,
        "p_w": args.prediction_weight,
        "v_w": args.feature_weight,
        "head_w": 0.1,
        "distill_loss": args.distill_loss,
        "distill_topk": args.distill_topk,
        "vsd_weight": args.vsd_weight,
        "vsd_num_samples": args.vsd_num_samples,
        "vsd_target_topk": args.vsd_target_topk,
        "vsd_temperature": args.vsd_temperature,
        "num_workers": 4,
        "embeding": True,
        "act": "No",
        "data_noise": True,
        "noise": "uniform",
        "mean": 0.0,
        "std": 0.2,
        "residual": "true,norm",
        "max_len": 2048,
        "config_path": f"{args.config}",
        "b1": 0.9,
        "b2": 0.95,
        "grad_clip": 0.5,
    }

    debug = args.debug


    torch.backends.cuda.matmul.allow_tf32 = True
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    set_seed(66)
    use_bf16 = True
    accelerator = Accelerator(mixed_precision="bf16")

    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    if rank == 0:
        import wandb
        wandb.init(
            project="VSD",
            name=args.name,
            config=train_config
        )

    try:
        with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(args.basepath, head_path),
                    framework="pt",
                    device="cpu") as f:
            tensor_slice = f.get_slice("lm_head.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            tensor = tensor_slice[:, :hidden_dim].float()
    except:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path))
        tensor = weights["lm_head.weight"].float()

    head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
    head.weight.data = tensor


    class CustomDataset(Dataset):
        def __init__(self, datapath_v, datapath_t, total_epoch, transform=None):
            self.data_v = datapath_v
            self.data_t = datapath_t
            self.total_epoch = total_epoch
            self.transform = transform

            self.epoch_now = 1
            self.dataset_v_size = len(self.data_v)
            self.dataset_t_size = len(self.data_t)

            self.current_ratio = 0.0

        def update_ratio(self):
            if self.epoch_now <= self.total_epoch // 2:
                self.current_ratio = 0.0
            elif self.total_epoch // 2 < self.epoch_now < self.total_epoch:
                self.current_ratio = (self.epoch_now - self.total_epoch // 2) / self.total_epoch * 2
            else:
                self.current_ratio = 1.0

        def get_data_source(self):
            if random.random() < self.current_ratio:
                return self.data_v
            else:
                return self.data_t
            # return self.data_t
            
        def __len__(self):
            return max(len(self.data_v), len(self.data_t))
        
        def load_data(self, idx, dataset):
            data = torch.load(dataset[idx])
            return data

        def __getitem__(self, index):
            dataset = self.get_data_source()
            index = index % len(dataset) # ori
            data = self.load_data(index, dataset)
            new_data = {}
            hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
            input_ids = data['input_ids'][:train_config["max_len"]][None, :]
            loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]
            if "inputs_embeds" in data.keys():
                inputs_embeds = data["inputs_embeds"][:train_config["max_len"]][None, :]
            else:
                inputs_embeds = embedding(input_ids).to(torch.bfloat16)

            image_tokens_num = 0
            if -200 in input_ids:
                image_tokens_num = 576
            elif 151652 in input_ids:
                image_tokens_num = (input_ids == 151655).sum().item()

            length = hidden_state.shape[1]
            attention_mask = [1] * length
            loss_mask = loss_mask[0].tolist()
            loss_mask[-1] = 0

            input_ids_target = input_ids[:, 1:]
            zeropadding = torch.tensor([[0]])
            input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

            inputs_embeds_target = inputs_embeds[:, 1:, :]
            inputs_embeds_target = torch.cat((inputs_embeds_target, zeropadding_embed), dim=1)
            new_data["inputs_embeds"] = inputs_embeds_target.to(torch.bfloat16)

            target = hidden_state[:, 1:, :]
            zeropadding = torch.zeros(1, 1, target.shape[2])
            target = torch.cat((target, zeropadding), dim=1)
            loss_mask[-1] = 0
            new_data["attention_mask"] = attention_mask
            new_data["loss_mask"] = loss_mask
            new_data["target"] = target.to(torch.bfloat16)
            new_data["hidden_state_big"] = hidden_state.to(torch.bfloat16)
            new_data["input_ids"] = input_ids_target
            new_data["image_tokens_num"] = image_tokens_num


            if self.transform:
                new_data = self.transform(new_data)

            return new_data
        
        def on_epoch_end(self):
            self.epoch_now += 1
            self.update_ratio()


    class DataCollatorWithPadding:

        def paddingtensor(self, intensors, N):
            B, n, S = intensors.shape
            padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
            outtensors = torch.cat((intensors, padding_tensor), dim=1)
            return outtensors

        def paddingtensor2D(self, intensors, N, have_img=False):
            if have_img == True:
                img_padding_tensor = torch.zeros(intensors.shape[0], 575, dtype=intensors.dtype)
                intensors = torch.cat((img_padding_tensor, intensors), dim=1)
            if intensors.shape[1] >= train_config["max_len"]:
                intensors = intensors[:,:train_config["max_len"]]
            B, n = intensors.shape
            # N maybe 2048

            # print(f"B is {B}, n is {n}")

            if n > N:
                print(f"B is {B}, n is {n}")
                raise ValueError(f"n ({n}) > N ({N})")

            padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
            outtensors = torch.cat((intensors, padding_tensor), dim=1)
            return outtensors

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

            def effective_length(item):
                base = item['hidden_state_big'].shape[1]
                if -200 in item['input_ids']:
                    base += 575
                return base

            max_length = max(effective_length(item) for item in features)

            valid_features = []
            for item in features:
                if effective_length(item) > max_length:
                    continue

                try:
                    _ = self.paddingtensor2D(
                        item['input_ids'],
                        max_length,
                        have_img=(-200 in item['input_ids'])
                    )
                except ValueError:
                    continue

                valid_features.append(item)

            features = valid_features

            batch_input_ids = torch.cat([
                self.paddingtensor2D(
                    item['input_ids'],
                    max_length,
                    have_img=(-200 in item['input_ids'])
                )
                for item in features
            ], dim=0)

            batch_hidden_states = torch.cat([
                self.paddingtensor(item['hidden_state_big'], max_length)
                for item in features
            ], dim=0)

            batch_inputs_embeds = torch.cat([
                self.paddingtensor(item['inputs_embeds'], max_length)
                for item in features
            ], dim=0)

            batch_target = torch.cat([
                self.paddingtensor(item['target'], max_length)
                for item in features
            ], dim=0)

            batch_loss_mask = torch.tensor([
                item['loss_mask'] + [0] * (max_length - len(item['loss_mask']))
                for item in features
            ])

            batch_attention_mask = torch.tensor([
                item['attention_mask'] + [0] * (max_length - len(item['attention_mask']))
                for item in features
            ])

            batch_image_tokens_num = [
                item['image_tokens_num'] for item in features
            ]

            return {
                "input_ids": batch_input_ids,
                "inputs_embeds": batch_inputs_embeds,
                "hidden_states": batch_hidden_states,
                "target": batch_target,
                "attention_mask": batch_attention_mask,
                "loss_mask": batch_loss_mask,
                "image_tokens_num": batch_image_tokens_num
            }



    def top_accuracy(output, target, topk=(1,)):
        # output.shape (bs, num_classes), target.shape (bs, )
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return res

    def compute_loss(target_features, target_p, target_logits, predict, loss_mask):
        out_head = head_engine(predict)
        prediction_loss = compute_distillation_loss(
            draft_logits=out_head,
            target_logits=target_logits,
            target_probs=target_p,
            loss_mask=loss_mask,
            mode=train_config["distill_loss"],
            topk=train_config["distill_topk"],
        )

        feature_loss = criterion(
            predict.float(), target_features.float().to(rank)
        ).to(torch.bfloat16)
        feature_loss = torch.sum(
            torch.mean(loss_mask * feature_loss, dim=2)
        ) / (loss_mask.sum() + 1e-5)

        base_loss = (
            train_config["v_w"] * feature_loss
            + train_config["p_w"] * prediction_loss
        )
        vsd_loss = compute_vsd_loss(
            draft_logits=out_head,
            target_logits=target_logits,
            loss_mask=loss_mask,
            num_samples=train_config["vsd_num_samples"],
            target_topk=train_config["vsd_target_topk"],
            temperature=train_config["vsd_temperature"],
            use_arw=True,
            use_car=True,
        )

        return (
            vsd_loss,
            base_loss,
            prediction_loss,
            feature_loss,
            out_head,
        )

    llama_path = args.basepath
    embed_tensor = get_embed(llama_path)
    embedding = torch.nn.Embedding(embed_tensor.shape[0], embed_tensor.shape[1])
    embedding.weight.data = embed_tensor
    for param in embedding.parameters():
        param.requires_grad = False

    if train_config["data_noise"]:
        if train_config["noise"] == "uniform":
            aug = AddUniformNoise(std=train_config["std"])
        else:
            aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
    else:
        aug = None

    datapath_v = list_files(train_config["datapath_v"])
    datapath_t = list_files(train_config["datapath_t"])

    if debug == True:
        traindatapath_v = datapath_v[:100]
        traindatapath_t = datapath_t[:100]
        testdatapath_v = datapath_v[100:110]
        testdatapath_t = datapath_t[100:110]
    else:
        traindatapath_v = datapath_v[:int(len(datapath_v) * 0.95)]
        traindatapath_t = datapath_t[:int(len(datapath_t) * 0.95)]
                                    
        testdatapath_v = datapath_v[int(len(datapath_v) * 0.95):]
        testdatapath_t = datapath_t[int(len(datapath_t) * 0.95):]
    traindataset = CustomDataset(traindatapath_v, traindatapath_t, total_epoch=train_config["num_epochs"], transform=aug)
    testdataset = CustomDataset(testdatapath_v, testdatapath_t, total_epoch=train_config["num_epochs"])

    if rank == 0:
        if not os.path.exists(args.cpdir):
            os.makedirs(args.cpdir)

    if "Qwen" in args.basepath:
        config = Qwen2VLConfig.from_pretrained(train_config["config_path"])
    else:
        config = EConfig.from_pretrained(train_config["config_path"])

    model = Model(config, path=args.basepath, load_emb=True, train_embed=False, decouple=True)
    zeropadding_embed = model.embed_tokens(torch.tensor(0))[None, None, :].to(dtype=(torch.bfloat16 if use_bf16 else torch.float16)).detach() 

    criterion = nn.SmoothL1Loss(reduction="none")

    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]
    is_warmup = train_config["is_warmup"]

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                    model=model,
                                                                    model_parameters=model.parameters(),
                                                                    training_data=traindataset,
                                                                    collate_fn=DataCollatorWithPadding()
                                                                    )

    head_engine, _, test_loader, _ = deepspeed.initialize(args=args,
                                                        model=head,
                                                        model_parameters=head.parameters(),
                                                        training_data=testdataset,
                                                        collate_fn=DataCollatorWithPadding()
                                                        )


    for param in head.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.train()
        for batch_idx, data in enumerate(tqdm(train_loader)):

            model.zero_grad()

            predict = model_engine(data["hidden_states"].to(rank), input_ids=data["input_ids"].to(rank),
                                inputs_embeds=data["inputs_embeds"].to(rank),
                                attention_mask=data["attention_mask"].to(rank),
                                image_tokens_num=data["image_tokens_num"],
                                )

            with torch.no_grad():
                target_head = head_engine(data["target"].to(rank))
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            loss_mask = data["loss_mask"][:, :, None].to(rank)
            (
                vsd_loss,
                base_loss,
                prediction_loss,
                feature_loss,
                out_head,
            ) = compute_loss(
                data["target"], target_p, target_head, predict, loss_mask
            )

            loss = base_loss + train_config["vsd_weight"] * vsd_loss
            # loss.backward()
            model_engine.backward(loss)
            # accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])

            model_engine.step()

            with torch.no_grad():
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            if rank == 0 and ct != 0:
                logdict = {
                    "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                    "train/vsd_loss": vsd_loss.item(),
                    "train/base_loss": base_loss.item(),
                    "train/prediction_loss": prediction_loss.item(),
                    "train/feature_loss": feature_loss.item(),
                    "train/loss": loss.item(),
                    "train/acc": cc / ct,
                }
                for id, i in enumerate(top_3acc):
                    logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
                wandb.log(logdict)
                # for id,i in enumerate(top_3acc):
                #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

            del vsd_loss, base_loss, prediction_loss, feature_loss
            epoch_loss += loss.item()
            num_batches += 1
        train_loader.dataset.on_epoch_end()
        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        epoch_loss /= num_batches
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
        if accelerator.is_local_main_process:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
            wandb.log({"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss})

        model_engine.save_16bit_model(f"{args.cpdir}/state_{epoch}")

        if epoch % 10 == 0:
            deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.cpdir}/state_{epoch}")
            name = args.name
            save_path = f"{args.cpdir}/{name}_{epoch}"

            # Ensure the directory exists
            os.makedirs(save_path, exist_ok=True)

            # Then save the model
            model_engine.save_16bit_model(save_path)
            torch.distributed.barrier()
            config = train_config["config_path"]
            import shutil
            target_file = f"{args.cpdir}/{name}_{epoch}/config.json"
            shutil.copy(config, target_file)

        torch.distributed.barrier()

    name = args.name
    save_path = f"{args.cpdir}/{name}_{epoch}"

    os.makedirs(save_path, exist_ok=True)

    model_engine.save_16bit_model(save_path)
    torch.distributed.barrier()
    config = train_config["config_path"]
    import shutil
    target_file = f"{args.cpdir}/{name}_{epoch}/config.json"
    shutil.copy(config, target_file)
    torch.distributed.barrier()
    
if __name__ == "__main__":
    main()
