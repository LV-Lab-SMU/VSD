# Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance

<p align="center">
  <a href="https://arxiv.org/abs/2602.05774">
    <img src="https://img.shields.io/badge/📝-Paper-blue" height="25" alt="Paper">
  </a>
</p>

## Overview

> **TL;DR:** **Variational Speculative Decoding (VSD)** formulates draft-model training as variational inference over latent draft paths, thereby aligning the training objective with sequence-level acceptance during speculative decoding.

VSD is a sequence-level training framework that treats draft paths as latent proposals and maximizes the marginal probability that they are accepted by the target model. This formulation yields an evidence lower bound (ELBO) that encourages high-quality proposals while regularizing the draft distribution toward the target distribution.

## Released Models

We release pretrained VSD draft models for LLaVA-v1.5 at the following Hugging Face repositories:

| Target Model   | VSD Draft Model                                                           |
| -------------- | ------------------------------------------------------------------------- |
| LLaVA-v1.5-7B  | [VSD_llava_v15_7b](https://huggingface.co/yqcca/VSD_llava_v15_7b)   |
| LLaVA-v1.5-13B | [VSD_llava_v15_13b](https://huggingface.co/yqcca/VSD_llava_v15_13b) |

## Repository Structure

The installation and training commands below assume that the following repositories are located under the same project directory:

```text
.
├── EAGLE
├── LLaVA
├── lmms-eval
└── image_data
```

## Setup and Installation

Create a Conda environment and install the required repositories in editable mode:

```bash
conda create -n vsd python=3.10 -y
conda activate vsd

cd LLaVA
pip install -e .

cd ../EAGLE
pip install -e .

cd ../lmms-eval
pip install -e .
```

## Training

### 1. Prepare Annotation Data

Download the annotation files used for instruction tuning:

* [`ShareGPT_V3_unfiltered_cleaned_split.json`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)
* [`llava_v1_5_mix665k.json`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

### 2. Prepare Image Data

Download images from the following datasets:

* **COCO:** [train2017](http://images.cocodataset.org/zips/train2017.zip)
* **GQA:** [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
* **OCR-VQA:** [download scripts and files](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)
* **TextVQA:** [train and validation images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
* **Visual Genome:** [part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Organize the downloaded images under `./image_data` using the following directory structure:

```text
image_data
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

### 3. Generate Training Data

Run the data-allocation script to generate the training data required by the draft model.

The target model family is controlled through `--model_type`. Supported examples include:

* `llava_v15_t`
* `llava_v15_v`
* `qwen2_vl_t`
* `qwen2_vl_v`

```bash
cd EAGLE/eagle/ge_data

CUDA_VISIBLE_DEVICES=0 python -m eagle.ge_data.allocation \
    --outdir <output_data_dir> \
    --model_type <model_type> \
    --model <base_model_path> \
    --image_data_path <image_data_dir> \
    --json_data_path <annotation_file>
```

#### Arguments

| Argument            | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `<output_data_dir>` | Directory in which the processed training data will be saved |
| `<model_type>`      | Target-model and data-modality configuration                 |
| `<base_model_path>` | Path to the pretrained target model                          |
| `<image_data_dir>`  | Root directory containing the downloaded image datasets      |
| `<annotation_file>` | Path to the instruction-tuning annotation file               |

### 4. Train the VSD Model

Use DeepSpeed to train the speculative draft model:

```bash
cd EAGLE/eagle/train

deepspeed \
    --master_port 29504 \
    main_deepspeed_vsd.py \
    --deepspeed_config ds_config.json \
    --tmpdir_v <visual_data_path> \
    --tmpdir_t <text_data_path> \
    --basepath <base_model_path> \
    --cpdir <checkpoint_output_dir> \
    --config <training_config_path>
```

#### Arguments

| Argument                  | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| `<visual_data_path>`      | Directory containing preprocessed visual training data                      |
| `<text_data_path>`        | Directory containing preprocessed text training data                        |
| `<base_model_path>`       | Path to the pretrained target model                                         |
| `<checkpoint_output_dir>` | Directory in which training checkpoints will be saved                       |
| `<training_config_path>`  | Path to the training configuration file, such as `llava_v15_7B_config.json` |

## Inference and Evaluation

We use [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) for multimodal evaluation.

The following example evaluates a VSD model on **ChartQA**:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29506 \
    -m lmms_eval \
    --model <model_name> \
    --model_args pretrained="<base_model_path>" \
    --vsd_model_path <vsd_model_path> \
    --tasks chartqa \
    --batch_size 1 \
    --use_vsd
```

### Arguments

| Argument            | Description                                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| `<model_name>`      | Model adapter registered in `lmms-eval`, such as `llava_vsd` or `qwen2_vl_vsd` |
| `<base_model_path>` | Path to the pretrained target model                                            |
| `<vsd_model_path>`  | Path to the trained VSD checkpoint                                             |
| `--tasks`           | Evaluation task name, such as `chartqa`                                        |
| `--use_vsd`         | Enables VSD-based speculative decoding                                         |

To evaluate on another benchmark, replace `chartqa` with the corresponding task name supported by `lmms-eval`.

## Acknowledgments

This repository is built upon the following open-source projects:

* [Transformers](https://github.com/huggingface/transformers)
* [EAGLE](https://github.com/SafeAILab/EAGLE)
* [VSD](https://arxiv.org/abs/2602.05774)
* [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

We thank the open-source community for making their models, codebases, and evaluation tools publicly available.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zou2026variational,
  title   = {Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance},
  author  = {Zou, Xiandong and Li, Jianshu and Huang, Jing and Zhou, Pan},
  journal = {arXiv preprint arXiv:2602.05774},
  year    = {2026}
}
```
