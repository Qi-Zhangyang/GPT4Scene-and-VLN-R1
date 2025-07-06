# VLN-Ego (Dataset for VLN-R1) Production Process

<span style="color: red;">The fully processed dataset is available on [Hugging Face (alexzyqi/VLN-Ego)](https://huggingface.co/datasets/alexzyqi/VLN-Ego/), and you can directly download and unzip it. This page mainly demonstrates the production process of this dataset.</span>

VLN-Ego is created using [VLN-CE](https://jacobkrantz.github.io/vlnce/). Essentially, we utilize two trajectory sets from VLN-CE: Room-to-Room (**R2R**) and Room-Across-Room (**RxR**), and render ego-centric videos (in the form of frame-by-frame images) in [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7).


## Environment Setup

Note the Python version here: Habitat-Sim supports a maximum Python version of 3.8, so the Python version must be 3.8 or lower. This presents an issue where only Qwen2-VL can be used as the multimodal model, as Qwen2.5-VL is no longer compatible. The environment setup here refers to [VLN-CE](https://jacobkrantz.github.io/vlnce/).

```
conda create -n vlnr1-eval python=3.8
conda activate vlnr1-eval
```

VLN-CE uses [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) version 0.1.7. You can either [build from source](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation) or install via conda:

```
# Headless version
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless

```

Then install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) version 0.1.7:

```
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# Install habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all

```

Now you can install the VLN-CE component:

```
git clone https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1.git
cd VLN-Ego-making/
python -m pip install -r requirements.txt

```

### Pre-required Data Downloads
For data downloading, refer to the data preparation section in [VLN-CE](https://jacobkrantz.github.io/vlnce/). Here, we provide a [Hugging Face link (alexzyqi/VLN-Ego-making)](https://huggingface.co/datasets/alexzyqi/VLN-Ego-making/) integration of all required pre-downloads.

You only need to run the following Python code to download everything. Note that the total size is approximately **230GB**.

```
python download_prepare.py
```

The downloaded dataset structure is:
```
└── data
    ├── checkpoints               # Pre-trained models from VLN-CE, possibly unused in VLN-Ego creation.
    │   ├── cma
    │   ├── cma_aug
    │   └── rxr_cma_en
    ├── connectivity_graphs.pkl   # Not from VLN-CE; a discrete point connection graph, not a continuous environment, possibly unused in VLN-Ego creation.
    ├── datasets                  # Trajectories for R2R and RxR, required for VLN-Ego creation.
    │   ├── R2R_VLNCE_v1-3
    │   ├── R2R_VLNCE_v1-3_preprocessed
    │   └── RxR_VLNCE_v0
    ├── ddppo-models              # Pre-trained vision model weights, possibly unused in VLN-Ego creation.
    │   └── gibson-2plus-resnet50.pth
    ├── scene_datasets            # MP3D dataset, required for VLN-Ego creation.
    │   └── mp3d
    ├── tensorboard_dirs          # Possibly unused in VLN-Ego creation.
    │   ├── cma
    │   └── rxr_cma_en
    └── trajectories_dirs         # Inference trajectory files from VLN-CE, possibly unused in VLN-Ego creation.
        ├── cma
        ├── cma_aug
        ├── debug
        ├── r2r
        └── rxr_en_guide_trim250
```

The `text_feature` folder within `RxR_VLNCE_v0` is very large. However, we only use its English portion here.


## Generating Data by Running run.py

Our core process involves saving images while running `run.py`. Note that we have commented out the model forward pass section in `vlnce_baselines/recollect_trainer.py`, running only the dataloader component.

Below is a demonstration of the R2R data creation process:

```
python run.py \
  --exp-config vlnce_baselines/config/r2r_baselines/r2r_vln-ego-making.yaml \
  --run-type train
```

R2R splits include `train/val_seen/val_unseen`. Note that we primarily need the `train` portion. However, if you need `val_seen` and `val_unseen`, you can modify `r2r_vln-ego-making.yaml`:

```
    gt_file:
      data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
    RGB_SAVE_DIR: ./VLN-Ego/rgb_images_r2r_{split}
    preload_size: 50
```

Changing `split` to `val_seen` or `val_unseen` will generate the corresponding results.

-----------

The creation process for **RxR** is similar to R2R:
```
python run.py \
  --exp-config vlnce_baselines/config/rxr_baselines/rxr_vln-ego-making.yaml \
  --run-type train
```

## Post-Processing
After the previous step, we obtain ego-centric images:
```
└── VLN-Ego
    ├── rgb_images_r2r_train
    └── rgb_images_rxr_train
    └── rgb_images_r2r_debug        # A demo showcasing subsequent post-processing
```

To obtain the SFT format compatible with LLaMA-Factory training, we need the corresponding annotations. Use the following code:

```
python r2r_rxr_dataset_convert_sft \
    --base_folder "VLN-Ego/rgb_images_r2r_debug/" \
    --output_file "VLN-Ego/r2r_16_images_1act_debug.json" \
    --image_base_path "data/r2r_training_rgb/" \
    --max_frames 16 \
    --num_actions 6
```

Finally, place the resulting `r2r_16_images_1act_debug.json` and the original `rgb_images_r2r_debug` (**rename required**) into the `data` folder of LLaMA-Factory for SFT training.

## Citations

First, you need to cite VLN-R1:
```
@article{VLNR1,
  title={VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning},
  author={Zhangyang Qi and Zhixiong Zhang and Yizhou Yu and Jiaqi Wang and Hengshuang Zhao},
  journal={arXiv:2506.17221},
  year={2025}
}
```

Additionally, you need to cite RxR and VLN-CE:

```
@inproceedings{ku2020room,
  title={Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding},
  author={Ku, Alexander and Anderson, Peter and Patel, Roma and Ie, Eugene and Baldridge, Jason},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4392--4412},
  year={2020}
}

@inproceedings{krantz_vlnce_2020,
  title={Beyond the Nav-Graph: Vision and Language Navigation in Continuous Environments},
  author={Jacob Krantz and Erik Wijmans and Arjun Majundar and Dhruv Batra and Stefan Lee},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
 }

```
