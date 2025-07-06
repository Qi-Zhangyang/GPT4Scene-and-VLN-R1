# VLN-Ego（VLN-R1的数据集）制作过程

VLN-Ego是借助[VLN-CE](https://jacobkrantz.github.io/vlnce/)所制作的。本质来说，我们借用了VLN-CE中Room-to-Room (**R2R**) 和 Room-Across-Room (**RxR**)两个轨迹集，并且在[Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7)中渲染出ego centric video (以每一帧图片的形式给出)。


## 环境搭建

这里需要注意python版本，habitat-sim最高支持python版本为3.8,所以这里python版本必须是3.8以下。这带来一个问题，就是度模态的模型只能是Qwen2-VL，Qwen2.5-VL就已经无法支持了。这里的环境搭建参考[VLN-CE](https://jacobkrantz.github.io/vlnce/)。

```
conda create -n vlnr1-eval python=3.8
conda activate vlnr1-eval
```

VLN-CE 使用 [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) 0.1.7 版本，您可以[从源码构建](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation)或通过 conda 安装：

```
# 无头版本
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless

```

然后安装 [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) 0.1.7 版本：

```
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# 安装 habitat 和 habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all

```

现在您可以安装 VLN-CE部分：

```
git clone https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1.git
cd VLN-Ego/VLN-CE
python -m pip install -r requirements.txt

```

### 预先需要下载的数据
具体来说，数据的下载可以参考[VLN-CE](https://jacobkrantz.github.io/vlnce/)中的数据准备部分，在这里我们提供了一个[hugging face](https://huggingface.co/datasets/alexzyqi/VLN-Ego-making/)将所有的需要预先下载的东西进行集成。

您只需要运行下面python代码便可以下载所有，需要注意的是，这里总共大约**230G**。

```
python download_prepare.py
```

下载下来的数据集结构为
```
└── data
    ├── checkpoints               # VLN-CE训练好的模型，在制作VLN-Ego中可能没什么用。
    │   ├── cma
    │   ├── cma_aug
    │   └── rxr_cma_en
    ├── connectivity_graphs.pkl   # 不是VLN-CE，而是离散点的连结图，并非连续环境，在制作VLN-Ego中可能没什么用。
    ├── datasets                  # R2R和RxR的轨迹，在VLN-Ego的制作中需要。
    │   ├── R2R_VLNCE_v1-3
    │   ├── R2R_VLNCE_v1-3_preprocessed
    │   └── RxR_VLNCE_v0
    ├── ddppo-models              # 预训练的视觉模型权重，在制作VLN-Ego中可能没什么用。
    │   └── gibson-2plus-resnet50.pth
    ├── scene_datasets            # MP3D数据集，在VLN-Ego的制作中需要。
    │   └── mp3d
    ├── tensorboard_dirs          # 在制作VLN-Ego中可能没什么用。
    │   ├── cma
    │   └── rxr_cma_en
    └── trajectories_dirs         # VLN-CE的推理的轨迹文件，在制作VLN-Ego中可能没什么用。
        ├── cma
        ├── cma_aug
        ├── debug
        ├── r2r
        └── rxr_en_guide_trim250
```

在RxR_VLNCE_v0这个文件夹中，text_feature这个大小非常大。但是我们在这里也是仅仅使用其英语部分。



## 通过运行run.py生成数据

我们的核心是在运行run.py的时候保存图像，您可以注意我们注释掉了*vlnce_baselines/recollect_trainer.py*中模型forward的部分，仅仅运行dataloader部分。

下面我们演示R2R的数据制作流程。

```
python run.py \
  --exp-config vlnce_baselines/config/r2r_baselines/r2r_vln-ego-making.yaml \
  --run-type train
```

R2R的split有着*train/val_seen/val_unseen*。需要注意的是，我们主要需要train这部分。当然，如果您需要val_seen和val_unseen，您可以修改r2r_vln-ego-making.yaml.

```
    gt_file:
      data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
    RGB_SAVE_DIR: ./VLN-Ego/rgb_images_r2r_{split}
    preload_size: 50
```

将split改成val_seen或者val_unseen,便可以得到结果。

-----------

**RxR**的制造过程和R2R较为类似，
```
python run.py \
  --exp-config vlnce_baselines/config/rxr_baselines/rxr_vln-ego-making.yaml \
  --run-type train
```

## 后处理
经过上一步，我们得到了ego-centric的images。
```
└── VLN-Ego
    ├── rgb_images_r2r_train
    └── rgb_images_rxr_train
    └── rgb_images_r2r_debug        # 这里我们提供了一个demo展示后续的后处理
```
为了能够得到LLaMA-Factory能够训练的sft格式，我们需要得到相应的annotations。这里我们使用以下代码得到。

```
python r2r_rxr_dataset_convert_sft \
    --base_folder "VLN-Ego/rgb_images_r2r_debug/" \
    --output_file "VLN-Ego/r2r_16_images_1act_debug.json" \
    --image_base_path "data/r2r_training_rgb/" \
    --max_frames 16 \
    --num_actions 6
```

最后，将得到的r2r_16_images_1act_debug.json和原本的rgb_images_r2r_debug（**需要rename**）放入LLaMA-Factory的data下就可以进行sft训练。

## 引用

首先您需要引用VLN-R1:
```
@article{VLNR1,
  title={VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning},
  author={Zhangyang Qi and Zhixiong Zhang and Yizhou Yu and Jiaqi Wang and Hengshuang Zhao},
  journal={arXiv:2506.17221},
  year={2025}
}
```

其次，您需要引用RxR和VLN-CE:

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
