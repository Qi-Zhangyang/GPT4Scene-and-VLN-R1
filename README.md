
## <img src="./assets/vlnr1_logo.png" alt="Icon" width="70" height="70"> VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning

<div style="text-align: center;">
  <p class="title is-5 mt-2 authors"> 
    <a href="https://scholar.google.com/citations?user=kwVLpo8AAAAJ&hl=en/" target="_blank">Zhangyang Qi</a><sup>1,2*</sup>, 
    <a href="https://github.com/rookiexiong7" target="_blank">Zhixiong Zhang</a><sup>2*</sup>, 
    <a href="https://i.cs.hku.hk/~yzyu/publication/" target="_blank">Yizhou Yu</a><sup>1</sup>, 
    <a href="https://myownskyw7.github.io/" target="_blank">Jiaqi Wang</a><sup>2&#9993;</sup>,
    <a href="https://hszhao.github.io/" target="_blank">Hengshuang Zhao</a><sup>1&#9993;</sup>
  </p>
</div>

<div style="text-align: center;">
    <!-- contribution -->
    <p class="subtitle is-5" style="font-size: 1.0em; text-align: center;">
        <sup>*</sup> Equation Contribution,
        <sup>&#9993;</sup> Corresponding Authors,
    </p>
</div>

<div style="text-align: center;">
  <!-- affiliations -->
  <p class="subtitle is-5" style="font-size: 1.0em; text-align: center;"> 
    <sup>1</sup> The University of Hong Kong, 
    <sup>2</sup> Shanghai AI Laboratory,
  </p>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2506.17221" target='_**blank**'>
    <img src="https://img.shields.io/badge/arXiv-2501.01428📖-bron?">
  </a> 
  <a href="https://vlnr1.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project%20page-&#x1F680-yellow">
  </a>
  <a href="https://huggingface.co/datasets/alexzyqi/VLN-R1-datasets/" target='_blank'>
    <img src="https://img.shields.io/badge/Huggingface%20Datasets-🤗-blue">
  </a>
  <a href="https://x.com/Qi_Zhangyang" target='_blank'>
    <img src="https://img.shields.io/twitter/follow/Qi_Zhangyang">
  </a>
</p>


## <img src="./assets/gptscene_logo.png" alt="Icon" width="80" height="40"> GPT4Scene: Understand 3D Scenes from Videos with Vision-Language Models

<div style="text-align: center;">
  <p class="title is-5 mt-2 authors"> 
    <a href="https://scholar.google.com/citations?user=kwVLpo8AAAAJ&hl=en/" target="_blank">Zhangyang Qi</a><sup>1,2*</sup>, 
    <a href="https://github.com/rookiexiong7" target="_blank">Zhixiong Zhang</a><sup>2*</sup>, 
    <a href="https://github.com/Aleafy" target="_blank">Ye Fang</a><sup>2</sup>, 
    <a href="https://myownskyw7.github.io/" target="_blank">Jiaqi Wang</a><sup>2&#9993;</sup>,
    <a href="https://hszhao.github.io/" target="_blank">Hengshuang Zhao</a><sup>1&#9993;</sup>
  </p>
</div>

<div style="text-align: center;">
    <!-- contribution -->
    <p class="subtitle is-5" style="font-size: 1.0em; text-align: center;">
        <sup>*</sup> Equation Contribution,
        <sup>&#9993;</sup> Corresponding Authors,
    </p>
</div>

<div style="text-align: center;">
  <!-- affiliations -->
  <p class="subtitle is-5" style="font-size: 1.0em; text-align: center;"> 
    <sup>1</sup> The University of Hong Kong, 
    <sup>2</sup> Shanghai AI Laboratory,
  </p>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2501.01428" target='_**blank**'>
    <img src="https://img.shields.io/badge/arXiv-2501.01428📖-bron?">
  </a> 
  <a href="https://gpt4scene.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project%20page-&#x1F680-yellow">
  </a>
  <a href="https://huggingface.co/alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512" target='_blank'>
    <img src="https://img.shields.io/badge/Huggingface%20Models-🤗-blue">
  </a>
  <a href="https://x.com/Qi_Zhangyang" target='_blank'>
    <img src="https://img.shields.io/twitter/follow/Qi_Zhangyang">
  </a>
</p>

## 🔥 News

[2025/07/06] We have released the **[training data](https://huggingface.co/datasets/alexzyqi/VLN-R1-datasets/)**, **[tokenizers](https://huggingface.co/datasets/alexzyqi/GPT4Scene_VLN-R1_tokenizers/)** and **[data generation code](https://huggingface.co/datasets/alexzyqi/VLN-R1-datasets/)** for **VLN-R1**.

[2025/06/20] We release the **[VLN-R1](https://arxiv.org/abs/2506.17221)** paper in arxiv.

[2025/03/10] We release the **[training and validation dataset](https://huggingface.co/datasets/alexzyqi/GPT4Scene-All)** and training code.

[2025/01/21] We release the **[code](https://github.com/Qi-Zhangyang/GPT4Scene)** validation datasetand **[model weights](https://huggingface.co/alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512)**.

[2025/01/01] We release the **[GPT4Scene](https://arxiv.org/abs/2501.01428)** paper in arxiv. (**The first paper in 2025! 🎇🎇🎇**).


## 🔥 Note
This code currently focuses on GPT4Scene. We will merge the SFT section of VLN-R1 with GPT4Scene in the future. The following shows the open-source status of VLN-R1:

| Task | Status |
|------|--------|
| VLN-R1 training data | ✅ Completed |
| VLN-R1 dataset production process | ✅ Completed |
| VLN-R1 SFT training section | 🔄 In progress |
| VLN-R1 testing section (including engine) | 🔄 In progress |
| VLN-R1 RFT section | 🔄 In progress |


## 🔧 Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
conda create --name gpt4scene python=3.10
conda activate gpt4scene

git clone https://github.com/Qi-Zhangyang/GPT4Scene.git
cd GPT4Scene

pip install -e ".[torch,metrics]"
```

Sometimes, the PyTorch downloaded this way may encounter errors. In such cases, you need to manually install [Pytorch](https://pytorch.org/).

```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install qwen_vl_utils flash-attn
```

## 🎡 Models and Weights

| Function             | Model Name           | Template                                                        |
| ---------------------| -------------------- | ----------------------------------------------------------------|
| **Pretrain Models**  | Qwen2-VL-7B-Instruct | [Huggingface Link](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| **Trained Weights** | GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512                 | [Huggingface Link](https://huggingface.co/alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512)    |


```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 🗂️ Dataset (ScanAlign)

| Function             | Huggingface Dataset Link       | Local Dir                                                        |
| ---------------------| -------------------- | ----------------------------------------------------------------|
| **Train and Val Dataset and Train Annotations**  | [alexzyqi/GPT4Scene-All](https://huggingface.co/datasets/alexzyqi/GPT4Scene-All) | ./data/ |

You can download all trained model weights, dataset and annotations by 

```bash
python download.py
```

The folder structure is as follows.

```plaintext
GPT4Scene
├── ckpts
│   ├── Qwen2-VL-7B-Instruct
├── data
│   ├── annotation
│   │   ├── images_2D
│   │   ├── images_3D
│   │   ├── sharegpt_data_chat_scene_32_images_3D_mark.json
├── evaluate
│   ├── annotation
│   │   ├── multi3dref_mask3d_val.json
│   │   ├── ...
│   │   └── sqa3d_val.json
│   ├── ...
│   └── utils
├── model_outputs
│   └── GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512
├── ...
└── README.md
```

## 🚀 Inference

To inference, you can run the script

```bash
bash evaluate/infer.sh
```

It will **automatically detect the number of GPUs** in your current environment and perform chunked testing. Also you can use the **slurm system** to submit your evaluation task.

```bash
srun -p XXX --gres=gpu:4 --time=4-00:00:00 sh evaluate/infer.sh
```

## 🏗️ Training

You can start by getting the tokenizer - it only requires CPU resources.
```bash
bash gpt4scene_tokenizer_scripts/qwen2vl_7b_full_sft_mark_32_3D_img512.sh
```

Then you can run the training through 8 gpus.
```bash
bash gpt4scene_bash_scripts/qwen2vl_7b_full_sft_mark_32_3D_img512.sh
```


Also, you can run the torchrun code.


Please note that the initial run requires the tokenizer. It is recommended to disable the GPU initially and proceed with training after the tokenizer has completed.


## ⚖️ License

This repository is licensed under the [Apache-2.0 License](LICENSE).

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/), [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene). Thanks for their wonderful works.

## 🔗 Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{GPT4Scene,
  title={GPT4Scene: Understand 3D Scenes from Videos with Vision-Language Models},
  author={Zhangyang Qi and Zhixiong Zhang and Ye Fang and Jiaqi Wang and Hengshuang Zhao},
  journal={arXiv:2501.01428},
  year={2025}
}
```

```bibtex
@article{VLNR1,
  title={VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning},
  author={Zhangyang Qi and Zhixiong Zhang and Yizhou Yu and Jiaqi Wang and Hengshuang Zhao},
  journal={arXiv:2506.17221},
  year={2025}
}
```