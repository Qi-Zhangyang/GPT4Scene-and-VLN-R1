from huggingface_hub import snapshot_download
import os
import zipfile

## Download val dataset.
val_dataset = snapshot_download(
    repo_id="alexzyqi/GPT4Scene-All",
    repo_type="dataset",
    local_dir="./data/"
)

data_dir = "./data"
zip_files = ["images_2D.zip", "images_3D.zip"]

for zip_file in zip_files:
    zip_path = os.path.join(data_dir, zip_file)
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

## Download pretrained models.
pretrained_model_path = snapshot_download(
    repo_id="Qwen/Qwen2-VL-7B-Instruct",
    local_dir="./ckpts/models--Qwen--Qwen2-VL-7B-Instruct/"
)

## Download the validation annotations.
val_annotation = snapshot_download(
    repo_id="alexzyqi/GPT4Scene-Val-Annotation",
    repo_type="dataset",
    local_dir="./evaluate/annotation/"
)

## Download trained models.
trained_model_path = snapshot_download(
    repo_id="alexzyqi/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512",
    local_dir="./model_outputs/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512/"
)

