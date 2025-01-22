# ========================= annotations ==========================

anno_root = "evaluate/annotation"  # annotation dir
segmentor = "mask3d"
version = ""

val_tag = 'scanqa'

val_file_dict = {
    'scanqa': f"{anno_root}/scanqa_val.json",
    'scanrefer': f"{anno_root}/scanrefer_{segmentor}_val{version}.json",
    'scan2cap': f"{anno_root}/scan2cap_{segmentor}_val{version}.json",
    'sqa3d': f"{anno_root}/sqa3d_val.json",
    'multi3dref': f"{anno_root}/multi3dref_{segmentor}_val{version}.json",
}

seg_val_attr_file = f"{anno_root}/scannet_{segmentor}_val_attributes.pt"

scene_anno = f"{anno_root}/selected_images_mark_3D_val_32.json"

# ========================= running ==========================
evaluate = True
num_workers = 32
batch_size = 4
save_interval = 3

dist_url = "env://"
device = "cuda"
num_chunks = 1
chunk_idx = 0

output_dir = "outputs/tmp"  # output dir

calculate_score_tag = 'scanqa#sqa3d#scan2cap#scanrefer#multi3dref'


# ========================= data & model ==========================
model = 'qwenvl'
apikey = None
model_path = 'models_output/GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512'
base_model = False

resolution = 512
multi_frames = True