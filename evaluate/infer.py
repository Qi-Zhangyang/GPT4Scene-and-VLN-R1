import os
import json
import torch
from tqdm import tqdm
from val_dataset import create_dataset, create_loader, collate_fn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils.config_utils import setup_main

def inference(model, val_loader, device, config):
    eval_name = val_loader.dataset.datasets[0].dataset.dataset_name
    output_dir = os.path.join(config.output_dir, f"{model}_{eval_name}")
    predictions_file = os.path.join(output_dir, f"predictions_chunk_{config.chunk_idx}.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(predictions_file):
        with open(predictions_file, "r") as f:
            all_preds = json.load(f)
        processed_qids = {item["qid"] for item in all_preds}
        print(f"Loaded {len(all_preds)} existing predictions from {predictions_file}.")
    else:
        all_preds = []
        processed_qids = set()

    save_interval = config.save_interval
    batch_count = 0

    if model == 'qwenvl':
        print(f"Loading model from {config.model_path}")
        vmodel = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_path, torch_dtype="auto", attn_implementation="flash_attention_2", device_map=None
        )
        vmodel.eval()
        vmodel.to(device)

        processor = AutoProcessor.from_pretrained(config.model_path)
        processor.tokenizer.padding_side = "left"

    with tqdm(total=len(val_loader), desc=f"Inference {eval_name}") as pbar:
        for batch in val_loader:
            unprocessed_indices = [
                i for i, qid in enumerate(batch["qid"]) if qid not in processed_qids
            ]

            if not unprocessed_indices:
                pbar.update(1)
                continue

            batch_size = len(unprocessed_indices)
            unprocessed_batch = {k: [v[i] for i in unprocessed_indices] for k, v in batch.items()}

            for k in unprocessed_batch.keys():
                if isinstance(unprocessed_batch[k], torch.Tensor):
                    unprocessed_batch[k] = unprocessed_batch[k].to(device)

            if model == 'qwenvl':
                from qwen2_vl import get_response_qwen as get_response
                pred = get_response(vmodel, processor, unprocessed_batch['custom_prompt'], unprocessed_batch['bev_mf_paths'], config, eval_name)

            for i, index in enumerate(unprocessed_indices):
                all_preds.append({
                    "scene_id": batch["scene_id"][index],
                    "qid": batch["qid"][index],
                    "gt_id": int(batch["obj_ids"][index]),
                    "pred_id": int(batch["pred_ids"][index]),
                    "prompt": batch["custom_prompt"][index],
                    "pred": pred[i],
                    "ref_captions": batch["ref_captions"][index],
                    "type_info": batch["type_infos"][index],
                })
                processed_qids.add(batch["qid"][index])

            batch_count += 1
            pbar.update(1)

            if batch_count % save_interval == 0:
                with open(predictions_file, "w") as f:
                    json.dump(all_preds, f, indent=4)
                print(f"Intermediate results saved at batch {batch_count} to {predictions_file}")

    with open(predictions_file, "w") as f:
        json.dump(all_preds, f, indent=4)
    print(f"Final results saved to {predictions_file}")

def main():
    config = setup_main()
    device = torch.device(config.device)

    val_loaders = create_loader(
        create_dataset(config),
        [None],
        batch_size=[config.batch_size],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[collate_fn],
    )

    for val_loader in val_loaders:
        inference(config.model, val_loader, device, config)

if __name__ == "__main__":
    main()
