import logging
from torch.utils.data import Dataset
import torch
import json
import torch
from torch.utils.data import ConcatDataset, DataLoader


logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def __init__(self, 
                 anno_file, 
                 config):
        
        self.media_type = "no_point_cloud"
        self.anno = json.load(open(anno_file, 'r'))
        self.scene_anno = json.load(open(config.scene_anno, 'r'))
        # self.img_2d_base_path = config.img_2d_base_path
        # self.img_bev_base_path = config.img_bev_base_path

        self.paths = []
        for entry in self.anno:
            scene_id = entry["scene_id"]

            # img_2d_path = os.path.join(self.img_2d_base_path, f"{scene_id}_composite.jpg") if self.img_2d_base_path else None
            # img_bev_path = os.path.join(self.img_bev_base_path, scene_id, f"{scene_id}_3D_with_text.jpg") if self.img_bev_base_path else None

            bev_mf_paths = self.scene_anno[scene_id]
            self.paths.append((scene_id, bev_mf_paths))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index):
        scene_id, bev_mf_paths = self.paths[index]
        return scene_id, bev_mf_paths

class ValDataset(BaseDataset):

    def __init__(self, anno_file, dataset_name, config, **kwargs):
        super().__init__(anno_file=anno_file, config=config)
        self.dataset_name = dataset_name
        self.anno = json.load(open(anno_file, 'r'))

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, bev_mf_paths = self.get_anno(index)
        obj_id = int(self.anno[index].get('obj_id', 0))
        pred_id = int(self.anno[index].get('pred_id', 0))
        type_info = int(self.anno[index].get('sqa_type', 0))
        if 'sqa_type' in self.anno[index]:
            type_info = self.anno[index]['sqa_type']
        elif 'eval_type' in self.anno[index]:
            type_info = self.anno[index]['eval_type'] 
        elif 'type_info' in self.anno[index]:
            type_info = self.anno[index]['type_info']
        if 'prompt' in self.anno[index]:
            prompt = self.anno[index]["prompt"]
        ref_captions = self.anno[index]["ref_captions"].copy() if "ref_captions" in self.anno[index] else []
        qid = self.anno[index]["qid"] if "qid" in self.anno[index] else 0
        
        return prompt, ref_captions, scene_id, bev_mf_paths, qid, obj_id, pred_id, type_info




def create_dataset(config):
    val_files = {}
    for val_name in config.val_tag.split('#'):
        if val_name not in config.val_file_dict:
            raise NotImplementedError(f"{val_name} not found in val_file_dict")
        val_files[val_name] = config.val_file_dict[val_name]

    subset_val_datasets = []
    for k, v in val_files.items():
        datasets = []
        if type(v[0]) != list:
            v = [v]
        for val_file in v:
            dataset = ValDataset(anno_file=val_file, dataset_name=k, config=config)
            total_size = len(dataset)
            chunk_size = total_size // config.num_chunks
            start_idx = config.chunk_idx * chunk_size
            end_idx = total_size if config.chunk_idx == config.num_chunks - 1 else (config.chunk_idx + 1) * chunk_size
            subset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
            datasets.append(subset)
        subset_val_datasets.append(ConcatDataset(datasets))

    return subset_val_datasets



def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data


def collate_fn(batch):
    prompts, ref_captions, scene_ids, bev_mf_paths, qids, obj_id, pred_ids, type_infos = zip(*batch)
    return {
        "custom_prompt": prompts,
        "ref_captions": ref_captions,
        "scene_id": scene_ids,
        "bev_mf_paths": bev_mf_paths,
        "qid": qids,
        "obj_ids": obj_id,
        "pred_ids": pred_ids,
        "type_infos": type_infos
    }