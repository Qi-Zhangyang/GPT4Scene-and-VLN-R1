import os
import torch
from tqdm import tqdm
from typing import List
from pathlib import Path
from habitat import logger
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.recollection_dataset import TeacherRecollectionDataset
import time

@baseline_registry.register_trainer(name="iter_loader_runner")
class IterProgressRunner(BaseVLNCETrainer):
    """带进度显示的迭代式数据遍历器"""
    
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config: Config):
        super().__init__(config)
        # 状态跟踪
        self.epoch = 0
        self.global_batch = 0
        self.total_batches = 0
        # 进度显示
        self.pbar = None
        # 数据集相关
        self.dataset = None
        self.data_iter = None
        self.dataset_state = None

    def _make_dirs(self):
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)

    def save_state(self):
        """保存训练状态"""
        state = {
            "epoch": self.epoch,
            "global_batch": self.global_batch,
            "dataset_state": self.dataset.get_state() if self.dataset else None
        }
        torch.save(state, Path(self.config.CHECKPOINT_FOLDER)/"iter_state.pth")

    def load_state(self):
        """加载训练状态"""
        state_path = Path(self.config.CHECKPOINT_FOLDER)/"iter_state.pth"
        if state_path.exists():
            state = torch.load(state_path)
            self.epoch = state["epoch"]
            self.global_batch = state["global_batch"]
            self.dataset_state = state["dataset_state"]
            logger.info(f"恢复至 Epoch {self.epoch+1} 第 {self.global_batch} 个全局批次")

    def setup_environment(self):
        """初始化数据集环境"""
        self.dataset = TeacherRecollectionDataset(self.config)
        if self.dataset_state:
            self.dataset.set_state(self.dataset_state)
        # 计算批次总数
        self.total_batches = self.dataset.length // self.dataset.batch_size

def train(self) -> None:
    self._make_dirs()
    self.load_state()
    self.setup_environment()

    try:
        # 主训练循环
        while self.epoch < self.config.IL.epochs:
            self.data_iter = iter(self.dataset)
            
            # 计算当前epoch的批次参数
            batches_in_epoch = self.dataset.length // self.dataset.batch_size
            initial_batch = self.global_batch % batches_in_epoch  # 当前epoch已处理的批次

            # 初始化进度条（每个epoch独立）
            self.pbar = tqdm(
                total=batches_in_epoch,
                initial=initial_batch,
                desc=f"Epoch {self.epoch+1}/{self.config.IL.epochs}",
                dynamic_ncols=True,
                unit='batch',
                postfix={
                    'global_batch': self.global_batch,
                    'speed': '0.0 batches/s'
                }
            )

            try:
                while True:
                    # 推进迭代器
                    start_time = time.time()
                    next(self.data_iter)
                    
                    # 更新计数器
                    self.global_batch += 1
                    elapsed = time.time() - start_time

                    # 正确更新进度条（仅更新当前epoch进度）
                    self.pbar.update(1)  # 每次+1
                    self.pbar.set_postfix({
                        'global_batch': self.global_batch,
                        'speed': f'{1/elapsed:.1f} batches/s',
                        'epoch_batch': f"{self.pbar.n}/{batches_in_epoch}"
                    })

                    # 定期保存状态
                    if self.global_batch % self.config.IL.save_interval == 0:
                        self.save_state()

            except StopIteration:
                # 完成当前epoch
                self.epoch += 1
                self.pbar.close()
                self.save_state()
                if self.epoch < self.config.IL.epochs:
                    self.setup_environment()  # 准备下一epoch

    except KeyboardInterrupt:
        logger.info("\n检测到中断，保存当前状态...")
        self.pbar.close()
        self.save_state()
        raise
    finally:
        if self.pbar:
            self.pbar.close()

# ===== 数据集状态方法保持不变 =====