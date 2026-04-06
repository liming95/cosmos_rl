from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Optional

from cosmos_rl.policy.model.wfm import tokenizer
import torch
import toml
from torch.utils.data import Dataset
from transformers import AutoConfig
import hydra.utils as hyu

from cosmos_rl.dispatcher.data.packer import DataPacker, Qwen3_VL_DataPacker
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.logging import logger
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import (
    load_physical_aiavdataset,
)
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.base_model import (
    TRAJ_TOKEN,
    tokenize_history_trajectory,
)

from transformers import AutoProcessor, AutoTokenizer

ALPAMAYO_SOURCE_MODEL_ENV = "ALPAMAYO_SOURCE_MODEL_PATH"
CONFIG_BASE_DIR: Path | None = None


def _load_manifest(path: str) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if manifest_path.suffix == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if manifest_path.suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        raise ValueError(f"Unsupported json structure in {manifest_path}")

    if manifest_path.suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(
        f"Unsupported manifest suffix {manifest_path.suffix}, expected .jsonl/.json/.csv"
    )


def _resolve_dataset_records(dataset_cfg) -> list[dict[str, Any]]:
    dataset_name = getattr(dataset_cfg, "name", None)
    if not dataset_name:
        raise ValueError("dataset.name must point to a local manifest file for Alpamayo")
    dataset_path = Path(dataset_name)
    if not dataset_path.exists() and CONFIG_BASE_DIR is not None:
        dataset_path = (CONFIG_BASE_DIR / dataset_name).resolve()
    return _load_manifest(str(dataset_path))


# def _export_vlm_if_needed(
#     source_model_path: str,
#     export_dir: Path,
# ) -> None:
#     config_file = export_dir / "config.json"
#     tokenizer_file = export_dir / "tokenizer_config.json"
#     if config_file.exists() and tokenizer_file.exists():
#         logger.info(f"[AlphaGRPO] Reusing exported VLM at {export_dir}")
#         return

#     export_dir.mkdir(parents=True, exist_ok=True)
#     logger.info(f"[AlphaGRPO] Exporting VLM submodule from {source_model_path} to {export_dir}")

#     model = Alpamayo1_5.from_pretrained(source_model_path, dtype=torch.bfloat16)
#     vlm = model.vlm

#     import math
#     old_vocab_size = vlm.config.vocab_size
#     new_vocab_size = math.ceil(old_vocab_size / 8) * 8

#     if new_vocab_size != old_vocab_size:
#         logger.info(f"[AlphaGRPO] Resize vocab: {old_vocab_size} -> {new_vocab_size}")
#         vlm.resize_token_embeddings(new_vocab_size)
#         vlm.config.vocab_size = new_vocab_size

#     vlm.save_pretrained(export_dir)

#     # tokenizer = AutoTokenizer.from_pretrained(
#     #     source_model_path,
#     #     fix_mistral_regex=True
#     # )
#     # tokenizer = AutoTokenizer.from_pretrained(
#     #     export_dir,
#     #     fix_mistral_regex=True
#     # )
#     tokenizer = model.tokenizer
#     if new_vocab_size != old_vocab_size:
#         logger.info("tokenizer")
#         tokenizer.model_max_length = new_vocab_size
#     # model.tokenizer.save_pretrained(export_dir)
#     tokenizer.save_pretrained(export_dir)
#     processor = helper.get_processor(tokenizer)
#     processor.save_pretrained(export_dir)
#     del model

def _export_vlm_if_needed(
    source_model_path: str,
    export_dir: Path,
) -> None:
    import math
    from transformers import AutoTokenizer

    config_file = export_dir / "config.json"
    tokenizer_file = export_dir / "tokenizer_config.json"

    if config_file.exists() and tokenizer_file.exists():
        logger.info(f"[AlphaGRPO] Reusing exported VLM at {export_dir}")
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[AlphaGRPO] Exporting VLM submodule from {source_model_path} to {export_dir}")

    # 1. 加载模型
    model = Alpamayo1_5.from_pretrained(source_model_path, dtype=torch.bfloat16)
    vlm = model.vlm

    # tokenizer = model.tokenizer
    # # vlm.config.vocab_size = 152064
    # logger.info(f"[my test] vocab_size : {len(tokenizer)}")


    # 2. 调整 vocab_size 为 8 的倍数
    old_vocab_size = len(tokenizer) #vlm.config.vocab_size
    new_vocab_size = math.ceil(old_vocab_size / 8) * 8

    if new_vocab_size != old_vocab_size:
        logger.info(f"[AlphaGRPO] Resize vocab: {old_vocab_size} -> {new_vocab_size}")
        # token embeddings and lm_head
        # vlm.resize_token_embeddings(new_vocab_size)
        vlm.config.vocab_size = new_vocab_size

        # sync tokenizer
        tokenizer = model.tokenizer
        logger.info(f"old tokenizer size: {tokenizer.model_max_length}, {len(tokenizer)}")
        # tokenizer.model_max_length = new_vocab_size
        # if hasattr(tokenizer, "vocab_size"):
        #     tokenizer.vocab_size = new_vocab_size
        # additional_tokens = new_vocab_size -len(tokenizer)
        # logger.info(f"addtional tokens: {new_vocab_size} - {len(tokenizer)} = {additional_tokens}")
        # if additional_tokens > 0:
        #     additional_tokens_list = [f"<extra_id_{i}>" for i in range(additional_tokens)]
        #     tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens_list})
            # tokenizer.add_special_tokens({"additional_special_tokens": ["<pad>"] * additional_tokens})
        logger.info(f"old tokenizer size: {tokenizer.model_max_length}, {len(tokenizer)}")
    else:
        tokenizer = model.tokenizer

    # 3.
    vlm.save_pretrained(export_dir)

    # 4.
    tokenizer.save_pretrained(export_dir)
    processor = helper.get_processor(tokenizer)
    processor.save_pretrained(export_dir)

    # 5. check
    # lm_weight_shape = vlm.lm_head.weight.shape
    # if lm_weight_shape[0] != len(tokenizer):
    #     raise RuntimeError(
    #         f"[AlphaGRPO] lm_head.weight shape {lm_weight_shape} "
    #         f"does not match tokenizer vocab_size {tokenizer.vocab_size}!"
    #     )
    # else:
    #     logger.info(f"[AlphaGRPO] Check passed: lm_head.weight {lm_weight_shape} matches vocab_size {tokenizer.vocab_size}")

    del model
    logger.info(f"[AlphaGRPO] Export completed at {export_dir}")


class AlphaDataset(Dataset):
    """Alpamayo GRPO dataset.

    Unlike the HF examples, each item here is just a lightweight descriptor of one
    physical_ai_av sample. The heavy multimodal reconstruction happens later in
    `AlphaDataPacker.get_rollout_input()`.

    Expected manifest fields:
    - clip_id: str
    - t0_us: int, optional
    - question: str, optional
    - reference_answer: str, optional
    - num_history_steps / num_future_steps / num_frames: optional
    """

    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.config = config
        dataset_config_list = _resolve_dataset_records(config.train.train_policy.dataset)
        logger.info(f"[AlphaDataset] Loaded {len(dataset_config_list)} training records")

        data_list = []
        for idx, record in enumerate(dataset_config_list):
            if "clip_id" not in record:
                raise KeyError(f"Dataset record at idx={idx} is missing required `clip_id` field")
            data = load_physical_aiavdataset(
                clip_id=record["clip_id"], t0_us=record.get("t0_us", 5_100_000), num_history_steps=16, num_future_steps=64, time_step=0.1, num_frames=4
            )
            data_list.append(data)
        self.dataset = data_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        payload = self.dataset[idx]
        
        image_frames = payload.get("image_frames")
        camera_indices = payload.get("camera_indices")
        ego_history_xyz = payload.get("ego_history_xyz")
        ego_history_rot = payload.get("ego_history_rot")
        
        # tokenizer
        model_name_or_path = os.environ.get(ALPAMAYO_SOURCE_MODEL_ENV, self.config.policy.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            fix_mistral_regex=True,
        )

        # images context
        # traj context for ego
        # nav context
        # prompt context
        messages = helper.create_message_with_ego(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            ego_history_xyz=ego_history_xyz,
            ego_history_rot=ego_history_rot,
            tokenizer=tokenizer
        )

        
        # build chat input fields for reward functions that may need them
        # question = payload.get("question")
        
        return messages   #conversations

    def get_reference_answer(self, idx: int) -> Any:
        payload = self.dataset[idx]
        return payload.get("reference_answer", "")


class AlphaValDataset(AlphaDataset):
    def setup(self, config: Config, *args, **kwargs):
        if not config.validation.enable:
            logger.warning(
                "Validation is not enabled in the config. Skipping setup for AlphaValDataset."
            )
            return

        self.config = config
        self.dataset = _resolve_dataset_records(config.validation.dataset)
        logger.info(f"[AlphaValDataset] Loaded {len(self.dataset)} validation records")


class AlphaDataPacker(DataPacker):
    """Wrap Qwen3 VL packer, but replace data source with physical_ai_av.

    The key difference from `cosmos_grpo.py` is that the input is not a HF image/video
    path. We reconstruct the Alpamayo VLM message from physical_ai_av data here, then
    delegate the final rollout-engine formatting to `Qwen3_VL_DataPacker`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying_data_packer = Qwen3_VL_DataPacker()

    def setup(self, config: Config, *args, **kwargs):
        super().setup(config, *args, **kwargs)
        self.underlying_data_packer.setup(config, *args, **kwargs)
        
    def get_rollout_input(self, item: Any) -> Any:
        """
        Convert dataset item into what rollout engine (e.g. vllm) expects
        """
        return self.underlying_data_packer.get_rollout_input(item)

    def rollout_collate_fn(self, items: list[Any]) -> Any:
        """
        Collate the rollout inputs into a mini-batch for rollout engine
        """
        return self.underlying_data_packer.rollout_collate_fn(items)

    def get_policy_input(
        self, item: Any, rollout_output: str, n_ignore_prefix_tokens: int = 0
    ) -> Any:
        """
        Process samples & rollout output before collating them into a mini-batch
        """
        return self.underlying_data_packer.get_policy_input(
            item, rollout_output, n_ignore_prefix_tokens
        )

    def policy_compute_max_len(self, processed_samples: list[Any]) -> int:
        """
        Compute the maximum sequence length of the mini-batch
        """
        return self.underlying_data_packer.policy_compute_max_len(processed_samples)

    def policy_collate_fn(
        self, processed_samples: list[Any], computed_max_len: int
    ) -> dict[str, Any]:
        """
        Collate the mini-batch into the kwargs required by the policy model
        """
        return self.underlying_data_packer.policy_collate_fn(
            processed_samples, computed_max_len
        )


def fake_reward_fn(
    to_be_evaluated: str, reference: Optional[str] = None, *args, **kwargs
) -> float:
    del reference, args, kwargs
    if not isinstance(to_be_evaluated, str):
        return 0.1
    # Deterministic fake reward to keep the GRPO pipeline numerically alive.
    return 0.1 + min(len(to_be_evaluated.strip()) / 1024.0, 0.9)


def get_dataset(config: CosmosConfig) -> Dataset:
    return AlphaDataset()


def get_val_dataset(config: CosmosConfig) -> Dataset:
    return AlphaValDataset() if config.validation.enable else None

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker, worker_entry_parser
from cosmos_rl.utils.logging import logger
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    # parser = worker_entry_parser()
    # args = parser.parse_args()
    # config_path = Path(args.config).resolve()
    # CONFIG_BASE_DIR = config_path.parent
    # with open(config_path, encoding="utf-8") as f:
    #     config_dict = toml.load(f)
    # config = Config.from_dict(config_dict)

    # source_model_path = os.environ.get(
    #     ALPAMAYO_SOURCE_MODEL_ENV,
    #     config.policy.model_name_or_path,
    # )
    # os.environ[ALPAMAYO_SOURCE_MODEL_ENV] = source_model_path
    # export_dir = (CONFIG_BASE_DIR.parent / "exported_vlm").resolve()
    # _export_vlm_if_needed(source_model_path, export_dir)
    # config_dict["policy"]["model_name_or_path"] = str(export_dir)
    # runtime_config_path = CONFIG_BASE_DIR / "_runtime_rl.toml"
    # with open(runtime_config_path, "w", encoding="utf-8") as f:
    #     toml.dump(config_dict, f)
    # args.config = str(runtime_config_path)
    # logger.info(f"[my test] runtime_config_path: {runtime_config_path}, new model:{export_dir}")

    def dataset_factory(config: CosmosConfig) -> Dataset:
        return get_dataset(config)

    def val_dataset_factory(config: CosmosConfig) -> Dataset:
        return get_val_dataset(config)

    launch_worker(
        dataset=dataset_factory,
        reward_fns=[fake_reward_fn],
        data_packer=AlphaDataPacker(),
        val_dataset=val_dataset_factory,
        val_reward_fns=[fake_reward_fn],
        val_data_packer=AlphaDataPacker(),
        # args=args,
    )
