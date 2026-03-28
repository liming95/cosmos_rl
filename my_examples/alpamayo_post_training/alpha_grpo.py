from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Optional

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
from my_examples.alpamayo1_5 import helper
from my_examples.alpamayo1_5.load_physical_aiavdataset import (
    load_physical_aiavdataset,
)
from my_examples.alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from my_examples.alpamayo1_5.models.base_model import (
    TRAJ_TOKEN,
    tokenize_history_trajectory,
)

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


def _export_vlm_if_needed(
    source_model_path: str,
    export_dir: Path,
) -> None:
    config_file = export_dir / "config.json"
    tokenizer_file = export_dir / "tokenizer_config.json"
    if config_file.exists() and tokenizer_file.exists():
        logger.info(f"[AlphaGRPO] Reusing exported VLM at {export_dir}")
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[AlphaGRPO] Exporting VLM submodule from {source_model_path} to {export_dir}")

    model = Alpamayo1_5.from_pretrained(source_model_path, dtype=torch.bfloat16)
    model.vlm.save_pretrained(export_dir)
    model.tokenizer.save_pretrained(export_dir)
    processor = helper.get_processor(model.tokenizer)
    processor.save_pretrained(export_dir)
    del model


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
        self.dataset = _resolve_dataset_records(config.train.train_policy.dataset)
        logger.info(f"[AlphaDataset] Loaded {len(self.dataset)} training records")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        payload = dict(self.dataset[idx])
        if "clip_id" not in payload:
            raise KeyError(f"Dataset record at idx={idx} does not contain `clip_id`")
        return payload

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
        source_model_path = os.environ.get(
            ALPAMAYO_SOURCE_MODEL_ENV,
            config.policy.model_name_or_path,
        )
        self.alpamayo_hf_config = AutoConfig.from_pretrained(
            source_model_path,
            trust_remote_code=True,
        )
        self.hist_token_start_idx = self.alpamayo_hf_config.traj_token_start_idx
        self.hist_traj_tokenizer = None
        if getattr(self.alpamayo_hf_config, "hist_traj_tokenizer_cfg", None) is not None:
            self.hist_traj_tokenizer = hyu.instantiate(
                self.alpamayo_hf_config.hist_traj_tokenizer_cfg
            )
            if getattr(self.alpamayo_hf_config, "traj_tokenizer_cfg", None) is not None:
                self.hist_token_start_idx += self.alpamayo_hf_config.traj_vocab_size
        elif getattr(self.alpamayo_hf_config, "traj_tokenizer_cfg", None) is not None:
            self.hist_traj_tokenizer = hyu.instantiate(
                self.alpamayo_hf_config.traj_tokenizer_cfg,
                load_weights=False,
            )

    def _build_messages(self, item: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        clip_id = item["clip_id"]
        t0_us = int(item.get("t0_us", 5_100_000))
        num_history_steps = int(item.get("num_history_steps", 16))
        num_future_steps = int(item.get("num_future_steps", 64))
        time_step = float(item.get("time_step", 0.1))
        num_frames = int(item.get("num_frames", 4))

        data = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=t0_us,
            num_history_steps=num_history_steps,
            num_future_steps=num_future_steps,
            time_step=time_step,
            num_frames=num_frames,
        )

        question = item.get("question")
        if question:
            messages = helper.create_vqa_message(
                frames=data["image_frames"].flatten(0, 1),
                camera_indices=data["camera_indices"],
                question=question,
                num_frames_per_camera=num_frames,
            )
        else:
            messages = helper.create_message(
                frames=data["image_frames"].flatten(0, 1),
                camera_indices=data["camera_indices"],
                num_frames_per_camera=num_frames,
                nav_text=item.get("nav_text"),
                use_nav_prompt=bool(item.get("use_nav_prompt", False)),
            )

        return messages, data

    def _combine_image_traj(self, input_data: str, traj_data_vlm: dict[str, Any]) -> str:
        """Char-level version of `fuse_traj_tokens`.

        `fuse_traj_tokens` replaces the tokenizer id for `<|traj_history|>` with the
        discretized history trajectory token ids. Here we do the same replacement one
        level earlier, directly on the prompt string, by swapping the repeated
        `<|traj_history|>` placeholders with concrete `<i...>` token strings.
        """
        if not isinstance(input_data, str):
            raise TypeError(
                f"_combine_image_traj expects a prompt string, but got {type(input_data)}"
            )
        if self.hist_traj_tokenizer is None:
            return input_data
        if (
            traj_data_vlm is None
            or traj_data_vlm.get("ego_history_xyz") is None
            or traj_data_vlm.get("ego_history_rot") is None
        ):
            return input_data

        hist_idx = tokenize_history_trajectory(
            self.hist_traj_tokenizer,
            traj_data_vlm,
            self.hist_token_start_idx,
        )
        if hist_idx.shape[0] != 1:
            raise ValueError(
                f"_combine_image_traj currently expects batch size 1, got {hist_idx.shape[0]}"
            )

        hist_token_strings = self.underlying_data_packer.tokenizer.convert_ids_to_tokens(
            hist_idx[0].tolist()
        )
        fused_hist_str = "".join(hist_token_strings)

        start_token = TRAJ_TOKEN["history_start"]
        pad_token = TRAJ_TOKEN["history"]
        end_token = TRAJ_TOKEN["history_end"]
        start_idx = input_data.find(start_token)
        end_idx = input_data.find(end_token)
        if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
            return input_data

        content_start = start_idx + len(start_token)
        inner = input_data[content_start:end_idx]
        if pad_token not in inner:
            return input_data

        return input_data[:content_start] + fused_hist_str + input_data[end_idx:]
        
    def _build_vllm_prompt(self, prompt: str, data: dict[str, Any]) -> str:
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        _, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        return self._combine_image_traj(prompt, traj_data_vlm)

    def get_rollout_input(self, item: Any) -> Any:
        """
        Convert one Alpamayo sample descriptor into the rollout-engine input format.

        Output format is delegated to `Qwen3_VL_DataPacker`, so the final return value
        is still the standard:
        {
            "prompt": ...,
            "multi_modal_data": {"image": ...},
        }
        """
        assert isinstance(item, dict), (
            f"AlphaDataPacker expects dict items from dataset, but got {type(item)}"
        )
        messages, data = self._build_messages(item)
        vllm_inputs = self.underlying_data_packer.get_rollout_input(messages)
        vllm_inputs = {k: v for k, v in vllm_inputs.items() if k != "prompt"}

        prompt = self.underlying_data_packer.hf_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        vllm_inputs["prompt"] = self._build_vllm_prompt(prompt, data)
        return vllm_inputs

    def rollout_collate_fn(self, items: list[Any]) -> Any:
        return self.underlying_data_packer.rollout_collate_fn(items)

    def get_policy_input(
        self, item: Any, rollout_output: str, n_ignore_prefix_tokens: int = 0
    ) -> Any:
        messages, _ = self._build_messages(item)
        return self.underlying_data_packer.get_policy_input(
            messages,
            rollout_output,
            n_ignore_prefix_tokens,
            add_generation_prompt=False,
        )

    def policy_compute_max_len(self, processed_samples: list[Any]) -> int:
        return self.underlying_data_packer.policy_compute_max_len(processed_samples)

    def policy_collate_fn(
        self, processed_samples: list[Any], computed_max_len: int
    ) -> dict[str, Any]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    config_path = Path(args.config).resolve()
    CONFIG_BASE_DIR = config_path.parent
    with open(config_path, encoding="utf-8") as f:
        config_dict = toml.load(f)
    config = Config.from_dict(config_dict)

    source_model_path = os.environ.get(
        ALPAMAYO_SOURCE_MODEL_ENV,
        config.policy.model_name_or_path,
    )
    os.environ[ALPAMAYO_SOURCE_MODEL_ENV] = source_model_path
    export_dir = (CONFIG_BASE_DIR.parent / "exported_vlm").resolve()
    _export_vlm_if_needed(source_model_path, export_dir)
    config_dict["policy"]["model_name_or_path"] = str(export_dir)
    runtime_config_path = CONFIG_BASE_DIR / "_runtime_rl.toml"
    with open(runtime_config_path, "w", encoding="utf-8") as f:
        toml.dump(config_dict, f)
    args.config = str(runtime_config_path)

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
        args=args,
    )
