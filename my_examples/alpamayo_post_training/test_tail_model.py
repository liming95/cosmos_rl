import copy
import numpy as np
import torch
from pathlib import Path
from transformers import StoppingCriteriaList, AutoProcessor

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.base_model import TrajectoryFusionMixin
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

# Qwen3-VL Generation
from transformers import Qwen3VLForConditionalGeneration

# 自定义 stopping criteria
class StopAfterEOS:
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        return (input_ids[0, -1] == self.eos_token_id).item()


class ExportedModel(TrajectoryFusionMixin):
    def __init__(self, config=None):
        super().__init__()
        self.vlm = None
        self.tokenizer = None
        self.config = config or {
            "tokens_per_future_traj": 256,
            "traj_token_start_idx": 0,
            "traj_vocab_size": 512,
        }

    def infer(self):
        clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
        print(f"Loading dataset for clip_id: {clip_id}...")
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        print("Dataset loaded.")

        # 创建消息
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"]
        )

        # 加载 Qwen3-VLGeneration 模型
        exported_vlm_path = "/workspace/models/exported_vlm"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            exported_vlm_path,
            torch_dtype=torch.bfloat16
        ).to("cuda")

        self.vlm = model

        # self.tokenizer = Auto.from_pretrained(exported_vlm_path)
        # processor = helper.get_processor(model.tokenizer)

        processor = AutoProcessor.from_pretrained(exported_vlm_path)
        tokenizer = processor.tokenizer

        self.tokenizer = tokenizer

        # 将消息转换为 tokenized_data
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }

        model_inputs = helper.to_device(model_inputs, "cuda")

        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self.generate(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
            )

        print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        print("minADE:", min_ade, "meters")
        if min_ade >= 1.0:
            print(f"WARNING: minADE ({min_ade:.2f}m) is above 1.0m. Model sampling can be stochastic.")

    def generate(
        self,
        data: dict,
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict | None = None,
        *args,
        **kwargs,
    ):
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."

        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")

        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        max_generation_length = kwargs.get(
            "max_generation_length", self.config["tokens_per_future_traj"]
        )
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])

        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            **tokenized_data,
            return_extra=True,
        )

        pred_xyz = vlm_outputs.get("pred_xyz", torch.zeros((B, 1, max_generation_length, 3), device=device))
        pred_rot = vlm_outputs.get("pred_rot", torch.zeros((B, 1, max_generation_length, 3, 3), device=device))
        extra = vlm_outputs.get("extra", {"cot": ["example_chain_of_causation"] * B})

        return pred_xyz, pred_rot, extra

if __name__== "__main__":
    model = ExportedModel()
    model.infer()