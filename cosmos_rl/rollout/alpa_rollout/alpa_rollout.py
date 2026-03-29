import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, GenerationConfig

from cosmos_rl.dispatcher.data import RLPayload
from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.policy.config import Config
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
import cosmos_rl.utils.util as util

from my_examples.alpamayo1_5 import helper
from my_examples.alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from my_examples.alpamayo1_5.models.token_utils import extract_text_tokens

ALPAMAYO_SOURCE_MODEL_ENV = "ALPAMAYO_SOURCE_MODEL_PATH"


@dataclass
class _SamplingConfig:
    n: int
    top_p: float
    top_k: int | None
    temperature: float
    repetition_penalty: float
    max_tokens: int
    stop_token_ids: list[int]
    include_stop_str_in_output: bool
    detokenize: bool
    prompt_logprobs: Any = None


@RolloutRegistry.register(rollout_type="alpa")
class AlpaRollout(RolloutBase):
    """Alpamayo rollout backend.

    This backend keeps the rollout side on the full Alpamayo checkpoint while the
    policy side can still train only the exported VLM submodule. Generated
    completions stay structured so that future dual-model training can reuse the
    action branch information without changing the framework message passing now.
    """

    def __init__(
        self,
        config: Config,
        parallel_dims: ParallelDims,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(config, parallel_dims, device, **kwargs)

    def post_init_hook(self, **kwargs):
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation
        self._model_param_map = None

        model_path = self.config.policy.model_name_or_path
        self.model_config = util.retry(AutoConfig.from_pretrained)(model_path)

        try:
            generation_config = util.retry(GenerationConfig.from_pretrained)(model_path)
            self.eos_token_ids = generation_config.eos_token_id
            if isinstance(self.eos_token_ids, int):
                self.eos_token_ids = [self.eos_token_ids]
        except Exception as exc:
            logger.warning(
                "[AlpaRollout] Failed to load generation config from %s: %s. "
                "Falling back to the tokenizer defaults.",
                model_path,
                exc,
            )
            self.eos_token_ids = [151645, 151643]

        self.rollout_engine = None

        self.val_sampling_params = _SamplingConfig(
            n=self.config.validation.n_generation,
            top_p=self.config.validation.top_p
            if self.config.validation.top_p is not None
            else self.config.rollout.sampling_config.top_p,
            top_k=self.config.validation.top_k
            if self.config.validation.top_k is not None
            else self.config.rollout.sampling_config.top_k,
            temperature=self.config.validation.temperature
            if self.config.validation.temperature is not None
            else self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.validation.repetition_penalty
            if self.config.validation.repetition_penalty is not None
            else self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.validation.max_response_length
            if self.config.validation.max_response_length is not None
            else self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
            prompt_logprobs=None,
        )
        self.sampling_params = _SamplingConfig(
            n=self.config.rollout.n_generation,
            top_p=self.config.rollout.sampling_config.top_p,
            top_k=self.config.rollout.sampling_config.top_k,
            temperature=self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
            prompt_logprobs=None,
        )

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        del quantization, seed, load_format, kwargs
        if self._engine_initialized:
            return

        source_model_path = os.environ.get(
            ALPAMAYO_SOURCE_MODEL_ENV,
            self.config.policy.model_name_or_path,
        )
        self.rollout_engine = Alpamayo1_5.from_pretrained(
            source_model_path,
            dtype=torch.bfloat16,
        ).to(self.device)
        self.rollout_engine.eval()
        self._engine_initialized = True
        logger.info(
            "[AlpaRollout] Engine initialized from %s (policy side still uses %s).",
            source_model_path,
            self.config.policy.model_name_or_path,
        )

    def _build_model_inputs(
        self,
        payload: RLPayload,
        data_packer: BaseDataPacker,
    ) -> Dict[str, Any]:
        rollout_input = data_packer.get_rollout_input(payload.prompt)
        messages = rollout_input["messages"]
        processor = helper.get_processor(self.rollout_engine.tokenizer)
        tokenized_data = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": tokenized_data,
            "ego_history_xyz": rollout_input["ego_history_xyz"],
            "ego_history_rot": rollout_input["ego_history_rot"],
        }
        return helper.to_device(model_inputs, self.device)

    @staticmethod
    def _tensor_batch_to_list(tensor: torch.Tensor) -> List[Any]:
        return tensor.detach().to("cpu", dtype=torch.float32).tolist()

    def _build_completion_records(
        self,
        vlm_outputs,
        sampled_action: torch.Tensor,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        text_outputs = extract_text_tokens(
            self.rollout_engine.tokenizer,
            vlm_outputs.sequences,
        )
        sampled_action_list = self._tensor_batch_to_list(sampled_action)
        sequence_ids = vlm_outputs.sequences.detach().cpu().tolist()

        completions: List[Dict[str, Any]] = []
        reasoning_list: List[str] = text_outputs.get("cot", [])
        meta_action_list: List[str] = text_outputs.get("meta_action", [])
        answer_list: List[str] = text_outputs.get("answer", [])

        total_generations = len(reasoning_list)
        for idx in range(total_generations):
            reasoning_text = reasoning_list[idx]
            completions.append(
                {
                    "reasoning": reasoning_text,
                    "reasoning_token_ids": self.rollout_engine.tokenizer(
                        reasoning_text,
                        add_special_tokens=False,
                    ).input_ids,
                    "meta_action_text": meta_action_list[idx]
                    if idx < len(meta_action_list)
                    else "",
                    "answer_text": answer_list[idx] if idx < len(answer_list) else "",
                    "action": sampled_action_list[idx],
                    "format": "alpamayo_reasoning_action_v1",
                }
            )

        extra_info = {
            "reasoning": reasoning_list,
            "reasoning_token_ids": [
                self.rollout_engine.tokenizer(
                    reasoning_text,
                    add_special_tokens=False,
                ).input_ids
                for reasoning_text in reasoning_list
            ],
            "meta_action_text": meta_action_list,
            "answer_text": answer_list,
            "sampled_action": sampled_action_list,
            "generated_token_ids": sequence_ids,
        }
        return completions, extra_info

    @torch.no_grad()
    def rollout_generation_single_payload(
        self,
        payload: RLPayload,
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        is_validation: bool = False,
    ) -> RolloutResult:
        model = self.get_engine()
        model_inputs = self._build_model_inputs(payload, data_packer)
        sampling_params = (
            self.val_sampling_params if is_validation else self.sampling_params
        )

        try:
            stream = torch.cuda.current_stream() if stream is None else stream
            with torch.cuda.stream(stream):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vlm_outputs, sampled_action = model.generate_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=sampling_params.top_p,
                        top_k=sampling_params.top_k,
                        temperature=sampling_params.temperature,
                        num_traj_samples=sampling_params.n,
                        max_generation_length=sampling_params.max_tokens,
                    )

            completions, extra_info = self._build_completion_records(
                vlm_outputs,
                sampled_action,
            )
            return RolloutResult(
                prompt=payload.prompt,
                completions=completions,
                completion_logprobs=None,
                completion_token_ids=None,
                cumulative_logprob=None,
                prompt_logprobs=None,
                prompt_token_ids=None,
                extra_info=extra_info,
            )
        except Exception as exc:
            logger.error("[AlpaRollout] Failed in rollout generation: %s", exc)
            import traceback

            traceback.print_exc()
            return RolloutResult(
                prompt=payload.prompt,
                completions=[],
                completion_logprobs=None,
                completion_token_ids=None,
                cumulative_logprob=None,
                prompt_logprobs=None,
                prompt_token_ids=None,
                extra_info={"error": str(exc)},
            )

    def rollout_generation(
        self,
        payloads: List[RLPayload] | List[Dict[str, torch.Tensor]],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        *args,
        **kwargs,
    ) -> List[RolloutResult] | List[Dict[str, torch.Tensor]]:
        del data_fetcher, args, kwargs
        responses: List[RolloutResult] = []
        for payload in payloads:
            responses.append(
                self.rollout_generation_single_payload(
                    payload,
                    stream,
                    data_packer,
                    is_validation,
                )
            )
        return responses

    def get_underlying_model(self) -> torch.nn.Module:
        """Expose only the VLM branch for weight sync with the policy model."""
        return self.get_engine().vlm

    def get_engine(self) -> Alpamayo1_5:
        if not self._engine_initialized:
            raise RuntimeError(
                "[AlpaRollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine
