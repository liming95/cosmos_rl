from pathlib import Path
import torch
import math
import logging

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import copy

import math
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def _export_vlm_if_needed(source_model_path: str, export_dir: Path, keep_ratio: float = 0.1) -> None:
    """
    Export the VLM submodule from an Alpamayo1_5 model to a given directory,
    reducing the number of language_model.layers by keep_ratio, keeping all other layers unchanged.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[AlphaGRPO] Exporting VLM submodule from {source_model_path} to {export_dir}")

    # 1. Load full model
    model = Alpamayo1_5.from_pretrained(source_model_path, dtype=torch.bfloat16)
    vlm = model.vlm
    tokenizer = model.tokenizer

    # 2. Adjust vocab size to be divisible by 16 (if needed)
    old_vocab_size = len(tokenizer)
    new_vocab_size = math.ceil(old_vocab_size / 16) * 16
    if new_vocab_size != old_vocab_size:
        additional_tokens = new_vocab_size - old_vocab_size
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<extra_id_{i}>" for i in range(additional_tokens)]})
        vlm.resize_token_embeddings(new_vocab_size)
        vlm.config.vocab_size = new_vocab_size
        logger.info(f"[AlphaGRPO] Resized vocab: {old_vocab_size} -> {new_vocab_size}")

    # 3. Determine layers to keep
    layers = vlm.model.language_model.layers
    original_layers = len(layers)
    keep_layers = max(1, int(original_layers * keep_ratio))
    config = copy.deepcopy(vlm.config)
    config.text_config.num_hidden_layers = keep_layers  # 更新配置中的层数
    logger.info(f"[AlphaGRPO] Reducing language_model layers: {original_layers} -> {keep_layers}")

    # 4. Create a new VLM instance (same config)
    new_vlm = type(vlm)(config)

    # 5. Copy state_dict
    orig_state_dict = vlm.state_dict()
    new_state_dict = {}
    for k, v in orig_state_dict.items():
        # 裁剪 language_model.layers
        if k.startswith("model.language_model.layers."):
            layer_id = int(k.split("model.language_model.layers.")[1].split(".")[0])
            if layer_id < keep_layers:
                new_state_dict[k] = v
        else:
            # 其他层直接保留原权重
            new_state_dict[k] = v

    missing, unexpected = new_vlm.load_state_dict(new_state_dict, strict=False)
    logger.info(f"[AlphaGRPO] Missing keys after pruning: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # 6. 裁剪 ModuleList 以匹配结构
    # new_vlm.model.language_model.layers = torch.nn.ModuleList(list(new_vlm.model.language_model.layers[:keep_layers]))

    # 7. 确保 embedding 与 tokenizer 对齐
    # new_vlm.resize_token_embeddings(len(tokenizer))

    # 8. Prepare processor
    processor = helper.get_processor(tokenizer)

    # 9. Save all components
    new_vlm.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    processor.save_pretrained(export_dir)

    # 10. Sanity check
    embedding = new_vlm.get_input_embeddings()
    lm_head = new_vlm.lm_head
    if embedding.weight.shape[0] != len(tokenizer) or lm_head.weight.shape[0] != len(tokenizer):
        raise RuntimeError(
            f"[AlphaGRPO] Mismatch after resizing: embedding {embedding.weight.shape[0]}, "
            f"lm_head {lm_head.weight.shape[0]}, tokenizer {len(tokenizer)}"
        )

    del model
    logger.info(f"[AlphaGRPO] Export completed at {export_dir}")


# def _export_vlm_if_needed(source_model_path: str, export_dir: Path) -> None:
    # """
    # Export the VLM submodule from an Alpamayo1_5 model to a given directory,
    # making sure the tokenizer and VLM vocab_size are consistent and
    # divisible by 16, and optionally reducing layers.
    # """
    # export_dir = Path(export_dir)
    # export_dir.mkdir(parents=True, exist_ok=True)

    # logger.info(f"[AlphaGRPO] Exporting VLM submodule from {source_model_path} to {export_dir}")

    # # 1. Load the full model
    # model = Alpamayo1_5.from_pretrained(source_model_path, dtype=torch.bfloat16)
    # vlm = model.vlm
    # tokenizer = model.tokenizer

    # # 2. Adjust vocab_size to be divisible by 16
    # old_vocab_size = len(tokenizer)
    # new_vocab_size = math.ceil(old_vocab_size / 16) * 16

    # if new_vocab_size != old_vocab_size:
    #     logger.info(f"[AlphaGRPO] Resize vocab: {old_vocab_size} -> {new_vocab_size}")

    #     additional_tokens = new_vocab_size - old_vocab_size
    #     logger.info(f"[AlphaGRPO] Additional tokens: {additional_tokens}")

    #     if additional_tokens > 0:
    #         additional_tokens_list = [f"<extra_id_{i}>" for i in range(additional_tokens)]
    #         tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens_list})

    #     # Resize token embeddings and LM head
    #     vlm.resize_token_embeddings(new_vocab_size)
    #     vlm.config.vocab_size = new_vocab_size

    #     logger.info(f"[AlphaGRPO] New tokenizer size: {len(tokenizer)}")
    # else:
    #     logger.info(f"[AlphaGRPO] Vocab size already divisible by 16: {old_vocab_size}")

    # # 3. Tailor the VLM (reduce layers)
    # keep_ratio = 0.5  # 例如保留 50% 的层
    # layers = vlm.model.language_model.layers
    # original_layers = len(layers)
    # keep_layers = max(1, int(original_layers * keep_ratio))

    # logger.info(f"[AlphaGRPO] Reducing VLM layers: {original_layers} -> {keep_layers}")

    # vlm.config.num_hidden_layers = keep_layers
    # new_vlm = type(vlm)(vlm.config)  # 创建新的 VLM 实例
    # # deepcopy 保证权重独立
    # new_vlm.model.language_model.layers = torch.nn.ModuleList(copy.deepcopy(layers[:keep_layers]))
    # # 保证 embedding 与 tokenizer 对齐
    # new_vlm.resize_token_embeddings(len(tokenizer))
    # vlm = new_vlm

    # logger.info(f"[AlphaGRPO] Final layer count: {len(vlm.model.language_model.layers)}")

    # # 4. Prepare processor
    # processor = helper.get_processor(tokenizer)

    # # 5. Save all components
    # vlm.save_pretrained(export_dir)
    # tokenizer.save_pretrained(export_dir)
    # processor.save_pretrained(export_dir)

    # # 6. Sanity check
    # embedding = vlm.get_input_embeddings()
    # lm_head = vlm.lm_head
    # logger.info(f"[AlphaGRPO] embedding shape: {embedding.weight.shape}, lm_head shape: {lm_head.weight.shape}")

    # if embedding.weight.shape[0] != len(tokenizer) or lm_head.weight.shape[0] != len(tokenizer):
    #     raise RuntimeError(
    #         f"[AlphaGRPO] Mismatch after resizing: embedding {embedding.weight.shape[0]}, "
    #         f"lm_head {lm_head.weight.shape[0]}, tokenizer {len(tokenizer)}"
    #     )

    # del model
    # logger.info(f"[AlphaGRPO] Export completed at {export_dir}")

# def _export_vlm_if_needed(source_model_path: str, export_dir: Path) -> None:
#     """
#     Export the VLM submodule from an Alpamayo1_5 model to a given directory,
#     making sure the tokenizer and VLM vocab_size are consistent and
#     divisible by 8.
#     """
#     export_dir = Path(export_dir)
#     export_dir.mkdir(parents=True, exist_ok=True)

#     logger.info(f"[AlphaGRPO] Exporting VLM submodule from {source_model_path} to {export_dir}")

#     # 1. Load the full model
#     model = Alpamayo1_5.from_pretrained(source_model_path, dtype=torch.bfloat16)
#     vlm = model.vlm
#     tokenizer = model.tokenizer

#     # 2. Adjust vocab_size to be divisible by 8
#     old_vocab_size = len(tokenizer)
#     new_vocab_size = math.ceil(old_vocab_size / 16) * 16

#     if new_vocab_size != old_vocab_size:
#         logger.info(f"[AlphaGRPO] Resize vocab: {old_vocab_size} -> {new_vocab_size}")

#         additional_tokens = new_vocab_size - len(tokenizer)
#         logger.info(f"[AlphaGRPO] additional tokens: {additional_tokens}")

#         if additional_tokens > 0:
#             additional_tokens_list = [f"<extra_id_{i}>" for i in range(additional_tokens)]
#             tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens_list})
        
#         # Resize token embeddings and LM head
#         vlm.resize_token_embeddings(new_vocab_size)
#         vlm.config.vocab_size = new_vocab_size

#         # Sync tokenizer
#         # if hasattr(tokenizer, "vocab_size"):
#         #     tokenizer.vocab_size = new_vocab_size

#         logger.info(f"[AlphaGRPO] New tokenizer size: {len(tokenizer)}")
#     else:
#         logger.info(f"[AlphaGRPO] Vocab size already divisible by 8: {old_vocab_size}")

#     # 2.1 tailor the vlm
#     # print(vlm)
#     # keep_ratio = 0.5  # Keep 50% of the layers
#     # original_layers = len(vlm.language_model.layers)
#     # keep_layers = max(1, int(original_layers * keep_ratio))
#     # vlm.language_model.layers = vlm.language_model.layers[:keep_layers]
#     # vlm.config.num_hidden_layers = keep_layers
#     # logger.info(f"[AlphaGRPO] Reduced VLM layers: {original_layers} -> {keep_layers}")

#     # 2.1 tailor the vlm (正确版本)

#     keep_ratio = 0.5
#     layers = vlm.model.language_model.layers  # ✅ 正确路径

#     original_layers = len(layers)
#     keep_layers = max(1, int(original_layers * keep_ratio))

#     logger.info(f"[AlphaGRPO] Reducing VLM layers: {original_layers} -> {keep_layers}")

#     vlm.config.num_hidden_layers = keep_layers
#     new_vlm = type(vlm)(vlm.config)  # 创建一个新的 VLM 实例
#     new_vlm.model.language_model.layers = torch.nn.ModuleList(layers[:keep_layers])  # 直接裁剪 ModuleList
#     new_vlm.resize_token_embeddings(len(tokenizer))  # 确保新模型的 vocab_size 和 tokenizer 一致
#     vlm = new_vlm  # 替换原来的 VLM


#     # # ❗ Step 1: 获取 state_dict
#     # state_dict = vlm.state_dict()
#     # new_state_dict = {}

#     # # ❗ Step 2: 裁剪权重（核心）
#     # for k, v in state_dict.items():
#     #     if "language_model.layers." in k:
#     #         # 提取 layer id
#     #         try:
#     #             layer_id = int(k.split("language_model.layers.")[1].split(".")[0])
#     #         except:
#     #             new_state_dict[k] = v
#     #             continue

#     #         if layer_id < keep_layers:
#     #             new_state_dict[k] = v
#     #     else:
#     #         new_state_dict[k] = v

#     # # ❗ Step 3: 更新 config
#     # vlm.config.num_hidden_layers = keep_layers

#     # # ❗ Step 4: 重新加载裁剪后的权重（让模型结构和权重一致）
#     # missing, unexpected = vlm.load_state_dict(new_state_dict, strict=False)

#     # logger.info(f"[AlphaGRPO] Missing keys after pruning: {len(missing)}")
#     # logger.info(f"[AlphaGRPO] Unexpected keys after pruning: {len(unexpected)}")

#     # # ❗ Step 5: 真正裁掉 ModuleList（保证 state_dict 和结构一致）
#     # vlm.model.language_model.layers = torch.nn.ModuleList(
#     #     list(vlm.model.language_model.layers[:keep_layers])
#     # )

#     logger.info(f"[AlphaGRPO] Final layer count: {len(vlm.model.language_model.layers)}")

#     # 3. Prepare processor
#     processor = helper.get_processor(tokenizer)

#     # 4. Save all components
#     vlm.save_pretrained(export_dir)
#     tokenizer.save_pretrained(export_dir)
#     processor.save_pretrained(export_dir)

#     # 5. Optional sanity check
#     embedding = vlm.get_input_embeddings()
#     lm_head = vlm.lm_head
#     logger.info(f"[AlphaGRPO] embedding shape: {embedding.weight.shape}, lm_head shape: {lm_head.weight.shape}")

#     if embedding.weight.shape[0] != len(tokenizer) or lm_head.weight.shape[0] != len(tokenizer):
#         raise RuntimeError(
#             f"[AlphaGRPO] Mismatch after resizing: embedding {embedding.weight.shape[0]}, "
#             f"lm_head {lm_head.weight.shape[0]}, tokenizer {len(tokenizer)}"
#         )

#     del model
#     logger.info(f"[AlphaGRPO] Export completed at {export_dir}")


# if __name__ == "__main__":
#     model_name_or_path = "nvidia/Alpamayo-1.5-10B"
#     export_dir = "/workspace/models/exported_vlm_test"
#     _export_vlm_if_needed(model_name_or_path, export_dir)

def load_model_and_tokenizer(export_dir: Path):
    from transformers import AutoTokenizer, AutoConfig, AutoProcessor, AutoModel

    # 1️⃣ 重新加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(export_dir, fix_mistral_regex=True)
    print(f"[Reload] tokenizer size: {len(tokenizer)}")

    # 2️⃣ 重新加载模型（HF）
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        export_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print(model)

    model.eval()

    # 3️⃣ 检查层数是否正确
    try:
        num_layers = len(model.model.language_model.layers)
    except:
        num_layers = "unknown"

    print(f"[Reload] num_hidden_layers (config): {model.config.text_config.num_hidden_layers}")
    print(f"[Reload] actual layers: {num_layers}")

    # 4️⃣ 检查 embedding / lm_head 是否对齐
    embedding = model.get_input_embeddings()
    lm_head = model.lm_head

    print(f"[Reload] embedding shape: {embedding.weight.shape}")
    print(f"[Reload] lm_head shape: {lm_head.weight.shape}")

    assert embedding.weight.shape[0] == len(tokenizer)
    assert lm_head.weight.shape[0] == len(tokenizer)

    return model, tokenizer

def forward_test(model, tokenizer):
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    model = model.cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    print("[Reload] forward pass success ✅")
    print(f"[Reload] logits shape: {outputs.logits.shape}")

if __name__ == "__main__":
    model_name_or_path = "nvidia/Alpamayo-1.5-10B"
    export_dir = "/workspace/models/exported_vlm_test"

    _export_vlm_if_needed(model_name_or_path, export_dir)

    print("\n========== Reload Test ==========\n")
    model, tokenizer = load_model_and_tokenizer(export_dir)
    forward_test(model, tokenizer)

    # import torch
    # from transformers import AutoTokenizer, AutoConfig, AutoProcessor, AutoModel

    # # 1️⃣ 重新加载 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(export_dir)
    # print(f"[Reload] tokenizer size: {len(tokenizer)}")

    # # 2️⃣ 重新加载模型（HF）
    # model = AutoModel.from_pretrained(
    #     export_dir,
    #     dtype=torch.bfloat16,   # ⚠️ 新版本写法
    #     trust_remote_code=True
    # )

    # model.eval()

    # # 3️⃣ 检查层数是否正确
    # try:
    #     num_layers = len(model.model.language_model.layers)
    # except:
    #     num_layers = "unknown"

    # print(f"[Reload] num_hidden_layers (config): {model.config.num_hidden_layers}")
    # print(f"[Reload] actual layers: {num_layers}")

    # # 4️⃣ 检查 embedding / lm_head 是否对齐
    # embedding = model.get_input_embeddings()
    # lm_head = model.lm_head

    # print(f"[Reload] embedding shape: {embedding.weight.shape}")
    # print(f"[Reload] lm_head shape: {lm_head.weight.shape}")

    # assert embedding.weight.shape[0] == len(tokenizer)
    # assert lm_head.weight.shape[0] == len(tokenizer)

    # # 5️⃣ forward 测试（关键）
    # input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    # model = model.cuda()

    # with torch.no_grad():
    #     outputs = model(input_ids=input_ids)

    # print("[Reload] forward pass success ✅")
    # print(f"[Reload] logits shape: {outputs.logits.shape}")

    print("\n========== Reload Test Passed ==========\n")
