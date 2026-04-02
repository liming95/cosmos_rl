from pathlib import Path
import torch
import math
import logging

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _export_vlm_if_needed(source_model_path: str, export_dir: Path) -> None:
    """
    Export the VLM submodule from an Alpamayo1_5 model to a given directory,
    making sure the tokenizer and VLM vocab_size are consistent and
    divisible by 8.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[AlphaGRPO] Exporting VLM submodule from {source_model_path} to {export_dir}")

    # 1. Load the full model
    model = Alpamayo1_5.from_pretrained(source_model_path, dtype=torch.bfloat16)
    vlm = model.vlm
    tokenizer = model.tokenizer

    # 2. Adjust vocab_size to be divisible by 8
    old_vocab_size = len(tokenizer)
    new_vocab_size = math.ceil(old_vocab_size / 8) * 8

    if new_vocab_size != old_vocab_size:
        logger.info(f"[AlphaGRPO] Resize vocab: {old_vocab_size} -> {new_vocab_size}")

        # Resize token embeddings and LM head
        vlm.resize_token_embeddings(new_vocab_size)
        vlm.config.vocab_size = new_vocab_size

        # Sync tokenizer
        # if hasattr(tokenizer, "vocab_size"):
        #     tokenizer.vocab_size = new_vocab_size

        additional_tokens = new_vocab_size - len(tokenizer)
        logger.info(f"[AlphaGRPO] additional tokens: {additional_tokens}")

        if additional_tokens > 0:
            additional_tokens_list = [f"<extra_id_{i}>" for i in range(additional_tokens)]
            tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens_list})

        logger.info(f"[AlphaGRPO] New tokenizer size: {len(tokenizer)}")
    else:
        logger.info(f"[AlphaGRPO] Vocab size already divisible by 8: {old_vocab_size}")

    # 3. Prepare processor
    processor = helper.get_processor(tokenizer)

    # 4. Save all components
    vlm.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    processor.save_pretrained(export_dir)

    # 5. Optional sanity check
    embedding = vlm.get_input_embeddings()
    lm_head = vlm.lm_head
    logger.info(f"[AlphaGRPO] embedding shape: {embedding.weight.shape}, lm_head shape: {lm_head.weight.shape}")

    if embedding.weight.shape[0] != len(tokenizer) or lm_head.weight.shape[0] != len(tokenizer):
        raise RuntimeError(
            f"[AlphaGRPO] Mismatch after resizing: embedding {embedding.weight.shape[0]}, "
            f"lm_head {lm_head.weight.shape[0]}, tokenizer {len(tokenizer)}"
        )

    del model
    logger.info(f"[AlphaGRPO] Export completed at {export_dir}")


if __name__ == "__main__":
    model_name_or_path = "nvidia/Alpamayo-1.5-10B"
    export_dir = "/workspace/models/exported_vlm"
    _export_vlm_if_needed(model_name_or_path, export_dir)