from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from transformers import AutoTokenizer, AutoProcessor
import torch
from typing import List, Any, Dict, Optional, Tuple, Union


def build_message():
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"Loading dataset for clip_id: {clip_id}...")
    payload = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print("Dataset loaded.")

    image_frames = payload.get("image_frames")
    camera_indices = payload.get("camera_indices")
    ego_history_xyz = payload.get("ego_history_xyz")
    ego_history_rot = payload.get("ego_history_rot")
    
    # tokenizer
    # model_name_or_path = os.environ.get(ALPAMAYO_SOURCE_MODEL_ENV, self.config.policy.model_name_or_path)
    model_name_or_path = "/workspace/models/exported_vlm"
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
    
    print(messages)
    return messages


def process_vision_info(sample: List[Dict[str, Any]]) -> Tuple[Any, Any]:
    # This only handles the case of single image, not video.
    image_inputs = []
    video_inputs = []
    for x in sample:
        if x["role"] == "user":
            for item in x["content"]:
                if item["type"] == "image":
                    image_inputs.append(item["image"])
    return image_inputs, video_inputs

def covert_to_input_id(messages, tokenize=False):
    # from cosmos_rl.dispatcher.data.packer.qwen3_vl_data_packer import process_vision_info
    # processor = helper.get_processor(tokenizer)
    processor = AutoProcessor.from_pretrained("/workspace/models/exported_vlm", trust_remote_code=True)
    # inputs = processor(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True

    # )
    if tokenize:
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        print(f"Tokenized inputs: {len(inputs[0])}, {inputs[0][:50]}")
    else:
        input_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        print(f"image_inputs: {image_inputs}, video_inputs: {video_inputs}")
        video_kwargs = {}
        kwarg = {
                "return_tensors": "pt",
                "images": image_inputs,
            }
        inputs = processor(text=[input_text], **kwarg, **video_kwargs)
        print(inputs['input_ids'].shape, inputs['input_ids'][0][:50])   

if __name__ == "__main__":
    messages = build_message()
    model_name_or_path = "/workspace/models/exported_vlm"
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fix_mistral_regex=True)
    covert_to_input_id(messages, tokenize=False)