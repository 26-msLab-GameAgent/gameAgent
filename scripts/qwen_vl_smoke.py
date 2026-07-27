"""Standalone Qwen-VL image recognition smoke test."""

from __future__ import annotations

import argparse

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


QWEN_VL_MODEL_IDS = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=sorted(QWEN_VL_MODEL_IDS), default="7B")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--image", default="/data/project/sink0324/gameagent/test.png")
    parser.add_argument("--prompt", default="Describe this mobile game screen briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id or QWEN_VL_MODEL_IDS[args.model_size],
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id or QWEN_VL_MODEL_IDS[args.model_size])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    output = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0]
    print(output)


if __name__ == "__main__":
    main()
