---
base_model:
- Qwen/Qwen2.5-VL-7B-Instruct
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
---

# GUI-Actor-7B with Qwen2.5-VL-7B as backbone VLM

This model was introduced in the paper [**GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents**](https://huggingface.co/papers/2506.03143).
It is developed based on [Qwen2.5-VL-7B-Instruct ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), augmented by an attention-based action head and finetuned to perform GUI grounding using the dataset [here](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data).

For more details on model design and evaluation, please check: [🏠 Project Page](https://microsoft.github.io/GUI-Actor/) | [💻 Github Repo](https://github.com/microsoft/GUI-Actor) | [📑 Paper](https://www.arxiv.org/pdf/2506.03143).

| Model Name                                  | Hugging Face Link                         |
|--------------------------------------------|--------------------------------------------|
| **GUI-Actor-7B-Qwen2-VL**                   | [🤗 Hugging Face](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2-VL)         |
| **GUI-Actor-2B-Qwen2-VL**                   | [🤗 Hugging Face](https://huggingface.co/microsoft/GUI-Actor-2B-Qwen2-VL)         |
| **GUI-Actor-7B-Qwen2.5-VL**                 | [🤗 Hugging Face](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)       |
| **GUI-Actor-3B-Qwen2.5-VL**                 | [🤗 Hugging Face](https://huggingface.co/microsoft/GUI-Actor-3B-Qwen2.5-VL)       |
| **GUI-Actor-7B-Qwen2.5-VL-LiteTrain**                 | [🤗 Hugging Face](https://huggingface.co/qianhuiwu/GUI-Actor-7B-Qwen2.5-VL-LiteTrain)       |
| **GUI-Actor-3B-Qwen2.5-VL-LiteTrain**                 | [🤗 Hugging Face](https://huggingface.co/qianhuiwu/GUI-Actor-3B-Qwen2.5-VL-LiteTrain)       |
| **GUI-Actor-Verifier-2B**                   | [🤗 Hugging Face](https://huggingface.co/microsoft/GUI-Actor-Verifier-2B)        |

## 📊 Performance Comparison on GUI Grounding Benchmarks
Table 1. Main results on ScreenSpot-Pro, ScreenSpot, and ScreenSpot-v2 with **Qwen2-VL** as the backbone. † indicates scores obtained from our own evaluation of the official models on Huggingface.
| Method           | Backbone VLM | ScreenSpot-Pro | ScreenSpot | ScreenSpot-v2 |
|------------------|--------------|----------------|------------|----------------|
| **_72B models:_**
| AGUVIS-72B       | Qwen2-VL     | -              | 89.2       | -              |
| UGround-V1-72B   | Qwen2-VL     | 34.5           | **89.4**   | -              |
| UI-TARS-72B      | Qwen2-VL     | **38.1**       | 88.4       | **90.3**       |
| **_7B models:_**
| OS-Atlas-7B      | Qwen2-VL     | 18.9           | 82.5       | 84.1           |
| AGUVIS-7B        | Qwen2-VL     | 22.9           | 84.4       | 86.0†          |
| UGround-V1-7B    | Qwen2-VL     | 31.1           | 86.3       | 87.6†          |
| UI-TARS-7B       | Qwen2-VL     | 35.7           | **89.5**   | **91.6**       |
| GUI-Actor-7B     | Qwen2-VL     | **40.7**       | 88.3       | 89.5           |
| GUI-Actor-7B + Verifier     | Qwen2-VL    | 44.2       | 89.7       | 90.9           |
| **_2B models:_**
| UGround-V1-2B    | Qwen2-VL     | 26.6           | 77.1       | -              |
| UI-TARS-2B       | Qwen2-VL     | 27.7           | 82.3       | 84.7           |
| GUI-Actor-2B     | Qwen2-VL     | **36.7**       | **86.5**   | **88.6**       |
| GUI-Actor-2B + Verifier     | Qwen2-VL    | 41.8       | 86.9       | 89.3           |

Table 2. Main results on the ScreenSpot-Pro and ScreenSpot-v2 with **Qwen2.5-VL** as the backbone.
| Method         | Backbone VLM | ScreenSpot-Pro | ScreenSpot-v2 |
|----------------|---------------|----------------|----------------|
| **_7B models:_**
| Qwen2.5-VL-7B  | Qwen2.5-VL    | 27.6           | 88.8           |
| Jedi-7B        | Qwen2.5-VL    | 39.5           | 91.7           |
| GUI-Actor-7B   | Qwen2.5-VL    | **44.6**       | **92.1**       |
| GUI-Actor-7B + Verifier   | Qwen2.5-VL    | 47.7       | 92.5       |
| **_3B models:_**
| Qwen2.5-VL-3B  | Qwen2.5-VL    | 25.9           | 80.9           |
| Jedi-3B        | Qwen2.5-VL    | 36.1           | 88.6           |
| GUI-Actor-3B   | Qwen2.5-VL    | **42.2**       | **91.0**       |
| GUI-Actor-3B + Verifier   | Qwen2.5-VL    | 45.9       | 92.4       |

## 🚀 Usage
```python
import torch

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from transformers import AutoProcessor
from gui_actor.constants import chat_template
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference


# load model
model_name_or_path = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
data_processor = AutoProcessor.from_pretrained(model_name_or_path)
tokenizer = data_processor.tokenizer
model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2"
).eval()

# prepare example
dataset = load_dataset("rootsautomation/ScreenSpot")["test"]
example = dataset[0]
print(f"Intruction: {example['instruction']}")
print(f"ground-truth action region (x1, y1, x2, y2): {[round(i, 2) for i in example['bbox']]}")

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).",
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": example["image"], # PIL.Image.Image or str to path
                # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
            },
            {
                "type": "text",
                "text": example["instruction"]
            },
        ],
    },
]

# inference
pred = inference(conversation, model, tokenizer, data_processor, use_placeholder=True, topk=3)
px, py = pred["topk_points"][0]
print(f"Predicted click point: [{round(px, 4)}, {round(py, 4)}]")

# >> Model Response
# Intruction: close this window
# ground-truth action region (x1, y1, x2, y2): [0.9479, 0.1444, 0.9938, 0.2074]
# Predicted click point: [0.9709, 0.1548]
```

## 📝 Citation
```
@article{wu2025gui,
  title={GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents},
  author={Wu, Qianhui and Cheng, Kanzhi and Yang, Rui and Zhang, Chaoyun and Yang, Jianwei and Jiang, Huiqiang and Mu, Jian and Peng, Baolin and Qiao, Bo and Tan, Reuben and others},
  journal={arXiv preprint arXiv:2506.03143},
  year={2025}
}
```