from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)


model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Provide a 2 sentence brief description of the landscape features visible in this remote sensing image, including any natural and man-made structures?"
image_file = "/home/eneskaranfil/projects/RS-Data/grounding/JPEGImages/00001.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)



import os
import json

def generate_descriptions_for_directory(root_dir, output_json):
    data_list = []
    unique_id = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                args = type('Args', (), {
                    "model_path": "liuhaotian/llava-v1.5-7b",
                    "model_base": None,
                    "model_name": get_model_name_from_path("liuhaotian/llava-v1.5-7b"),
                    "query": "Provide a 1 sentence brief description of the landscape features visible in this remote sensing image, including any natural and man-made structures?",
                    "conv_mode": None,
                    "image_file": image_path,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512
                })()
                description = eval_model(args)
                data_entry = {
                    "id": str(unique_id),
                    "image": file,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nBriefly describe this image."
                        },
                        {
                            "from": "gpt",
                            "value": description
                        }
                    ]
                }
                data_list.append(data_entry)
                unique_id += 1

    with open(output_json, 'w') as f:
        json.dump(data_list, f, indent=4)

generate_descriptions_for_directory('/home/eneskaranfil/projects/RS-Data/classification/PatternNet/images', 'synthetic_pattenet_dataset.json')
