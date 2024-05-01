import json
import random
import os

with open("/home/eneskaranfil/projects/RS-Data/captioning/RSICD/dataset_rsicd.json", "r") as f:
    captioning_data = json.load(f)
    
image_path_prefix = "/home/eneskaranfil/projects/RS-Data/captioning/RSICD/RSICD_images/"

questions = [
    "Briefly describe this image.",
    "Summarize this image in a few words."
]

def convert_captioning_dataset_to_qa_format(dataset, image_path_prefix):
    new_format_data = []
    
    for image in dataset["images"]:
        for sentence in image["sentences"]:
            question = random.choice(questions)  # Randomly choose a question
            new_entry = {
                "id": str(sentence["sentid"]),
                "image": f"{image_path_prefix}{image['filename']}",  # Convert .tif to .png and prepend path
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": sentence["raw"]
                    }
                ]
            }
            new_format_data.append(new_entry)
    
    return new_format_data

converted_data = convert_captioning_dataset_to_qa_format(captioning_data, image_path_prefix)


# Path to the JSON file
json_file_path = '/home/eneskaranfil/projects/RS-Data/captioning/Cap_ERA/CapERA_DATASET_train.json'

# Load JSON data
with open(json_file_path, 'r') as file:
    caption_data = json.load(file)['ERA_caption']

# Base directory for images
base_dir = '/home/eneskaranfil/projects/RS-Data/captioning/Cap_ERA/SingleFrames/Tra/'

# Placeholder for the converted data
converted_data = []

def create_question(caption):
    # Example questions, you might want to adjust them to fit your use case
    questions = [
        "Briefly describe this image.",
        "Summarize this image in a few words."
    ]
    question_template = random.choice(questions)
    return f"<image>\n{question_template}"

for item in caption_data:
    video_id = item['video_id']
    captions = item['annotation']['English_caption']
    
    sport_type = video_id.split('_')[0]  # Assuming naming convention like Baseball_001.mp4
    sport_dir = os.path.join(base_dir, sport_type)
    image_filename = f"{video_id.split('.')[0]} .png"
    image_path = os.path.join(sport_dir, image_filename)
    if os.path.exists(image_path):

        for caption in captions:
            new_entry = {
                "id": video_id.split('.')[0],  # Use the video_id without extension as a unique identifier
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": create_question(caption)
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            }
            converted_data.append(new_entry)

with open('instruction_tuning_dataset_capera.json', "w") as json_file:
    json.dump(converted_data, json_file, indent=4)
    
    
import os
import json

def combine_json_files(directory):
    combined_data = []  # This will store all the combined JSON data
    unique_id_counter = 0  # Counter to ensure unique IDs
    
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Check for JSON files
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)  # Load JSON data from file
                    for item in data:  # Assuming data is a list of items
                        item['id'] = str(unique_id_counter)  # Assign a new unique ID
                        combined_data.append(item)  # Append the item to the combined list
                        unique_id_counter += 1  # Increment the counter
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}")
            except Exception as e:
                print(f"An error occurred: {e}")

    return combined_data


full_data = combine_json_files('/home/eneskaranfil/projects/RS-Data/lang_inst_tuning/vqa/')




#Platforms: USGS and Open Street Map, Remote Sensing Visual Question Answering
import json

file_path = '/home/eneskaranfil/projects/RS-Data/vqa/USGS_Open_Street_Map_QA/USGSquestions.json'

with open(file_path, 'r') as file:
    questions = json.load(file)
    
file_path = '/home/eneskaranfil/projects/RS-Data/vqa/USGS_Open_Street_Map_QA/USGSanswers.json'

with open(file_path, 'r') as file:
    answer = json.load(file)
    
def convert_to_instruction_tuning_format(questions, answers):
    instruction_tuning_dataset = {"annotations": []}
    answers_by_id = {answer["id"]: answer for answer in answers["answers"]}
    
    for question in questions["questions"]:
        if question["active"]:  
            filename = f'{question["img_id"]}.png'
            qa_pairs = []
            
            for answer_id in question["answers_ids"]:
                if answer_id in answers_by_id:
                    answer = answers_by_id[answer_id]
                    if answer["active"]:  
                        qa_pairs.append({
                            "question": question["question"],
                            "answer": answer["answer"],
                            "type": question["type"]
                        })
            
            instruction_tuning_dataset["annotations"].append({
                "filename": filename,
                "qa_pairs": qa_pairs
            })
                        
    return instruction_tuning_dataset

from collections import defaultdict

def convert_dataset_to_required_format(dataset):
    converted_data = {}
    for item in dataset:
        image_id = item['filename'].rsplit('.', 1)[0]  # Extract filename without extension
        image_path = f"{image_id}.png"  # Convert to JPG and construct storage path
        
        if image_id not in converted_data:
            converted_data[image_id] = {
                "id": image_id,
                "image": image_path,
                "conversations": []
            }
        
        for qa_pair in item['qa_pairs']:
            converted_data[image_id]['conversations'].append({"from": "human", "value": "<image>\n" + qa_pair['question']})
            converted_data[image_id]['conversations'].append({"from": "gpt", "value": qa_pair['answer']})

    return list(converted_data.values())

#instruction_tuning_dataset = convert_to_instruction_tuning_format(questions, answer)
#converted_dataset = convert_dataset_to_required_format(instruction_tuning_dataset['annotations'])

# Function to convert dataset into the desired format
def convert_dataset_format(original_datasets):
    new_format = []
    # Iterate through each dataset in the list of datasets
    for dataset in original_datasets:
        for i in range(0, len(dataset["conversations"]), 2):
            entry = {
                "id": dataset["id"],
                "image": dataset["image"],
                "conversations": [dataset["conversations"][i], dataset["conversations"][i+1]]
            }
            new_format.append(entry)
    return new_format

# Convert the dataset
#converted_dataset = convert_dataset_format(converted_dataset)

#with open('rs-instruction-tuning.json', 'w') as f:
#    json.dump(converted_dataset, f, indent=4)
    
    
#floodnet dataset
floodnet_data = json.load(open("/home/eneskaranfil/projects/RS-Data/vqa/FloodNet/Questions/Training Question.json"))
img_root = '/home/eneskaranfil/projects/RS-Data/vqa/FloodNet/Images/Training Images/' 

# Convert FloodNet dataset format to the desired new format
def convert_format(floodnet_data):
    new_format_data = []
    for key, value in floodnet_data.items():
        new_entry = {
            "id": key,
            "image": img_root+value["Image_ID"], # Assuming conversion to .png is required
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + value["Question"]
                },
                {
                    "from": "gpt",
                    "value": value["Ground_Truth"]
                }
            ]
        }
        new_format_data.append(new_entry)
    return new_format_data

# Run the conversion
#new_dataset = convert_format(floodnet_data)

output_file_path = "instruction_tuning_dataset_floodnet.json"
#with open(output_file_path, "w") as json_file:
#    json.dump(new_dataset, json_file, indent=4)


rsvqaxbe_question = json.load(open('/home/eneskaranfil/projects/RS-Data/vqa/RSVQAxBEN/LRBENquestions.json'))
rsvqaxben_answer = json.load(open('/home/eneskaranfil/projects/RS-Data/vqa/RSVQAxBEN/LRBENanswers.json'))


root_path = '/home/eneskaranfil/projects/RS-Data/vqa/RSVQAxBEN/Images/'

def convert_rsvqaxbe_format(questions, answers):
   
    active_answers = {ans['question_id']: ans['answer'] for ans in answers if ans['active']}
    
    new_format_data = [
        {
            "id": str(question['id']),
            "image": f"{root_path}{question['img_id']}.tiff",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question['question']}"
                },
                {
                    "from": "gpt",
                    "value": active_answers.get(question['id'], "Unknown")  
                }
            ]
        }
        for question in questions if question['active'] and question['id'] in active_answers
    ]
    
    return new_format_data

rsqvqa_converted = convert_rsvqaxbe_format(rsvqaxbe_question['questions'], rsvqaxben_answer['answers'])



with open('instruction_tuning_dataset_rsvqaxben.json', "w") as json_file:
    json.dump(rsqvqa_converted, json_file, indent=4)