import os
import shutil

#cap-era dataset
def copy_image_files(source_root, target_directory):
    """
    Copies all image files from a nested directory structure starting at source_root
    to a single flat directory target_directory.
    
    Args:
    source_root (str): The root directory from which to start searching for image files.
    target_directory (str): The directory to which all image files should be copied.
    
    Returns:
    int: The count of files successfully copied.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif'}  
    copied_files_count = 0  # Counter for the number of files copied
    
    # Walk through all directories and files in the source_root
    for dirpath, dirnames, filenames in os.walk(source_root):
        for file in filenames:
            try:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    source_path = os.path.join(dirpath, file)
                    target_path = os.path.join(target_directory, file)
                    
                    # Copy file to target_directory, handle potential overwrites
                    if not os.path.exists(target_path):
                        shutil.copy(source_path, target_path)
                        copied_files_count += 1
                    else:
                        # Handling file name conflict by renaming
                        base, extension = os.path.splitext(file)
                        new_file = f"{base}{extension}"
                        target_path = os.path.join(target_directory, new_file)
                        shutil.copy(source_path, target_path)
                        copied_files_count += 1
            except Exception as e:
                print(e)

    return copied_files_count
# Example function call, commented out for development
print(copy_image_files('/home/eneskaranfil/projects/RS-Data/vqa/RSVQAxBEN/Images', '/home/eneskaranfil/projects/RS-Data/lang_inst_tuning/vqa/images'))



import json
import os
import re

def extract_filename(full_path):
    """
    Extracts the filename from a full path.
    
    Args:
    full_path (str): The full path to the file.
    
    Returns:
    str: The filename extracted from the full path.
    """
    return os.path.basename(full_path)

def clean_text(text):
    """
    Cleans the provided text by removing special characters and extra spaces, excluding meaningful punctuation.
    
    Args:
    text (str): The text to be cleaned.
    
    Returns:
    str: The cleaned text.
    """
    # Remove special characters except for punctuation marks often used in text analysis
    text = re.sub(r"[^a-zA-Z0-9.,'?!;\s]+", '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def process_json_data(json_file):
    """
    Processes a JSON file containing a list of entries, each expected to have an 'image' key and 'conversations' list.
    Each JSON object is processed to clean text and remove duplicates only for entries where "from": "gpt".
    
    Args:
    json_file (str): The path to the JSON file.
    
    Returns:
    list: A list of processed data with cleaned filenames and text for each entry, only modifying gpt entries.
    """
    with open(json_file, 'r') as file:
        data_list = json.load(file)  # Expecting a list of dictionaries
    
    processed_data_list = []
    
    for data in data_list:
        if 'image' in data:
            data['image'] = extract_filename(data['image'])
        
        if 'conversations' in data:
            seen_captions = set()
            unique_conversations = []
            for conv in data['conversations']:
                if conv['from'] == 'gpt':
                    value = str(conv['value']) if not isinstance(conv['value'], str) else conv['value']
                    cleaned_value = clean_text(value)
                    if cleaned_value not in seen_captions:
                        seen_captions.add(cleaned_value)
                        conv['value'] = cleaned_value
                unique_conversations.append(conv)
            
            data['conversations'] = unique_conversations
        
        processed_data_list.append(data)
    
    return processed_data_list
  

# Example of function usage, commented out for development
data = process_json_data('/home/eneskaranfil/projects/RS-Data/instruction_tuning_dataset_eurosat.json')

with open('instruction_tuning_dataset_eurosat.json', "w") as json_file:
    json.dump(data, json_file, indent=4)
    
#path checker
# List to hold entries to be removed if the image does not exist
to_delete = []
directory_path = '/home/eneskaranfil/projects/RS-Data/RS-LLaVA-Dataset'

# check data existence
# Iterate through the data and check if each image file exists
for item in data[:]:  # Make a shallow copy of the list for safe iteration
    image_path = os.path.join(directory_path, item['image'])
    if not os.path.exists(image_path):
        to_delete.append(item)
        data.remove(item)  # Remove the item from the original list