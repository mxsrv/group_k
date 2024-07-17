import os
import pandas as pd
import json
import shutil

# object classes: 
# used models adm, biggan, stablediffusion
# used object classes(ai_id, nature_id): - sports car(817, n04285008), samoyed(258, n02111889), peacock(84, n01806143), pizza(963, n07873807) and violin(889, n04536866)

model_paths = ["adm", "biggan", "stablediffusion"]
ai_prefixes = {"084_":"peacock", "817_":"sportscar", "258_":"samoyed", "963_":"pizza", "889_":"violin"}
natural_prefixes = {"n01806143":"peacock", "n04285008_":"sportscar", "n02111889":"samoyed", "n07873807":"pizza", "n04536866":"violin"}

def extract_and_copy_images(model_path, prefixes, type):
    base_path = os.path.join(os.getcwd(), model_path)

    for prefix in prefixes.keys():
        image_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith(prefix):
                    image_files.append(os.path.join(root, file))

        destination_folder = os.path.join("subset_evaluation", model_path, f"{type}_{prefixes[prefix]}")
        if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
        for file_path in image_files:
            destination_path = os.path.join(destination_folder, os.path.basename(file_path))
            if not os.path.exists(destination_path):
                shutil.copy(file_path, destination_path)

df = pd.read_csv("Selected_classes.csv")

with open("imagenet_class_index.json", 'r') as file:
    imgNet_classes = json.load(file)
    for number in df["Number"]:
        number_str = str(number)
        name = imgNet_classes[number_str][1]
        nature_prefix = imgNet_classes[number_str][0]
        ai_prefix = number_str.zfill(3) +"_"
        ai_prefixes[ai_prefix] = name
        natural_prefixes[nature_prefix] = name


for model_path in model_paths: 
    extract_and_copy_images(model_path, ai_prefixes, type="ai")
    extract_and_copy_images(model_path, natural_prefixes, type="nature")