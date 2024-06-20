import os
import shutil

# object classes: 
# used models adm, biggan, stablediffusion
# used object classes(ai_id, nature_id): - sports car(817, n04285008), samoyed(258, n02111889), peacock(84, n01806143), pizza(963, n07873807) and violin(889, n04536866)


model_paths = ["adm", "biggan", "stablediffusion"]
ai_prefixes = ["84_", "817_", "258_", "963_", "889_"]
natural_prefixes = ["n04285008_", "n04536866", "n07873807", "n01806143", "n02111889"]

def extract_and_copy_ai_images(model_path, prefixes):
    base_path = os.path.join(os.getcwd(), model_path)
    image_files = []
    for prefix in prefixes:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith(prefix):
                    image_files.append(os.path.join(root, file))

    destination_folder = os.path.join("subset_evaluation", model_path, "ai")
    if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
    for file_path in image_files:
        destination_path = os.path.join(destination_folder, os.path.basename(file_path))
        if not os.path.exists(destination_path):
            shutil.copy(file_path, destination_path)

def extract_and_copy_natural_images(model_path, prefixes):
    base_path = os.path.join(os.getcwd(), model_path)
    image_files = []
    for prefix in prefixes:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith(prefix):
                    image_files.append(os.path.join(root, file))

    destination_folder = os.path.join("subset_evaluation", model_path, "nature")
    if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
    for file_path in image_files:
        destination_path = os.path.join(destination_folder, os.path.basename(file_path))
        if not os.path.exists(destination_path):
            shutil.copy(file_path, destination_path)

for model_path in model_paths: 
    extract_and_copy_ai_images(model_path, ai_prefixes)
    extract_and_copy_natural_images(model_path, natural_prefixes)