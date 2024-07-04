import os
import pandas as pd


def get_image_paths(data_path):
    """
    Get the paths of all images in the data_path directory
    """
    image_paths = []
    for root, _, files in os.walk(data_path):
        print(len(files))
        for file in files:
            if (file.endswith(".PNG") or file.endswith(".JPEG") or file.endswith(".JPG") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg")):
                tmp_root = root
                print(f"New root {tmp_root}")
                image_paths.append(os.path.join(tmp_root, file))
    df = pd.DataFrame(image_paths, columns=['path'])
    return df

def generate_csv(data_path, csv_path, group_name,generate_header=False):
    """
    Generate a csv file with the paths of all images in the data_path directory
    """
    df = get_image_paths(data_path)
    df['group_name'] = group_name
    assert df['path'].all() != None
    split = 0.7
    df = df.sample(frac=1).reset_index(drop=True)
    print(df['path'].iloc[0])

    split_index = int(len(df) * split)

    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    os.makedirs(f'train/', exist_ok=True)
    os.makedirs(f'test/', exist_ok=True)
    mode = 'w' if generate_header else 'a'
    df_train.to_csv(f'train/{csv_path}', index=False, sep=";", mode=mode, header=generate_header)
    df_test.to_csv(f'test/{csv_path}', index=False, sep=";", mode=mode, header=generate_header)

    # print(f'CSV content saved to {csv_path}, added {len(df)} rows.')
# European_fire_salamander was removed
for models in ['adm', 'biggan', 'stablediffusion']:
    for group in ['basketball','canoe', 'electric_guitar','flamingo',
                    'jellyfish','laptop','Maltese_dog','obelisk','peacock',
                     'pizza', 'reflex_camera','samoyed','ski',
                        'sportscar','strawberry', 'tractor','vacuum', 'violin']:
        for type in ['ai', 'nature']:
            generate_csv(f'../subset_evaluation/{models}/{type}_{group}', f'{models}_{group}.csv', f'{type}_{group}', type=='ai')

# generate_csv('../subset_evaluation/adm/ai_peacock', './data/adm_peacock.csv', 'ai_peacock')
# generate_csv('../subset_evaluation/adm/ai_samoyed', './data/adm_samoyed.csv', 'ai_samoyed')
# generate_csv('../subset_evaluation/adm/ai_pizza', './data/adm_pizza.csv', 'ai_pizza')
# generate_csv('../subset_evaluation/adm/ai_sportscar', './data/adm_sportscar.csv', 'ai_sportscar')
# generate_csv('../subset_evaluation/adm/ai_violin', './data/adm_violin.csv', 'ai_violin')
# generate_csv('../subset_evaluation/adm/nature_peacock', './data/adm_peacock.csv', 'nature_peacock')
# generate_csv('../subset_evaluation/adm/nature_pizza', './data/adm_pizza.csv', 'nature_pizza')
# generate_csv('../subset_evaluation/adm/nature_samoyed', './data/adm_samoyed.csv', 'nature_samoyed')
# generate_csv('../subset_evaluation/adm/nature_sportscar', './data/adm_sportscar.csv', 'nature_sportscar')
# generate_csv('../subset_evaluation/adm/nature_violin', './data/adm_violin.csv', 'nature_violin')

# generate_csv('../subset_evaluation/biggan/ai_peacock', './data/biggan_peacock.csv', 'ai_peacock')
# generate_csv('../subset_evaluation/biggan/ai_samoyed', './data/biggan_samoyed.csv', 'ai_samoyed')
# generate_csv('../subset_evaluation/biggan/ai_pizza', './data/biggan_pizza.csv', 'ai_pizza')
# generate_csv('../subset_evaluation/biggan/ai_sportscar', './data/biggan_sportscar.csv', 'ai_sportscar')
# generate_csv('../subset_evaluation/biggan/ai_violin', './data/biggan_violin.csv', 'ai_violin')
# generate_csv('../subset_evaluation/biggan/nature_peacock', './data/biggan_peacock.csv', 'nature_peacock')
# generate_csv('../subset_evaluation/biggan/nature_pizza', './data/biggan_pizza.csv', 'nature_pizza')
# generate_csv('../subset_evaluation/biggan/nature_samoyed', './data/biggan_samoyed.csv', 'nature_samoyed')
# generate_csv('../subset_evaluation/biggan/nature_sportscar', './data/biggan_sportscar.csv', 'nature_sportscar')
# generate_csv('../subset_evaluation/biggan/nature_violin', './data/biggan_violin.csv', 'nature_violin')


# generate_csv('../subset_evaluation/stablediffusion/ai_peacock', './data/stablediffusion_peacock.csv', 'ai_peacock')
# generate_csv('../subset_evaluation/stablediffusion/ai_samoyed', './data/stablediffusion_samoyed.csv', 'ai_samoyed')
# generate_csv('../subset_evaluation/stablediffusion/ai_pizza', './data/stablediffusion_pizza.csv', 'ai_pizza')
# generate_csv('../subset_evaluation/stablediffusion/ai_sportscar', './data/stablediffusion_sportscar.csv', 'ai_sportscar')
# generate_csv('../subset_evaluation/stablediffusion/ai_violin', './data/stablediffusion_violin.csv', 'ai_violin')
# generate_csv('../subset_evaluation/stablediffusion/nature_peacock', './data/stablediffusion_peacock.csv', 'nature_peacock')
# generate_csv('../subset_evaluation/stablediffusion/nature_pizza', './data/stablediffusion_pizza.csv', 'nature_pizza')
# generate_csv('../subset_evaluation/stablediffusion/nature_samoyed', './data/stablediffusion_samoyed.csv', 'nature_samoyed')
# generate_csv('../subset_evaluation/stablediffusion/nature_sportscar', './data/stablediffusion_sportscar.csv', 'nature_sportscar')
# generate_csv('../subset_evaluation/stablediffusion/nature_violin', './data/stablediffusion_violin.csv', 'nature_violin')