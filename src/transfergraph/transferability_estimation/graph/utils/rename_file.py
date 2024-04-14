import os
from glob import glob

# dataset = 'stanfordcars' #
# dataset = 'cifar100'
dataset = 'oxford_flowers102'
dataset = 'oxford_iiit_pet'
root = f'../rank/{dataset}'

exclude_folders = ['contain_dataset_feature', 'not_contain_dataset_feature', '.DS_Store']

# folders = os.listdir(root)
folders = ['lr_node2vec', 'lr_node2vec+']

for folder in folders[:]:
    if '.csv' in folder: continue
    print(f'\n folder: {folder}')
    if folder in exclude_folders[-1]: continue
    if folder in exclude_folders[:2]:
        _dir = os.path.join(root, folder, 'not_contain_model_feature', 'domain_similarity')
    elif folder not in exclude_folders:
        _dir = os.path.join(root, folder)

    print(f'_dir: {_dir}')
    files = glob(f'{_dir}/*')
    print('\n', files[0])

    for file in files:
        if ',' in file: continue
        parts = []
        file_name = file.split('/')[-1]
        components = file_name.split('_')
        string = ''
        for component in components:
            # print(f'component: {component}')
            if ('=' not in component) or (component in ['vit_base_patch16']) or ('google' in component):
                if '224' in component or 'transfer' in component or 'metric' in component:
                    string += component
                    parts.append(string)
                    string = ''
                else:
                    string += component + '_'
            else:
                string += component
                parts.append(string)
                string = ''
        new_file_name = ','.join(parts)
        print(f'\n new_file_name')  # -- {new_file_name}')
        print(os.path.join(_dir, new_file_name))
        os.rename(os.path.join(_dir, file_name), os.path.join(_dir, new_file_name))
