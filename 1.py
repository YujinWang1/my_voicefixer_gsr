import os

root_dir = '/home/yujin/miniconda3/envs/env_voicefixer/lib/python3.9/site-packages/voicefixer'

def replace_parametrizations_weight_norm(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'nn.utils.parametrizations.weight_norm' in content:
        new_content = content.replace(
            'nn.utils.parametrizations.weight_norm',
            'nn.utils.weight_norm'
        )
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Modified: {file_path}')

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.py'):
            full_path = os.path.join(dirpath, filename)
            replace_parametrizations_weight_norm(full_path)
