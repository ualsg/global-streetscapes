from dataloaders import GlobalStreetScapes_simple
from model import GlobalStreetScapesClassificationTrainer
from torch.utils.data import DataLoader
import pandas as pd
import torch
from tqdm import tqdm
import os
from pathlib import Path
from huggingface_hub import snapshot_download

attributes = ['glare', 'lighting_condition', 'pano_status', 'platform', 'quality', 'reflection', 'view_direction', 'weather']

ckpt_dir = './models'
ckpt_dict = {
    'lighting_condition': os.path.join(ckpt_dir, 'lighting_condition_uniform_e14259c3-aea1-4469-a70d-451955255a80-epoch=1_validation_loss=0.08.ckpt'),
    'pano_status': os.path.join(ckpt_dir, 'pano_status_inverse_45c8eac3-7d0f-4b20-8f3e-9b7969279544-epoch=0_validation_loss=0.00.ckpt'),
    'glare': os.path.join(ckpt_dir, 'glare_inverse_f6b03038-0831-43ec-88d4-e6c7eb5f8539-epoch=0_validation_loss=0.30.ckpt'),
    'quality': os.path.join(ckpt_dir, 'quality_inverse_ce2c16b6-2950-4fb2-b064-1078ed31aa05-epoch=0_validation_loss=1.31.ckpt'),
    'reflection': os.path.join(ckpt_dir, 'reflection_inverse_7ef9a22e-9025-4362-87a0-1cb1fe96debc-epoch=0_validation_loss=0.33.ckpt'),
    'view_direction': os.path.join(ckpt_dir, 'view_direction_inverse_a0bd363c-60e8-455c-b077-e5852aca9371-epoch=0_validation_loss=0.35.ckpt'),
    'weather': os.path.join(ckpt_dir, 'weather_inverse_78ff5ce3-dba4-4265-8acb-c2d577ec5403-epoch=0_validation_loss=6.11.ckpt'),
    'platform': os.path.join(ckpt_dir, 'platform_inverse_e87c7bab-78f3-4192-93c4-e71d1f90598a-epoch=0_validation_loss=3.39.ckpt')
}

def check_id(out_csvPath):
    ids = set()
    if os.path.exists(out_csvPath):
        df = pd.read_csv(out_csvPath, header=None, names=['uuid', 'value'])
        ls_id = df['uuid'].tolist()
        ids.update(ls_id)
    return ids

def main():

    out_folderPath = "./sample_output" # specify the path to the output folder; the output for each attribute will be stored as a csv in the output folder
    in_csvPath = "../../download_imgs/sample_output/all/img_paths.csv" # specify the path to your input CSV; the input csv must contain a column that has the paths to all images
    path_field = 'path' # specify the column in the input csv that contains the paths to all images
    batch_size = 32

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(out_folderPath).mkdir(parents=True, exist_ok=True)

    # download model
    print('Downloading models...')
    snapshot_download(repo_id="NUS-UAL/global-streetscapes", repo_type='dataset', allow_patterns=["*.ckpt"], local_dir='.')

    for attribute in attributes:

        out_csvPath = os.path.join(out_folderPath, attribute+'.csv')
        if os.path.exists(out_csvPath) == False:
            df = pd.DataFrame(columns=['uuid',attribute])
            df.to_csv(out_csvPath, index=False)
        
        checkpoint_path = ckpt_dict[attribute]

        print(f'Loading checkpoint: {checkpoint_path}')
                
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        label2index = checkpoint['class_mapping']
        index2label = {index: str(label) for label, index in label2index.items()}

        # Extract the number of classes
        num_classes = checkpoint['state_dict']['model.classifier.5.weight'].shape[0]

        print(f'Checkpoint keys: {checkpoint.keys()}')
                
        # # Now load the model
        model = GlobalStreetScapesClassificationTrainer.load_from_checkpoint(checkpoint_path, 
                                                                                    num_classes=num_classes,
                                                                                    weight=None,
                                                                                    class_mapping=checkpoint['class_mapping'],
                                                                                    strict=False)
        model.eval() # important before we start predicting anything
        model.cuda()

        print('Running inference for:', attribute)

        already_id = check_id(out_csvPath)

        df_imgs = pd.read_csv(in_csvPath)
        df_imgs = df_imgs[~df_imgs['uuid'].isin(already_id)]
                    
        ## Load the test dataset
        dataset = GlobalStreetScapes_simple(data=df_imgs, path_field=path_field)
                    
        dataloader = DataLoader(dataset,
                                        batch_size=batch_size, # Set this value as high as your GPU can manage
                                        shuffle=False, 
                                        num_workers=16)

        for (image,uuid) in tqdm(dataloader,total=len(dataloader)):
            output = model(image.cuda()) #Model assumes a batch dimension BxCxHxW
            pred_values = [index2label[x.item()] for x in output.argmax(-1).detach().cpu()]
            df = pd.DataFrame(list(uuid), columns=['uuid'])
            df['value'] = pred_values
            df.to_csv(out_csvPath, mode='a', header=False, index=False)
    
if __name__ == '__main__':
    main()