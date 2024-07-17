# coding=UTF-8  
from transformers import AutoModel
from huggingface_hub import snapshot_download
import os
import pandas as pd
import torch 
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

ImageFile.LOAD_TRUNCATED_IMAGES = True

perception = ['Safe', 'Lively', 'Wealthy', 'Beautiful', 'Boring', 'Depressing']
model_dict = {
            'Safe':'/safety.pth', \
            'Lively': '/lively.pth', \
            'Wealthy': '/wealthy.pth',\
            'Beautiful':'/beautiful.pth',\
            'Boring': '/boring.pth',\
            'Depressing': '/depressing.pth',\
            }


train_transform = T.Compose([
    T.Resize((384,384)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


def predict(model, img_path, device):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = train_transform(img)
    img = img.view(1, 3, 384, 384)
    # inference
    img = img.to(device)
    pred = model(img)
    softmax = nn.Softmax(dim=1)
    pred = softmax(pred)[0][1].item()
    pred = round(pred*10, 2)
    
    return pred


def check_id(out_csvPath):
    ids = set()
    if os.path.exists(out_csvPath):
        df = pd.read_csv(out_csvPath)
        ls_id = df['uuid'].tolist()
        ids.update(ls_id)
    return ids


if __name__ == "__main__":
    
    # directory that stores all the models
    model_load_path = "./models" 
    Path(model_load_path).mkdir(parents=True, exist_ok=True)
    # main output directory
    out_Path = "./sample_output" 
    Path(out_Path).mkdir(parents=True, exist_ok=True)
    # input file path
    in_Path = "../../download_imgs/sample_output/all/img_paths.csv" 
    
    # download model
    print('Downloading models...')
    snapshot_download(repo_id="Jiani11/human-perception-place-pulse", allow_patterns=["*.pth", "README.md"], local_dir=model_load_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device:{} ".format(device))
    for p in perception:
        # Configure logger (uncomment this section if you wish to log images that failed to run)
        # current_time = datetime.now().strftime("%Y%m%d_%H%M")
        # logging.basicConfig(filename=os.path.join(out_Path, f'{p}_{current_time}.log'), format='%(asctime)s %(message)s', filemode='w') 
        # logger=logging.getLogger() 
        # logger.setLevel(logging.INFO) 
        # load model
        model_path = model_load_path + model_dict[p]
        model = torch.load(model_path, map_location=torch.device(device))  
        model = nn.DataParallel(model)
        model = model.to(device)
        print("######### current: {}  #########".format(p))
        model.eval()
        out_csvPath = os.path.join(out_Path, p+'.csv')
        if os.path.exists(out_csvPath) == False:
            df = pd.DataFrame(columns=['uuid',p])
            df.to_csv(out_csvPath, index=False)
        data_arr = []
        input_df = pd.read_csv(in_Path)
        alr_ids = check_id(out_csvPath)
        input_df = input_df[~input_df['uuid'].isin(alr_ids)].reset_index(drop=True)
        for _, img in tqdm(input_df.iterrows(),total=len(input_df)): 
            uuid = img['uuid']
            try:
                img_path = img['path']
                score = predict(model,img_path,device)
                data_arr = [uuid,score]
                df=pd.DataFrame(data_arr).T
                df.to_csv(out_csvPath, mode='a', header=False, index=False)  # save scores into csv
            except:
                print(f"Failed for: {uuid}")
                #logger.info(f"Failed for: {uuid}") # uncomment this if you wish to log images that failed to run
        print("Completed perception prediction for: {}".format(p))


        
    