from dataloaders import GlobalStreetScapes,create_dataloaders,create_train_val_datasets
from model import GlobalStreetScapesClassificationTrainer
from utils import visualize_dataset, KFoldDataset,load_yaml_config
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch
import os
import hydra
from omegaconf import OmegaConf
from uuid import uuid4

@hydra.main(config_name="config.yaml")
def main(cfg):
    
    seed_everything(42) # set seed for reproducibility
    
    torch.set_float32_matmul_precision('high') 
    
    CONFIG = cfg 
    wconf = OmegaConf.to_container(cfg, resolve=[True|False])
    
    root_path = CONFIG.get("ROOT_PATH")
    
    if CONFIG.get('RUN_ENVIRONMENT')=='KAROLINA':
        os.environ["WANDB_DIR"] = '/scratch/project/dd-23-73/data/'
    
    ## Use datasets on weather split into precipitation and non-precipitation
    if CONFIG.get('SPLIT_WEATHER_ATTRIBUTE'):    
        attr_dict = {'field_of_view': 'field_of_view.csv',
                    'glare': 'glare.csv',
                    'lighting_condition': 'lighting_condition.csv',
                    'pano_status': 'pano_status.csv',
                    'platform': 'platform.csv',
                    'quality': 'quality.csv',
                    'reflection': 'reflection.csv',
                    'view_direction': 'view_direction.csv',
                    'weather_precipitation': 'weather_precipitation.csv',
                    'weather_nonprecipitation': 'weather_nonprecipitation.csv'}
    else:
        attr_dict = {'field_of_view': 'field_of_view.csv',
                    'glare': 'glare.csv',
                    'lighting_condition': 'lighting_condition.csv',
                    'pano_status': 'pano_status.csv',
                    'platform': 'platform.csv',
                    'quality': 'quality.csv',
                    'reflection': 'reflection.csv', 
                    'view_direction': 'view_direction.csv',
                    'weather': 'weather.csv'}
        
    for attribute,filename in attr_dict.items():

        print(f"Training {attribute} model")  
        
        run_name = attribute  
        ckpt_path = f'./checkpoints/{run_name}_{CONFIG.get("WEIGHT_STRATEGY")}'
        unique_id = str(uuid4())

        if CONFIG.get('CROSS_VALIDATION'):
            folds = KFoldDataset(CSV_file=os.path.join(root_path,'train',filename),
                                        label_column=attribute, 
                                        img_path=os.path.join(root_path, "img/"),
                                        n_splits=CONFIG.get('NUM_K_FOLDS'))  # for example, 5-fold CV    
            
        else:
            
            train_ds, val_ds,class_weights = create_train_val_datasets(CSV_file=os.path.join(root_path,'train',filename),
                                                         label_column=attribute, 
                                                         img_path=os.path.join(root_path, "img/"),
                                                         class_weighting_strategy=CONFIG.get('WEIGHT_STRATEGY'), 
                                                         val_split=CONFIG.get('VAL_SPLIT_PROP')
                                                         )
            
        test_ds = GlobalStreetScapes(CSV_file=os.path.join(root_path,'test',filename),
                                    img_path=os.path.join(root_path, "img/"),
                                    label_column=attribute,
                                    class_weighting_strategy=None)

        if not CONFIG.get('CROSS_VALIDATION'):
            # Get dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(train_ds, 
                                                                    val_ds, 
                                                                    test_ds,
                                                                    batch_size=CONFIG.get('BATCH_SIZE'),
                                                                    num_workers=CONFIG.get('NUM_WORKERS')
                                                                    )
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            
            checkpoint_callback = ModelCheckpoint(monitor=CONFIG.get('ES_METRIC'), 
                                      save_top_k=1,
                                      save_last=False,
                                      save_weights_only=False,
                                      verbose=True,
                                      mode='min',
                                      every_n_epochs=1,
                                      dirpath=ckpt_path,
                                      filename=f'{unique_id}'+'-'+'{epoch}_{validation_loss:.2f}'
                                      )
        
        early_stop_callback = EarlyStopping(monitor=CONFIG.get('ES_METRIC'), 
                                                    patience=CONFIG.get('PATIENCE'),
                                                    mode=CONFIG.get('ES_MODE'),)
        
        
            

        if not CONFIG.get('CROSS_VALIDATION'):
            
            wandb_logger = WandbLogger(log_model=False,
                                        project="global-street-scapes",
                                        group='karolina-training-Final',
                                        config=wconf,
                                        reinit=True,
                                        name=str(unique_id+'-'+run_name)
                                        ) 
        
            ## Setting up the trainer
            
            trainer = Trainer(max_epochs = CONFIG.get('MAX_EPOCHS'), 
                                devices='auto',
                                check_val_every_n_epoch=CONFIG.get('VAL_EVERY_N_EPOCHS'),
                                callbacks=[early_stop_callback,checkpoint_callback],
                                logger = wandb_logger,
                                gradient_clip_val = CONFIG.get('GRAD_CLIPPING') if CONFIG.get('GRAD_CLIPPING') else 0,
                                precision = CONFIG.get('PRECISION'),
                                )
        
        ## kwargs dictionary:
        kwgs = {'attribute': attribute,
                'weighted': CONFIG.get('WEIGHT_LOSS'), 
                'weighting_strategy': CONFIG.get('WEIGHT_STRATEGY'),
                'kfold': False if not CONFIG.get('CROSS_VALIDATION') else True,
                'uuid': unique_id,
                }
        
        if not CONFIG.get('CROSS_VALIDATION'):
            ## Initialize the model
            pl_model = GlobalStreetScapesClassificationTrainer(lr=CONFIG.get('LR'),
                                                            num_classes=len(train_ds.label2index),
                                                            weight=class_weights,
                                                            kwargs=kwgs,
                                                            class_mapping=train_ds.label2index,
                                                            pretrained=CONFIG.get('PRETRAINED'),)
        
        if CONFIG.get('CROSS_VALIDATION'):
            
            for fold, (train_ds, val_ds) in enumerate(folds):
                
                kwgs['kfold'] = fold
                
                wandb_logger = WandbLogger(log_model=False,
                                    project="global-street-scapes",
                                    group='karolina-CV-training-Final',
                                    config=wconf,
                                    reinit=True,
                                    name=f"CROSS_VALIDATION_{attribute}_fold_{fold}") 
                
                trainer = Trainer(max_epochs = CONFIG.get('MAX_EPOCHS'), 
                                devices='auto',#cfg.meta.num_gpus,
                                check_val_every_n_epoch=1,
                                strategy='auto',
                                callbacks=[early_stop_callback],
                                logger = wandb_logger,
                                gradient_clip_val = CONFIG.get('GRAD_CLIPPING') if CONFIG.get('GRAD_CLIPPING') else 0,
                                precision = CONFIG.get('PRECISION'),
                                )
                
                ## Initialize the model
                pl_model = GlobalStreetScapesClassificationTrainer(lr=CONFIG.get('LR'),
                                                            num_classes=len(train_ds.label2index),
                                                            weight=train_ds.weights,
                                                            class_mapping=train_ds.label2index,
                                                            kwargs=kwgs,
                                                            pretrained=CONFIG.get('PRETRAINED'),)
                ## Get dataloaders
                train_loader, val_loader, test_loader = create_dataloaders(train_ds, 
                                                                    val_ds, 
                                                                    test_ds,
                                                                    batch_size=CONFIG.get('BATCH_SIZE'),
                                                                    num_workers=CONFIG.get('NUM_WORKERS'))
                
                try:
                    # Train the model
                    trainer.fit(pl_model, train_loader, val_loader)
                except Exception as e:   
                    print(f"ERROR: Training failed for this attribute {attribute} \n\n EXCEPTION: {e}")
                    continue
                
                ## Test the model
                # Create a new Trainer instance for testing with single device configuration (To mitigate multi-GPU/multi-node issues)
                test_trainer = Trainer(devices=1,num_nodes=1,strategy='auto',logger=wandb_logger,)

                # Test the model
                test_trainer.test(pl_model, test_loader)
                
                wandb.finish()
        
        else:    
            try:
                
                ## Train the model
                trainer.fit(pl_model, train_loader, val_loader)
            except Exception as e:
                print(f"ERROR: Training failed for this attribute {attribute} \n\n EXCEPTION: {e}")
                # continue
            
            # Create a new Trainer instance for testing with single device configuration (To mitigate multi-GPU/multi-node issues)
            test_trainer = Trainer(devices=1, num_nodes=1,strategy='auto',logger=wandb_logger,)

            # Test the model
            test_trainer.test(pl_model, test_loader)
            
        wandb.finish() # end the wandb run for attribute
    

if __name__ == "__main__":
    main()
