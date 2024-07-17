
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader
from dataloaders import GlobalStreetScapes
import yaml

def exists(obj):
    return obj is not None

def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_mapping(series):
    # Get unique values and create a dictionary
    unique_values = series.unique()
    mapping = {i: val for i, val in enumerate(unique_values)}
    return mapping


def denormalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes an image tensor by multiplying by the standard deviation and adding back the mean for each channel.

    Args:
        image_tensor (torch.Tensor): The normalized image tensor.
        mean (list[float]): The mean for each channel used in the normalization transform.
        std (list[float]): The standard deviation for each channel used in the normalization transform.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = torch.tensor(mean).reshape(1, -1, 1, 1)
    std = torch.tensor(std).reshape(1, -1, 1, 1)
    return image_tensor * std + mean

def visualize_dataset(dataset, num_images=25, randomize=False,figsize=(15,15)):
    # Get the class names and number of classes
    num_classes = len(dataset.index2label.values())

    # Set up the figure with a grid of images
    fig, axes = plt.subplots(nrows=int(np.sqrt(num_images)), ncols=int(np.sqrt(num_images)), figsize=figsize)

    # Randomize the image order if requested
    if randomize:
        indices = np.random.choice(len(dataset), num_images)
    else:
        indices = np.arange(num_images)

    # Loop over the selected images and plot them
    for i, ax in zip(indices, axes.flatten()):
        # Get the image and label
        image, label = dataset[i]

        # Plot the image and set the title
        ax.imshow(denormalize(image).squeeze().transpose(0,-1).transpose(1,0))
        ax.set_title(dataset.index2label[label])
        ax.axis('off')

    # Add a title to the plot
    plt.suptitle("Samples of Dataset with {} number of classes".format(num_classes))

    # Show the plot
    plt.show()
    
    
def KFoldDataset(CSV_file, label_column, img_path='./TRAIN_TEST/img/', class_weighting_strategy='uniform', n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    data = pd.read_csv(CSV_file)
    labels = data[label_column]

    datasets = []
    for fold_index, (train_index, test_index) in enumerate(skf.split(data, labels)):
        # Subset data for training and validation
        train_data = data.iloc[train_index].reset_index(drop=True)
        validation_data = data.iloc[test_index].reset_index(drop=True)
        
        # Save the subset data temporarily
        train_csv = CSV_file.replace('.csv', f'_train_fold_{fold_index}.csv')
        validation_csv = CSV_file.replace('.csv', f'_validation_fold_{fold_index}.csv')
        
        train_data.to_csv(train_csv, index=False)
        validation_data.to_csv(validation_csv, index=False)

        # Create datasets using the subset CSVs
        train_dataset = GlobalStreetScapes(train_csv, label_column, img_path, class_weighting_strategy)
        validation_dataset = GlobalStreetScapes(validation_csv, label_column, img_path, class_weighting_strategy)
        
        # Cleanup temporary CSVs
        os.remove(train_csv)
        os.remove(validation_csv)
        
        datasets.append((train_dataset, validation_dataset))

    return datasets

