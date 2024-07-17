from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

class GlobalStreetScapes(torch.utils.data.Dataset):
    def __init__(self, CSV_file, label_column,img_path='./TRAIN_TEST/img/', class_weighting_strategy='uniform',indices=None,training=False):
        # Load the CSV file
        self.data = pd.read_csv(CSV_file)

        # Store the image path
        self.img_path = img_path

        # Save the label column
        self.labels = self.data[label_column]
        
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
            self.labels = self.labels.iloc[indices].reset_index(drop=True)

        # Create the label to index mapping
        self.label2index = {label: index for index, label in enumerate(self.labels.unique())}
        self.index2label = {index: label for label, index in self.label2index.items()}

        # Compute class weights (either "inverse" or "uniform")
        self.weights = self.compute_weights(class_weighting_strategy)

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        if training:
            self.transform = transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),  # Adds shearing and zooming
                                                transforms.RandomHorizontalFlip(),  # Adds horizontal flipping
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
                                            ])

    def compute_weights(self, strategy):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        
        if strategy == "inverse":
            weights = 1.0 / torch.tensor(counts, dtype=torch.float)
            weights /= weights.sum()  # Normalize the weights
        elif strategy == "uniform":
            class_fractions = torch.tensor(counts, dtype=torch.float) / total_samples
            desired_avg_weight = 1.0 / len(unique_labels)
            weights = desired_avg_weight / class_fractions
            weights /= weights.sum()  # Normalize the weights to sum to 1
        elif strategy == None:
            # No class weighting
            weights = None
        else:
            raise ValueError(f"Unsupported class weighting strategy: {strategy}")
    
        return weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load image
        img_path = os.path.join(self.img_path,self.data['uuid'].iloc[index]+'.jpeg') # Assumes all .jpeg images
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Get label
        label = self.labels[index]
        label_idx = self.label2index[label]

        return image, label_idx

class GlobalStreetScapes2(torch.utils.data.Dataset):
    def __init__(self, data, img_path='./TRAIN_TEST/img/',training=False):
        # Load the CSV file
        self.data = data

        # Store the image path
        self.img_path = img_path

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        if training:
            self.transform = transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),  # Adds shearing and zooming
                                                transforms.RandomHorizontalFlip(),  # Adds horizontal flipping
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
                                            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get image uuid
        image_uuid = self.data['uuid'].iloc[index]

        # Load image
        img_path = os.path.join(self.img_path, self.data['device'].iloc[index], self.data['folder'].iloc[index], image_uuid+'.jpeg') # Assumes all .jpeg images
        print(image_uuid)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, image_uuid

class GlobalStreetScapes_simple(torch.utils.data.Dataset):
    def __init__(self, data, training=False, path_field='path'):
        # Load the CSV file
        self.data = data
        self.path_field = path_field

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        if training:
            self.transform = transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),  # Adds shearing and zooming
                                                transforms.RandomHorizontalFlip(),  # Adds horizontal flipping
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
                                            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get image path
        img_path = self.data[self.path_field].iloc[index]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        uuid = os.path.basename(img_path).split('.')[0]

        return image, uuid

# def create_train_val_datasets(CSV_file, label_column, img_path='./TRAIN_TEST/img/', class_weighting_strategy='uniform', val_split=0.20):
#     # Create a dataset to get the total length
#     full_dataset = GlobalStreetScapes(CSV_file, label_column, img_path, class_weighting_strategy)
    
#     global_class_weights = full_dataset.weights

#     # Compute lengths of splits
#     total_size = len(full_dataset)
#     val_size = int(val_split * total_size)
#     train_size = total_size - val_size

#     # Use random_split to get indices for splits
#     train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

#     # Create datasets using the indices from the split
#     train_dataset = GlobalStreetScapes(CSV_file, label_column, img_path, None, indices=train_dataset.indices)
#     val_dataset = GlobalStreetScapes(CSV_file, label_column, img_path, None, indices=val_dataset.indices)

#     return train_dataset, val_dataset,global_class_weights




def create_train_val_datasets(CSV_file, label_column, img_path='./TRAIN_TEST/img/', class_weighting_strategy='uniform', val_split=0.20):
    # Create a dataset to get the total length
    full_dataset = GlobalStreetScapes(CSV_file, label_column, img_path, class_weighting_strategy)
    
    global_class_weights = full_dataset.weights

    # Using train_test_split for stratified split
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=full_dataset.labels, random_state=42)

    # Create datasets using the stratified splits
    train_dataset = GlobalStreetScapes(CSV_file, label_column, img_path, class_weighting_strategy, indices=train_indices, training=True)
    val_dataset = GlobalStreetScapes(CSV_file, label_column, img_path, class_weighting_strategy, indices=val_indices)

    return train_dataset, val_dataset, global_class_weights


def create_dataloaders(train_dataset, val_dataset, test_dataset=None, batch_size=64,num_workers=4):
    """
    Creates PyTorch dataloaders from given datasets.
    
    Args:
    - train_dataset: PyTorch Dataset for training data
    - val_dataset: PyTorch Dataset for validation data
    - test_dataset: PyTorch Dataset for test data (optional)
    - batch_size: Batch size for loading data (default is 64)

    Returns:
    - train_loader: Dataloader for training data
    - val_loader: Dataloader for validation data
    - test_loader: Dataloader for test data if provided, otherwise None
    """
    
    # Training dataloader (set shuffle=True for training)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Validation dataloader (no need to shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Test dataloader (no need to shuffle)
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader