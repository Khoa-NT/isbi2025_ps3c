"""Training script for Pap Smear Cell Classification"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

from sklearn.metrics import classification_report
from tqdm import tqdm

import wandb

from utils.pytorch_utils import seed_worker, SeedAll


###------ Dataset Class -------###
class PapSmearDataset(Dataset):
    """Dataset class for Pap Smear Cell Classification
    
    Args:
        data_dir (str): Root directory containing class folders
        transform (callable, optional): Transform to be applied to images
        merge_bothcells (bool): Whether to merge bothcells class with unhealthy
    """
    def __init__(self, data_dir, transform=None, merge_bothcells=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        ### Create class mapping
        if merge_bothcells:
            self.class_map = {
                'healthy': 0,
                'unhealthy': 1,
                'bothcells': 1,  ### Merged with unhealthy
                'rubbish': 2
            }
        else:
            self.class_map = {
                'healthy': 0,
                'unhealthy': 1,
                'bothcells': 2,
                'rubbish': 3
            }
        
        ### Create list of (image_path, label) tuples
        self.samples = []
        for class_name in self.class_map.keys():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((img_path, class_name))
    

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        
        ### Add error handling for missing files
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_map[class_name]
        return image, label


###------ Model Class -------###
class PapSmearClassifier(nn.Module):
    """Classifier model using timm pretrained models
    
    Args:
        model_name (str): Name of the timm model to use
        num_classes (int): Number of classes to classify
        pretrained (bool): Whether to use pretrained weights
    """
    def __init__(self, model_name='efficientnet_b0', num_classes=3, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.pretrained_cfg = self.model.pretrained_cfg
            
    def forward(self, x):
        return self.model(x)


###------ Training Function -------###
def train_epoch(model, dataloader, criterion, optimizer, device, merge_bothcells=True):
    """Training function for one epoch"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    ### Define target names based on merge_bothcells
    target_names = ['healthy', 'unhealthy', 'rubbish'] if merge_bothcells else \
                    ['healthy', 'unhealthy', 'bothcells', 'rubbish']
    
    pbar = tqdm(dataloader, desc='Training', ncols=120)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        ### Store labels and predictions for metrics
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
        ### Update progress bar
        correct = sum(1 for x, y in zip(all_predictions, all_labels) if x == y)
        total = len(all_labels)
        acc = 100.*correct/total
        pbar.set_postfix({'loss': f'{running_loss/total:.5f}', 
                          'acc': f'{acc:.5f}%'})
    
    ### Generate classification report with correct target names
    report = classification_report(all_labels, all_predictions, 
                                   target_names=target_names,
                                   digits=5)
    
    ### We can reuse the `acc` variable to calculate accuracy if we want to save some computation
    ### instead of re-calculating the accuracy from scratch:
    ### running_acc = 100.*sum(1 for x, y in zip(all_predictions, all_labels) if x == y)/len(all_labels)
    running_acc = acc
    
    return running_loss/len(dataloader), running_acc, report


###------ Validation Function -------###
def validate(model, dataloader, criterion, device):
    """Validation function
    Haven't used this function yet because there is no validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(dataloader), correct/total


###------ Main Function -------###
def main():
    ### Parse arguments
    ### TODO: Add validation set
    ### TODO: Use Hydra to manage configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/isbi2025-ps3c-train-dataset')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')
    parser.add_argument('--load_ckpt', type=str, default='')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--merge_bothcells', action='store_true')
    parser.add_argument('--export_each_epoch', action='store_true')
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    ### Initialize wandb
    wandb_run = wandb.init(
        entity="GUGC_ISBI2025_PS3C",
        project="isbi2025-ps3c",
        name=f"{args.model_name}_{'3class' if args.merge_bothcells else '4class'}", ### Feel free to change the name
        notes=args.notes, ### Note for the experiment
        tags=[args.model_name, '3class' if args.merge_bothcells else '4class'], ### Tags for the experiment
        config=args, ### Get config from argparse
        config_exclude_keys=["notes"] ### Exclude `args.notes` from wandb config to prevent redundancy
    )
    wandb_run.define_metric("train/loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/acc", summary="max", step_metric="epoch")
    wandb_run.define_metric("val/loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("val/acc", summary="max", step_metric="epoch")


    ### Create random generator collection and set seed for reproducibility
    rng = SeedAll(args.random_seed)
    
    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Create model
    num_classes = 3 if args.merge_bothcells else 4
    model = PapSmearClassifier(args.model_name, num_classes=num_classes)
    
    ### Load checkpoint if provided
    if args.load_ckpt:
        print(f"Loading checkpoint from {args.load_ckpt}")
        model.load_state_dict(torch.load(args.load_ckpt))
    else:
        print("No checkpoint provided, training/fine-tuning from scratch")
    
    ### Move model to device
    model = model.to(device)
    model.train()

    ### Data transforms
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    
    ### Create datasets
    train_dataset = PapSmearDataset(
        data_dir=args.data_dir,
        transform=transform,
        merge_bothcells=args.merge_bothcells
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers,
                              worker_init_fn=seed_worker, generator=rng.torch_generator)
    
    ### Create model save path with configuration
    model_name = f"best_model_{args.model_name}_{'3class' if args.merge_bothcells else '4class'}.pth"
    ckpt_dir_path = Path(f"ckpt/{args.model_name}")
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir_path / model_name
    
    ### Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    ### Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        
        ### Pass merge_bothcells to train_epoch
        train_loss, train_acc, train_report = train_epoch(model, train_loader, criterion, 
                                               optimizer, device, 
                                               merge_bothcells=args.merge_bothcells)
        
        ### Evaluate model on validation set
        ### TODO: Uncomment this when we have a validation set
        # val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print('\nClassification Training Report:')
        print(train_report)
        
        wandb_run.log({
            "train/loss": train_loss,
            "train/acc": train_acc,

            ### TODO: Uncomment this when we have a validation set
            # "val/loss": val_loss,
            # "val/acc": val_acc,
        })
        
        scheduler.step(train_loss)
        
        ### Save best model with configuration in filename
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f'Saved best model as {ckpt_path} at epoch {epoch+1}')

        ### Save each epoch model for analysis
        if args.export_each_epoch:
            model_name = f"model_{epoch+1}.pth"
            temp_ckpt_path = ckpt_dir_path / model_name
            torch.save(model.state_dict(), temp_ckpt_path)
            print(f'Saved model as {temp_ckpt_path} at epoch {epoch+1}')
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()


# Train with bothcells merged with unhealthy (recommended)
# python train.py --merge_bothcells --num_epochs 100

# Or train keeping bothcells as separate class
# python train.py --num_epochs 100


