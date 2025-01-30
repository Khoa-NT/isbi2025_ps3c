"""Training script for Student Model with Soft Labels from Teacher"""

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
from train_binary import BinaryPapSmearClassifier


###------ Dataset Class -------###
class SoftLabelPapSmearDataset(Dataset):
    """Dataset class for Pap Smear Cell Classification with Soft Labels
    
    Args:
        data_dir (str): Root directory containing class folders
        csv_path (str): Path to CSV file containing image paths and labels
        student_transform (callable, optional): Transform to be applied for student model
        teacher_transform (callable, optional): Transform to be applied for teacher model
    """
    def __init__(self, data_dir, csv_path, student_transform=None, teacher_transform=None):
        self.data_dir = Path(data_dir)
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        
        ### Create class mapping - fixed order for soft labels
        self.class_map = {
            'healthy': 0,
            'unhealthy': 1,
            'bothcells': 2,
            'rubbish': 3
        }
        
        ### Read CSV file and create samples list
        df = pd.read_csv(csv_path)
        
        ### Create list of (image_path, label) tuples from CSV
        self.samples = []
        for _, row in df.iterrows():
            if pd.notna(row['image_name']) and pd.notna(row['label']):
                img_path = self.data_dir / row['label'] / f"{row['image_name']}"
                if img_path.exists():
                    self.samples.append((img_path, row['label']))
    

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        
        ### Add error handling for missing files
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        ### Load image
        image = Image.open(img_path).convert('RGB')
        
        ### Apply transforms
        student_image = self.student_transform(image) if self.student_transform else image
        teacher_image = self.teacher_transform(image) if self.teacher_transform else image
        
        label = self.class_map[class_name]
        return student_image, teacher_image, label, class_name


###------ Model Class -------###
class StudentPapSmearClassifier(nn.Module):
    """Student classifier model using timm pretrained models
    
    Args:
        model_name (str): Name of the timm model to use
        pretrained (bool): Whether to use pretrained weights
    """
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=4)
        self.pretrained_cfg = self.model.pretrained_cfg
            
    def forward(self, x):
        return self.model(x)


###------ Training Function -------###
def train_epoch(student_model, teacher_model, dataloader, criterion, 
                student_optimizer, teacher_optimizer, device, train_teacher=False):
    """Training function for one epoch"""
    student_model.train()
    if train_teacher:
        teacher_model.train()
    else:
        teacher_model.eval()
        
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    ### Define target names
    target_names = ['healthy', 'unhealthy', 'bothcells', 'rubbish']
    
    pbar = tqdm(dataloader, desc='Training', ncols=120)
    for student_images, teacher_images, labels, class_names in pbar:
        student_images = student_images.to(device)
        teacher_images = teacher_images.to(device)
        labels = labels.to(device)
        
        ### Create soft labels
        soft_labels = torch.zeros(len(labels), 4).to(device)  ### 4 classes
        
        ### Set ground truth labels
        soft_labels.scatter_(1, labels.unsqueeze(1), 1.0)
        
        ### Create mask for unhealthy/bothcells images
        relevant_mask = torch.tensor([(cn in ['unhealthy', 'bothcells']) for cn in class_names], 
                                   device=device)
        
        if relevant_mask.any():
            ### Get teacher predictions for relevant images in batch
            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_images[relevant_mask])
                teacher_probs = torch.softmax(teacher_outputs, dim=1)
                
                ### Update soft labels for relevant images
                soft_labels[relevant_mask, 1] = teacher_probs[:, 0]  ### unhealthy
                soft_labels[relevant_mask, 2] = teacher_probs[:, 1]  ### bothcells
        
        ### Forward pass through student
        student_optimizer.zero_grad()
        student_outputs = student_model(student_images)
        student_probs = torch.softmax(student_outputs, dim=1)
        
        ### Calculate student loss
        student_loss = criterion(student_probs, soft_labels)
        
        ### Store true labels and predictions for metrics
        predicted = student_probs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
        ### Train teacher if specified
        if train_teacher and teacher_optimizer is not None:
            teacher_optimizer.zero_grad()
            
            ### Process all images with teacher
            teacher_outputs = teacher_model(teacher_images)
            teacher_probs = torch.softmax(teacher_outputs, dim=1)
            
            ### Create binary labels for teacher (batch_size, 2)
            teacher_labels = torch.zeros(len(class_names), 2).to(device)
            
            ### Set labels based on class
            for i, class_name in enumerate(class_names):
                if class_name == 'unhealthy':
                    teacher_labels[i, 0] = 1.0  ### [1, 0] for unhealthy
                elif class_name == 'bothcells':
                    teacher_labels[i, 1] = 1.0  ### [0, 1] for bothcells
                ### else: remains [0, 0] for other classes
            
            ### Calculate teacher loss
            teacher_loss = criterion(teacher_probs, teacher_labels)
            total_loss = student_loss + teacher_loss
        else:
            total_loss = student_loss
        
        ### Backward pass
        total_loss.backward()
        student_optimizer.step()
        if train_teacher and teacher_optimizer is not None:
            teacher_optimizer.step()
        
        running_loss += total_loss.item()
        
        ### Update progress bar with current accuracy
        correct = sum(1 for x, y in zip(all_predictions, all_labels) if x == y)
        total = len(all_labels)
        acc = 100.*correct/total
        pbar.set_postfix({'loss': f'{running_loss/total:.5f}', 
                         'acc': f'{acc:.5f}%'})
    
    ### Generate classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=target_names,
                                 digits=5)
    
    return running_loss/len(dataloader), acc, report


###------ Main Function -------###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/isbi2025-ps3c-train-dataset')
    parser.add_argument('--csv_path', type=str, 
                       default='dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-train-dataset.csv')
    parser.add_argument('--student_model', type=str, default='efficientnet_b0')
    parser.add_argument('--teacher_model', type=str, default='eva02_base_patch14_448')
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--train_teacher', action='store_true',
                       help='Whether to train teacher model together')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--student_lr', type=float, default=1e-4)
    parser.add_argument('--teacher_lr', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    ### Initialize wandb
    wandb_run = wandb.init(
        entity="GUGC_ISBI2025_PS3C",
        project="isbi2025-ps3c",
        name=f"student_{args.student_model}_teacher_{args.teacher_model}"
             f"{'_train_teacher' if args.train_teacher else ''}",
        tags=[
            f"student_{args.student_model}",
            f"teacher_{args.teacher_model}",
            "train_teacher" if args.train_teacher else "fixed_teacher",
        ],
        config=args,
        notes=f"Student-Teacher training with {args.student_model} student and {args.teacher_model} teacher. "
              f"{'Training both models.' if args.train_teacher else 'Teacher model is fixed.'}"
    )
    wandb_run.define_metric("train/loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/acc", summary="max", step_metric="epoch")

    ### Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = SeedAll(args.random_seed)

    ### Create and load teacher model
    teacher_model = BinaryPapSmearClassifier(args.teacher_model)
    teacher_model.load_state_dict(torch.load(args.teacher_ckpt, weights_only=True))
    teacher_model = teacher_model.to(device)

    ### Create student model
    student_model = StudentPapSmearClassifier(args.student_model)
    student_model = student_model.to(device)

    ### Data transforms
    student_cfg = timm.data.resolve_data_config(student_model.pretrained_cfg)
    teacher_cfg = timm.data.resolve_data_config(teacher_model.pretrained_cfg)
    student_transform = timm.data.create_transform(**student_cfg)
    teacher_transform = timm.data.create_transform(**teacher_cfg)

    ### Create dataset and dataloader
    dataset = SoftLabelPapSmearDataset(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        student_transform=student_transform,
        teacher_transform=teacher_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=rng.torch_generator
    )

    ### Loss function and optimizers
    criterion = nn.MSELoss()  ### Using MSE loss for soft labels
    student_optimizer = optim.AdamW(student_model.parameters(), lr=args.student_lr)
    teacher_optimizer = optim.AdamW(teacher_model.parameters(), lr=args.teacher_lr) if args.train_teacher else None

    ### Create checkpoint directory
    ckpt_dir = Path(f"ckpt/student_{args.student_model}")
    if args.train_teacher:
        ckpt_dir = ckpt_dir / "train_teacher"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ### Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        
        train_loss, train_acc, train_report = train_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            dataloader=dataloader,
            criterion=criterion,
            student_optimizer=student_optimizer,
            teacher_optimizer=teacher_optimizer,
            device=device,
            train_teacher=args.train_teacher
        )
        
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print('\nClassification Training Report:')
        print(train_report)
        
        wandb_run.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
        })
        
        ### Save best student model
        if train_loss < best_loss:
            best_loss = train_loss
            student_ckpt_path = ckpt_dir / f"best_student_{args.student_model}.pth"
            torch.save(student_model.state_dict(), student_ckpt_path)
            print(f'Saved best student model to {student_ckpt_path}')
            
            ### Save teacher model if training together
            if args.train_teacher:
                teacher_ckpt_path = ckpt_dir / f"best_teacher_{args.teacher_model}.pth"
                torch.save(teacher_model.state_dict(), teacher_ckpt_path)
                print(f'Saved best teacher model to {teacher_ckpt_path}')

    print("\nTraining completed!")


if __name__ == '__main__':
    main() 