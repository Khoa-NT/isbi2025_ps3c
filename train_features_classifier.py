"""Training script for classifying extracted features with feature masking"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import classification_report
import wandb

from utils.pytorch_utils import seed_worker, SeedAll


###------ Dataset Class -------###
class FeatureDataset(Dataset):
    """Dataset class for extracted features with masking
    
    Args:
        csv_path (str): Path to original CSV file containing image paths and labels
        features_csv_path (str): Path to CSV file containing extracted features
        masking_path (str): Path to CSV file containing feature masking
        masking_method (str): Method to use for feature masking
        merge_bothcells (bool): Whether to merge bothcells class with unhealthy
    """
    def __init__(self, csv_path, features_csv_path, masking_path, masking_method, merge_bothcells=True):
        ### Read CSV files
        self.df = pd.read_csv(csv_path)
        self.features_df = pd.read_csv(features_csv_path)
        
        ### Read masking file and get feature mask
        df_masking = pd.read_csv(masking_path)
        masking_method = masking_method.replace('_', ' ')
        if masking_method == "None":
            ### No masking
            self.feature_mask = np.ones(self.features_df.shape[1]-1, dtype=bool)
        else:
            ### Masking
            self.feature_mask = df_masking[df_masking["Model"] == masking_method].iloc[0, 1:].values.astype(bool)
        
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
        
        ### Get feature columns from extracted features (excluding image_name)
        self.feature_cols = self.features_df.columns[1:].tolist()
        ### Apply feature masking
        self.masked_features = self.features_df[self.feature_cols].values[:, self.feature_mask]
        
        ### Get labels from original CSV and convert using class map
        self.labels = [self.class_map[label] for label in self.df['label'].values]
        
        ### Calculate class weights
        label_counts = pd.Series(self.labels).value_counts()
        max_count = label_counts.max()
        self.class_weights = torch.zeros(len(set(self.labels)))
        for label, count in label_counts.items():
            self.class_weights[label] = max_count/count
    

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.masked_features[idx])
        label = self.labels[idx]
        return features, label


###------ Model Class -------###
class FeatureClassifier(nn.Module):
    """MLP classifier for extracted features
    
    Args:
        input_dim (int): Input dimension (number of masked features)
        hidden_dims (list): List of hidden layer dimensions
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=3, dropout=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        ### Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        ### Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.model(x)


###------ Training Function -------###
def train_epoch(model, dataloader, criterion, optimizer, device, merge_bothcells=True):
    """Training function for one epoch"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    pbar = tqdm(dataloader, desc='Training', ncols=120)
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
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
    
    ### Generate classification report
    target_names = ['healthy', 'unhealthy', 'rubbish'] if merge_bothcells else \
                    ['healthy', 'unhealthy', 'bothcells', 'rubbish']
    report = classification_report(all_labels, all_predictions, 
                                   target_names=target_names,
                                   digits=5)
    
    return running_loss/len(dataloader), acc, report


###------ Main Function -------###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path_train', type=str,
                       default='dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-train-dataset.csv',
                       help='Path to original training CSV file with labels')
    parser.add_argument('--features_train_csv', type=str, required=True,
                       help='Path to training CSV file with extracted features')
    parser.add_argument('--masking_path', type=str, required=True,
                       help='Path to CSV file containing feature masking')
    parser.add_argument('--masking_method', type=str, required=True,
                       choices=['Gradient_Boosting', 'Random_Forest', 'Logistic_Regression', 'None'],
                       help='Method to use for feature masking')
    parser.add_argument('--merge_bothcells', action='store_true',
                       help='Merge bothcells class with unhealthy')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights in loss function')
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    ### Initialize wandb
    wandb_run = wandb.init(
        entity="GUGC_ISBI2025_PS3C",
        project="isbi2025-ps3c",
        name=f"Feature_classifier_{args.masking_method}_{'3class' if args.merge_bothcells else '4class'}"
             f"{'_weighted' if args.use_class_weights else ''}",
        notes=args.notes,
        tags=['Feature_classifier', 
              args.masking_method,
              '3class' if args.merge_bothcells else '4class',
              'weighted' if args.use_class_weights else 'unweighted'],
        config=args,
        config_exclude_keys=["notes"]
    )
    wandb_run.define_metric("train/loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/acc", summary="max", step_metric="epoch")

    ### Set random seed
    rng = SeedAll(args.random_seed)
    
    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Create dataset
    train_dataset = FeatureDataset(
        csv_path=args.csv_path_train,
        features_csv_path=args.features_train_csv,
        masking_path=args.masking_path,
        masking_method=args.masking_method,
        merge_bothcells=args.merge_bothcells
    )
    
    ### Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=rng.torch_generator
    )
    
    ### Create model
    input_dim = train_dataset.masked_features.shape[1]  ### Number of masked features
    num_classes = 3 if args.merge_bothcells else 4
    model = FeatureClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    
    ### Setup loss function and optimizer
    if args.use_class_weights:
        class_weights = train_dataset.class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\nUsing weighted CrossEntropyLoss with class weights:")
        for i, weight in enumerate(class_weights):
            print(f"Class {i}: {weight:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\nUsing standard CrossEntropyLoss without class weights")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    ### Create checkpoint directory
    ckpt_dir = Path(f"ckpt/feature_classifier/{Path(args.features_train_csv).stem}/{args.masking_method}")
    if args.use_class_weights:
        ckpt_dir = ckpt_dir / "use_class_weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ### Training loop
    print(f"\nTraining {args.masking_method} on {args.features_train_csv}")
    print(f"Input dimension: {input_dim}")
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        
        train_loss, train_acc, train_report = train_epoch(
            model, train_loader, criterion, optimizer, device, merge_bothcells=args.merge_bothcells
        )
        scheduler.step(train_loss)
        
        ### Logging
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print('Classification Report:')
        print(train_report)
        
        wandb_run.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        ### Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            model_name = f"best_model_feature_classifier_{args.masking_method}_{'3class' if args.merge_bothcells else '4class'}"
            if args.use_class_weights:
                model_name += "_weighted"
            model_name += ".pth"
            torch.save(model.state_dict(), ckpt_dir / model_name)
            print(f'Saved best model at epoch {epoch+1}')
    
    wandb_run.finish()
    print("\nTraining completed!")


if __name__ == '__main__':
    main() 