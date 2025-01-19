"""Train Sparse Autoencoder on extracted features"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.pytorch_utils import seed_worker, SeedAll
import wandb


###------ Dataset Class -------###
class FeatureDataset(Dataset):
    """Dataset for extracted features
    
    Args:
        csv_path (str): Path to CSV file containing features
        feature_cols (list): List of feature column names
    """
    def __init__(self, csv_path, feature_cols):
        self.df = pd.read_csv(csv_path)
        self.features = self.df[feature_cols].values
        self.features = torch.FloatTensor(self.features)
    

    def __len__(self):
        return len(self.features)
    

    def __getitem__(self, idx):
        return self.features[idx]


###------ Model Class -------###
class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder model
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        activation (str): Activation function. Defaults to "relu"
        bias (bool): Whether to use bias. Defaults to True
    """
    def __init__(self, input_dim, hidden_dim, activation="relu", bias=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        ### Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=bias)
        
        ### Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=bias)
        
        ### Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    

    def encode(self, x):
        return self.activation(self.encoder(x))
    

    def decode(self, h):
        return self.decoder(h)
    

    def forward(self, x):
        h = self.encode(x)
        reconstruction = self.decode(h)
        return reconstruction, h


###------ Trainer Class -------###
class SparseAutoencoderTrainer:
    """Trainer for Sparse Autoencoder
    
    Args:
        model (SparseAutoencoder): Model to train
        learning_rate (float): Learning rate. Defaults to 1e-3
        l1_coefficient (float): L1 regularization coefficient. Defaults to 1e-3
        device (str): Device to use. Defaults to "cuda" if available
    """
    def __init__(self, model, learning_rate=1e-3, l1_coefficient=1e-3,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.l1_coefficient = l1_coefficient
        self.mse_loss = nn.MSELoss()
    

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        ### Move batch to device
        batch = batch.to(self.device)
        
        ### Forward pass
        reconstruction, encoded = self.model(batch)
        
        ### Calculate losses
        reconstruction_loss = self.mse_loss(reconstruction, batch)
        l1_loss = torch.mean(torch.abs(encoded))
        total_loss = reconstruction_loss + self.l1_coefficient * l1_loss
        
        ### Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "l1_loss": l1_loss.item(),
            "mean_activation": encoded.mean().item(),
            "sparsity": (encoded == 0).float().mean().item()
        }
    

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        epoch_stats = []
        for batch in tqdm(dataloader, desc='Training', ncols=120):
            stats = self.train_step(batch)
            epoch_stats.append(stats)
            
        ### Average stats across batches
        return {k: np.mean([s[k] for s in epoch_stats]) for k in epoch_stats[0].keys()}


###------ Feature Extraction Function -------###
def extract_sparse_features(model, dataloader, device):
    """Extract sparse features using trained model
    
    Args:
        model (SparseAutoencoder): Trained model
        dataloader (DataLoader): DataLoader containing features
        device (str): Device to use
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features', ncols=120):
            batch = batch.to(device)
            encoded = model.encode(batch)
            all_features.append(encoded.cpu())
            
    return torch.cat(all_features, dim=0)


###------ Main Function -------###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path_train', type=str, required=True,
                        help='Path to training CSV file with extracted features')
    parser.add_argument('--csv_path_test', type=str, required=True,
                        help='Path to test CSV file with extracted features')
    parser.add_argument('--hidden_dim_multiplier', type=float, default=2.0,
                        help='Multiplier for hidden dimension relative to input dimension')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l1_coefficient', type=float, default=1e-3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    ### ---------------- Initialize wandb ---------------- ###
    wandb_run = wandb.init(
        entity="GUGC_ISBI2025_PS3C",
        project="isbi2025-ps3c",
        name=f"SAE_features_{Path(args.csv_path_train).stem}",
        notes=f"Training SAE on features from {Path(args.csv_path_train).name}",
        tags=["SAE", f"hidden_dim_{args.hidden_dim_multiplier:.0f}x"],
        config=args,
        config_exclude_keys=["notes"]
    )
    wandb_run.define_metric("train/total_loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/reconstruction_loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/l1_loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/mean_activation", summary="mean", step_metric="epoch")
    wandb_run.define_metric("train/sparsity", summary="max", step_metric="epoch")

    ### Set random seed
    rng = SeedAll(args.random_seed)
    
    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Load training data and get feature columns
    train_df = pd.read_csv(args.csv_path_train)
    ### feature_cols = [col for col in train_df.columns if col.startswith('feature_')] ### Naive way
    feature_cols = train_df.columns[1:].tolist() ### exclude image_name column which is the first column
    input_dim = len(feature_cols)
    hidden_dim = int(input_dim * args.hidden_dim_multiplier)
    
    ### Create datasets
    train_dataset = FeatureDataset(args.csv_path_train, feature_cols)
    test_dataset = FeatureDataset(args.csv_path_test, feature_cols)
    
    ### Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=rng.torch_generator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    ### Initialize model and trainer
    model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    )
    
    trainer = SparseAutoencoderTrainer(
        model=model,
        learning_rate=args.learning_rate,
        l1_coefficient=args.l1_coefficient,
        device=device
    )
    
    ### Training loop
    print(f"\nTraining SAE with {input_dim} input features -> {hidden_dim} hidden features")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        stats = trainer.train_epoch(train_loader)
        
        print(f"Total Loss: {stats['total_loss']:.4f}")
        print(f"Reconstruction Loss: {stats['reconstruction_loss']:.4f}")
        print(f"L1 Loss: {stats['l1_loss']:.4f}")
        print(f"Mean Activation: {stats['mean_activation']:.4f}")
        print(f"Sparsity: {stats['sparsity']:.4f}")
        
        ### Log to wandb
        wandb_run.log({
            "epoch": epoch + 1,
            "train/total_loss": stats['total_loss'],
            "train/reconstruction_loss": stats['reconstruction_loss'],
            "train/l1_loss": stats['l1_loss'],
            "train/mean_activation": stats['mean_activation'],
            "train/sparsity": stats['sparsity']
        })
    
    ### Extract and save sparse features
    print("\nExtracting sparse features...")
    
    ### Create output directory
    train_path = Path(args.csv_path_train)
    output_dir = train_path.parent / f"sae_{args.hidden_dim_multiplier:.0f}x_features_{Path(args.csv_path_train).stem}"
    output_dir.mkdir(exist_ok=True)
    
    ### Extract and save training features
    train_sparse_features = extract_sparse_features(model, train_loader, device)
    train_df = pd.read_csv(args.csv_path_train)
    train_sparse_df = pd.DataFrame(
        train_sparse_features.numpy(),
        columns=[f'sparse_feature_{i}' for i in range(hidden_dim)]
    )
    train_sparse_df['image_name'] = train_df['image_name']
    train_output_path = output_dir / f"train_sparse_{args.hidden_dim_multiplier:.0f}x_features_{Path(args.csv_path_train).stem}.csv"
    train_sparse_df.to_csv(train_output_path, index=False)
    print(f"\nSaved training sparse features to {train_output_path}")
    
    ### Extract and save test features
    test_sparse_features = extract_sparse_features(model, test_loader, device)
    test_df = pd.read_csv(args.csv_path_test)
    test_sparse_df = pd.DataFrame(
        test_sparse_features.numpy(),
        columns=[f'sparse_feature_{i}' for i in range(hidden_dim)]
    )
    test_sparse_df['image_name'] = test_df['image_name']
    test_output_path = output_dir / f"test_sparse_{args.hidden_dim_multiplier:.0f}x_features_{Path(args.csv_path_test).stem}.csv"
    test_sparse_df.to_csv(test_output_path, index=False)
    print(f"Saved test sparse features to {test_output_path}")
    
    ### Save model
    model_path = output_dir / f"sae_model_{args.hidden_dim_multiplier:.0f}x_features_{Path(args.csv_path_train).stem}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    ### Close wandb run
    wandb_run.finish()


if __name__ == '__main__':
    main() 