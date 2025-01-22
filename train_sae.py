"""Train Sparse Autoencoder on extracted features"""

import argparse
import sys
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
    def __init__(self, model, 
                 learning_rate=1e-3, 
                 l1_coefficient=1e-3,
                 decorr_coefficient=0.1,
                 entropy_coefficient=0.1,
                 num_epochs=None,
                 steps_per_epoch=None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        # self.optimizer = optim.RAdam(model.parameters(), lr=learning_rate)

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=20, factor=0.9)  ### patience=5, factor=0.5
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate, steps_per_epoch=steps_per_epoch, epochs=num_epochs, pct_start=0.1)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs//30) ### Step in one epoch

        self.l1_coefficient = l1_coefficient
        self.decorr_coefficient = decorr_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.mse_loss = nn.MSELoss()

    def l1_loss(self, activations):
        return torch.mean(torch.abs(activations)) * self.l1_coefficient
    
    def feature_decorrelation_loss(self, activations):
        ### Calculate correlation matrix
        corr = torch.corrcoef(activations.T)
        ### Penalize high correlations (excluding diagonal)
        mask = ~torch.eye(corr.shape[0], dtype=bool)
        return torch.mean(torch.abs(corr[mask])) * self.decorr_coefficient
    
    def entropy_loss(self, activations):
        probs = torch.softmax(activations, dim=1)
        return -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1)) * self.entropy_coefficient

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
        l1_loss = self.l1_loss(encoded)
        decorr_loss = self.feature_decorrelation_loss(encoded) if self.decorr_coefficient > 0 else torch.tensor(0.0)
        entropy_loss = self.entropy_loss(encoded) if self.entropy_coefficient > 0 else torch.tensor(0.0)
        total_loss = reconstruction_loss + l1_loss + decorr_loss + entropy_loss

        ### Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "l1_loss": l1_loss.item(),
            "decorr_loss": decorr_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "mean_activation": encoded.mean().item(),
            "sparsity": (encoded == 0).float().mean().item()
        }
    

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        epoch_stats = []
        for batch in tqdm(dataloader, desc='Training', ncols=120):
            stats = self.train_step(batch)
            epoch_stats.append(stats)
            
            ### Step OneCycleLR scheduler after each batch
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

        ### Average stats across batches
        avg_stats = {k: np.mean([s[k] for s in epoch_stats]) for k in epoch_stats[0].keys()}
        
        ### Step other schedulers once per epoch
        if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_stats['total_loss'])
            else:
                self.scheduler.step()
        
        return avg_stats


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
    parser.add_argument('--activation', type=str, default="relu",
                        help='Activation function: [relu, gelu]. Defaults to "relu"')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l1_coefficient', type=float, default=1e-3)
    parser.add_argument('--decorr_coefficient', type=float, default=0.1)
    parser.add_argument('--entropy_coefficient', type=float, default=0.1)
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
        tags=["SAE", f"hidden_dim_{args.hidden_dim_multiplier:.0f}x", f"activation_{args.activation}"],
        config=args,
        config_exclude_keys=["notes"]
    )
    wandb_run.define_metric("train/total_loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/reconstruction_loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/l1_loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/decorr_loss", summary="min", step_metric="epoch")
    wandb_run.define_metric("train/entropy_loss", summary="min", step_metric="epoch")
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
        hidden_dim=hidden_dim,
        activation=args.activation
    )
    
    trainer = SparseAutoencoderTrainer(
        model=model,
        learning_rate=args.learning_rate,
        l1_coefficient=args.l1_coefficient,
        decorr_coefficient=args.decorr_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        num_epochs=args.num_epochs,
        steps_per_epoch=len(train_loader),
        device=device
    )

    ### Create output directory
    train_path = Path(args.csv_path_train)
    output_dir = train_path.parent / f"sae_{args.hidden_dim_multiplier:.0f}x_features_{Path(args.csv_path_train).stem}" / f"activation_{args.activation}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ### Training loop
    print(f"\nTraining SAE with {input_dim} input features -> {hidden_dim} hidden features")
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        stats = trainer.train_epoch(train_loader)
        
        ### Log to wandb
        wandb_run.log({
            "epoch": epoch,
            "learning_rate": trainer.scheduler.get_last_lr()[0],
            "train/total_loss": stats['total_loss'],
            "train/reconstruction_loss": stats['reconstruction_loss'],
            "train/l1_loss": stats['l1_loss'],
            "train/decorr_loss": stats['decorr_loss'],
            "train/entropy_loss": stats['entropy_loss'],
            "train/mean_activation": stats['mean_activation'],
            "train/sparsity": stats['sparsity']
        })

        ### Save best model
        if stats['total_loss'] < best_loss:
            best_loss = stats['total_loss']
            best_epoch = epoch
            model_path = output_dir / f"sae_model_{args.hidden_dim_multiplier:.0f}x_features_{Path(args.csv_path_train).stem}.pth"
            torch.save(model.state_dict(), model_path)
            
        ### Print stats
        print(f"Total Loss: {stats['total_loss']:.4f}\t\tBest Loss: {best_loss:.4f} at Epoch {best_epoch}")
        print(f"Reconstruction Loss: {stats['reconstruction_loss']:.4f}")
        print(f"L1 Loss: {stats['l1_loss']:.4f}")
        print(f"Decorrelation Loss: {stats['decorr_loss']:.4f}")
        print(f"Entropy Loss: {stats['entropy_loss']:.4f}")
        print(f"Mean Activation: {stats['mean_activation']:.4f}")
        print(f"Sparsity: {stats['sparsity']:.4f}")

        ### Exit system if total loss is NaN
        if np.isnan(stats['total_loss']):
            print("Total loss is NaN. Exiting system.")
            wandb_run.finish()
            sys.exit(1)


    ### Extract and save sparse features
    print(f"\nExtracting sparse features from best model with loss {best_loss:.4f} at epoch {best_epoch}...")
    model.load_state_dict(torch.load(model_path))
        
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
    
    ### Close wandb run
    wandb_run.finish()


if __name__ == '__main__':
    main() 