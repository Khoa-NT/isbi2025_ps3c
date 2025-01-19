"""Inference script for SAE Classifier"""

import argparse
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train_sae_classifier import SAEVectorDataset, SAEClassifier
from utils.pytorch_utils import seed_worker, SeedAll


###------ Prediction Function -------###
def prediction(model, dataloader, df, idx_to_class, device, ckpt_path):
    """Run prediction on the dataset and export results
    
    Args:
        model (nn.Module): The model to use for prediction
        dataloader (DataLoader): DataLoader containing the dataset
        df (pd.DataFrame): DataFrame to store predictions
        idx_to_class (dict): Mapping from index to class names
        device (torch.device): Device to run prediction on
        ckpt_path (Path): Path to checkpoint
    """
    ### Create prediction columns if they don't exist
    df['pred_label'] = ''
    for class_name in idx_to_class.values():
        df[f'prob_{class_name}'] = 0.0

    ### ------------------------------- Predicting ------------------------------- ###
    with torch.no_grad():
        for features, indices in tqdm(dataloader, desc='Predicting', ncols=120):
            features = features.to(device)
            
            ### Get predictions
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            ### Update dataframe with predictions
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                idx = indices[i].item()
                df.at[idx, 'pred_label'] = idx_to_class[pred.item()]
                for class_idx, class_name in idx_to_class.items():
                    df.at[idx, f'prob_{class_name}'] = prob[class_idx].cpu().item()

    ### ------------------------------- Export predictions ------------------------------- ###
    ### Save predictions with all columns to ckpt_path
    pred_path = ckpt_path.parent / "predictions.csv"
    df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")

    ### Save submission format
    submission_path = Path("submission")
    submission_path.mkdir(parents=True, exist_ok=True)
    submission_path = submission_path / f"predictions_{ckpt_path.stem}.csv"
    df.to_csv(submission_path, index=False, columns=['image_name', 'pred_label'])
    print(f"Submission saved to {submission_path}")


###------ Inference Dataset Class -------###
class InferenceSAEDataset(Dataset):
    """Dataset class for SAE inference
    
    Args:
        df (pd.DataFrame): DataFrame containing image paths
        sae_df (pd.DataFrame): DataFrame containing SAE features
    """
    def __init__(self, df, sae_df):
        self.df = df
        self.sae_df = sae_df
        
        ### Get feature columns
        self.feature_cols = [col for col in self.sae_df.columns if col.startswith('sparse_feature_')]
        self.features = self.sae_df[self.feature_cols].values
    

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        return features, idx


###------ Main Function -------###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
                       default='dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-test-dataset.csv',
                       help='Path to CSV file with image paths')
    parser.add_argument('--sae_csv', type=str, required=True,
                       help='Path to CSV file with SAE features')
    parser.add_argument('--load_ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--merge_bothcells', action='store_true',
                       help='Merge bothcells class with unhealthy')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512, 256],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    ### Create random generator collection and set seed for reproducibility
    rng = SeedAll(args.random_seed)

    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Create class mapping
    if args.merge_bothcells:
        idx_to_class = {0: 'healthy', 1: 'unhealthy', 2: 'rubbish'} ### bothcells merged into unhealthy
    else:
        idx_to_class = {0: 'healthy', 1: 'unhealthy', 2: 'bothcells', 3: 'rubbish'}
    
    ### Read dataset CSV files
    df = pd.read_csv(args.csv_path)
    sae_df = pd.read_csv(args.sae_csv)
    
    ### Create dataset and dataloader
    dataset = InferenceSAEDataset(df, sae_df)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=rng.torch_generator
    )

    ### Create and load model
    num_classes = 3 if args.merge_bothcells else 4
    model = SAEClassifier(
        input_dim=len(dataset.feature_cols),
        hidden_dims=args.hidden_dims,
        num_classes=num_classes,
        dropout=args.dropout
    )
    
    ### Load checkpoint
    ckpt_path = Path(args.load_ckpt)
    print(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model = model.to(device)
    model.eval()

    ### Run prediction
    prediction(
        model=model,
        dataloader=dataloader,
        df=df,
        idx_to_class=idx_to_class,
        device=device,
        ckpt_path=ckpt_path
    )


if __name__ == '__main__':
    main() 