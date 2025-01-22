"""Inference script for Feature Classifier"""

import argparse
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train_features_classifier import FeatureClassifier
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
    submission_path = Path("submission") / "feature_classifier"
    submission_path.mkdir(parents=True, exist_ok=True)
    submission_path = submission_path / f"predictions_{ckpt_path.stem}.csv"

    ### Rename the 'pred_label' to 'label'  
    df.rename(columns={'pred_label': 'label'}, inplace=True)

    ### Save submission format
    df.to_csv(submission_path, index=False, columns=['image_name', 'label'])
    print(f"Submission saved to {submission_path}")


###------ Inference Dataset Class -------###
class InferenceFeatureDataset(Dataset):
    """Dataset class for feature inference with masking
    
    Args:
        df (pd.DataFrame): DataFrame containing image paths
        features_df (pd.DataFrame): DataFrame containing extracted features
        masking_path (str): Path to CSV file containing feature masking
        masking_method (str): Method to use for feature masking
    """
    def __init__(self, df, features_df, masking_path, masking_method):
        self.df = df
        self.features_df = features_df
        
        ### Read masking file and get feature mask
        df_masking = pd.read_csv(masking_path)
        masking_method = masking_method.replace('_', ' ')
        self.feature_mask = df_masking[df_masking["Model"] == masking_method].iloc[0, 1:].values.astype(bool)
        
        ### Get feature columns from extracted features (excluding image_name)
        self.feature_cols = self.features_df.columns[1:].tolist()
        ### Apply feature masking
        self.masked_features = self.features_df[self.feature_cols].values[:, self.feature_mask]
    

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.masked_features[idx])
        return features, idx


###------ Main Function -------###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
                       default='dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-test-dataset.csv',
                       help='Path to CSV file with image paths')
    parser.add_argument('--features_csv', type=str, required=True,
                       help='Path to CSV file with extracted features')
    parser.add_argument('--masking_path', type=str, required=True,
                       help='Path to CSV file containing feature masking')
    parser.add_argument('--masking_method', type=str, required=True,
                       choices=['Gradient_Boosting', 'Random_Forest', 'Logistic_Regression'],
                       help='Method to use for feature masking')
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
    features_df = pd.read_csv(args.features_csv)
    
    ### Create dataset and dataloader
    dataset = InferenceFeatureDataset(
        df=df, 
        features_df=features_df,
        masking_path=args.masking_path,
        masking_method=args.masking_method
    )
    
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
    model = FeatureClassifier(
        input_dim=dataset.masked_features.shape[1],  ### Number of masked features
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