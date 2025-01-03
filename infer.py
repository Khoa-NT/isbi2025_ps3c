"""Inference script for Pap Smear Cell Classification"""

import argparse
import pandas as pd
from pathlib import Path
import os

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import timm
from train import PapSmearClassifier


###------ Prediction Function -------###
def predict_batch(model, images, transform, device):
    """Predict batch of images
    
    Args:
        model (nn.Module): The trained model
        images (List[PIL.Image]): List of PIL images
        transform (transforms): Image transformations
        device (torch.device): Device to run prediction on
        
    Returns:
        tuple: Predictions and probabilities for the batch
    """
    ### Transform all images
    batch = torch.stack([transform(img) for img in images])
    batch = batch.to(device)
    
    with torch.no_grad():
        output = model(batch)
        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1)
    
    return preds, probs


###------ Dataset Class -------###
class InferenceDataset(Dataset):
    """Dataset class for inference
    
    Args:
        df (pd.DataFrame): DataFrame containing image paths
        data_dir (Path): Root directory containing images
        transform (callable): Transform to be applied to images
        is_train (bool): Whether this is training set (affects path construction)
    """
    def __init__(self, df, data_dir, transform=None, is_train=False):
        self.df = df
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        ### Filter out rows with empty image_name
        self.df = self.df[self.df['image_name'].notna()].reset_index(drop=True)
    

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        ### Construct image path based on dataset type
        if self.is_train:
            img_path = self.data_dir / row['label'] / row['image_name']
        else:
            img_path = self.data_dir / row['image_name']
            
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            ### Return a blank image in case of error
            image = torch.zeros((3, 224, 224))
            
        return image, idx


###------ Main Function -------###
def main():
    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['train', 'test'], default='test',
                       help='Choose between training or test dataset (default: test)')
    parser.add_argument('--data_dir_train', type=str, 
                       default='dataset/isbi2025-ps3c-train-dataset')
    parser.add_argument('--data_dir_test', type=str, 
                       default='dataset/isbi2025-ps3c-test-dataset')
    parser.add_argument('--csv_path_train', type=str,
                       default='dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-train-dataset.csv')
    parser.add_argument('--csv_path_test', type=str,
                       default='dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-test-dataset.csv')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')
    parser.add_argument('--load_ckpt', type=str, default='')
    parser.add_argument('--merge_bothcells', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers for data loading (default: 8)')
    args = parser.parse_args()

    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Create model path based on configuration
    if args.load_ckpt:
        ckpt_path = Path(args.load_ckpt)
        print(f"Loading checkpoint from {ckpt_path=}")
    else:
        if args.dataset == 'test':
            raise ValueError("Checkpoint must be provided for test set prediction")
        print(f"Using pretrained model {args.model_name=}")
        pretrained_dir = Path("ckpt") / args.model_name
        pretrained_dir.mkdir(parents=True, exist_ok=True)

    ### Create and load model
    num_classes = 3 if args.merge_bothcells else 4
    model = PapSmearClassifier(args.model_name, num_classes=num_classes)
    if args.load_ckpt:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model = model.to(device)
    model.eval()

    ### Data transforms
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    
    ### Class mapping based on configuration
    if args.merge_bothcells:
        idx_to_class = {0: 'healthy', 1: 'unhealthy', 2: 'rubbish'} ### bothcells is merged into unhealthy
    else:
        idx_to_class = {0: 'healthy', 1: 'unhealthy', 2: 'bothcells', 3: 'rubbish'}
    
    ### Read dataset CSV file
    data_dir = Path(args.data_dir_train if args.dataset == 'train' else args.data_dir_test)
    csv_path = args.csv_path_train if args.dataset == 'train' else args.csv_path_test
    df = pd.read_csv(csv_path)
    
    ### Create probability columns
    label_col = 'pred_label' if args.dataset == 'train' else 'label'
    df[label_col] = ''
    for class_name in idx_to_class.values():
        df[f'prob_{class_name}'] = 0.0
    
    ### Create dataset and dataloader
    dataset = InferenceDataset(
        df=df,
        data_dir=data_dir,
        transform=transform,
        is_train=(args.dataset == 'train')
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  ### Use user-specified number of workers
        pin_memory=True
    )

    ### Predict
    model.eval()
    with torch.no_grad():
        for images, indices in tqdm(dataloader, desc='Predicting', ncols=120):
            images = images.to(device)
            
            ### Get predictions
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            ### Update dataframe with predictions
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                idx = indices[i].item()
                df.at[idx, label_col] = idx_to_class[pred.item()]
                for class_idx, class_name in idx_to_class.items():
                    df.at[idx, f'prob_{class_name}'] = prob[class_idx].cpu().item()

    ### ------------------------------- Export predictions ------------------------------- ###
    if args.dataset == 'train':
        ### Save predictions with all columns
        if args.load_ckpt:
            pred_path = ckpt_path.parent / "train_predictions.csv"
        else:
            pred_path = pretrained_dir / f"train_predictions_PreTrained_{args.model_name}.csv"
        df.to_csv(pred_path, index=False)
        print(f"\nPredictions saved to {pred_path=}")
    else:
        ### Save predictions with specific columns (no capitalization)
        submission_path = Path("submission")
        submission_path.mkdir(parents=True, exist_ok=True)
        submission_path = submission_path / f"predictions_{ckpt_path.stem}.csv"
        df.to_csv(submission_path, index=False, columns=['image_name', 'label'])
        print(f"\nPredictions saved to {submission_path=}")

        ### Save predictions with all columns to ckpt_path
        pred_path = ckpt_path.parent / "predictions.csv"
        df.to_csv(pred_path, index=False)
        print(f"\nPredictions saved to {pred_path=}")


if __name__ == '__main__':
    ### Check the predict.sh file for example usage
    main()
