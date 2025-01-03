"""Prediction script for Pap Smear Cell Classification"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import timm
from train import PapSmearClassifier


###------ Prediction Function -------###
def predict(model, image_path, transform, device):
    """Predict single image
    
    Args:
        model (nn.Module): The trained model
        image_path (str): Path to the image
        transform (transforms): Image transformations
        device (torch.device): Device to run prediction on
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
    
    return pred, prob[0]


###------ Main Function -------###
def main():
    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/isbi2025-ps3c-test-dataset')
    parser.add_argument('--csv_path', type=str, default='dataset/pap-smear-cell-classification-challenge/isbi2025-ps3c-test-dataset.csv')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')
    parser.add_argument('--load_ckpt', type=str, default='')
    parser.add_argument('--merge_bothcells', action='store_true')
    args = parser.parse_args()

    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Create model path based on configuration
    ckpt_path = Path(args.load_ckpt)
    print(f"Loading checkpoint from {ckpt_path}")

    ### Create and load model
    num_classes = 3 if args.merge_bothcells else 4
    model = PapSmearClassifier(args.model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path))
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
    
    ### Read test CSV file
    test_df = pd.read_csv(args.csv_path)
    
    ### Create probability columns
    test_df['label'] = ''
    for class_name in idx_to_class.values():
        test_df[f'prob_{class_name}'] = 0.0
    
    ### Predict
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Predicting', ncols=120):
        ### Get image path based on image name column
        img_path = Path(args.data_dir) / row['image_name']
        
        ### Skip if image_name is empty
        if not isinstance(row['image_name'], str):
            continue
            
        pred, prob = predict(model, img_path, transform, device)
        
        ### Update dataframe directly
        test_df.at[idx, 'label'] = idx_to_class[pred]
        
        ### Add probability for each class from the tensor
        for class_idx, class_name in idx_to_class.items():
            test_df.at[idx, f'prob_{class_name}'] = prob[class_idx].cpu().item()
    

    ### ------------------------------- Export predictions ------------------------------- ###
    ### Save predictions with specific columns (no capitalization)
    submission_path = Path("submission")
    submission_path.mkdir(parents=True, exist_ok=True)
    submission_path = submission_path / f"predictions_{ckpt_path.stem}.csv"
    test_df.to_csv(submission_path, index=False, columns=['image_name', 'label'])
    print(f"\nPredictions saved to {submission_path}")

    ### Save predictions with all columns to ckpt_path
    pred_path = ckpt_path.parent / "predictions.csv"
    test_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")


if __name__ == '__main__':
    main()

# # For 3-class model (with bothcells merged)
# python predict.py --merge_bothcells

# # For 4-class model
# python predict.py