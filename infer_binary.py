"""Inference script for Binary Pap Smear Cell Classification"""

import argparse
import pandas as pd
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import timm
from train_binary import BinaryPapSmearClassifier
from utils.pytorch_utils import seed_worker, SeedAll


###------ Dataset Class -------###
class InferenceBinaryDataset(Dataset):
    """Dataset class for binary inference
    
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
            raise ValueError(f"Error loading image {img_path}: {e}")
            
        return image, idx


###------ Prediction Function -------###
def prediction(model, dataloader, df, class_0, class_1, label_col, device,
               dataset_type, ckpt_path=None, pretrained_dir=None):
    """Run prediction on the dataset and export results
    
    Args:
        model (nn.Module): The model to use for prediction
        dataloader (DataLoader): DataLoader containing the dataset
        df (pd.DataFrame): DataFrame to store predictions
        class_0 (List[str]): List of class names mapped to label 0
        class_1 (List[str]): List of class names mapped to label 1
        label_col (str): Column name for predictions
        device (torch.device): Device to run prediction on
        dataset_type (str): Either 'train' or 'test'
        ckpt_path (Path, optional): Path to checkpoint. Defaults to None
        pretrained_dir (Path, optional): Directory for pretrained model. Defaults to None
    """
    ### Create class mapping
    class_0_str = '+'.join(class_0)
    class_1_str = '+'.join(class_1)
    idx_to_class = {0: class_0_str, 1: class_1_str}
    
    ### ------------------------------- Predicting ------------------------------- ###
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
    if dataset_type == 'train':
        ### Save predictions with all columns
        if ckpt_path:
            pred_path = ckpt_path.parent / "train_predictions_binary.csv"
        else:
            pred_path = pretrained_dir / f"train_predictions_binary_PreTrained_{model.model_name}.csv"
        df.to_csv(pred_path, index=False)
        print(f"\nPredictions saved to {pred_path=}")
    else:
        ### Save predictions with specific columns (no capitalization)
        submission_path = Path("submission")
        submission_path.mkdir(parents=True, exist_ok=True)
        submission_path = submission_path / f"predictions_binary_{ckpt_path.stem}.csv"
        df.to_csv(submission_path, index=False, columns=['image_name', 'label'])
        print(f"\nPredictions saved to {submission_path=}")

        ### Save predictions with all columns to ckpt_path
        pred_path = ckpt_path.parent / "predictions_binary.csv"
        df.to_csv(pred_path, index=False)
        print(f"\nPredictions saved to {pred_path=}")


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
    parser.add_argument('--load_ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--class_0', nargs='+', default=['rubbish'],
                        help='Classes mapped to label 0 (separated by space)')
    parser.add_argument('--class_1', nargs='+', 
                        default=['healthy', 'unhealthy', 'bothcells'],
                        help='Classes mapped to label 1 (separated by space)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    ### Create random generator collection and set seed for reproducibility
    rng = SeedAll(args.random_seed)

    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Create model path based on configuration
    ckpt_path = Path(args.load_ckpt)
    print(f"Loading checkpoint from {ckpt_path=}")

    ### Create and load model
    model = BinaryPapSmearClassifier(args.model_name)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model = model.to(device)
    model.eval()

    ### Data transforms
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    
    ### Read dataset CSV file
    data_dir = Path(args.data_dir_train if args.dataset == 'train' else args.data_dir_test)
    csv_path = args.csv_path_train if args.dataset == 'train' else args.csv_path_test
    df = pd.read_csv(csv_path)
    
    ### Create probability columns
    label_col = 'pred_label' if args.dataset == 'train' else 'label'
    df[label_col] = ''
    class_0_str = '+'.join(args.class_0)
    class_1_str = '+'.join(args.class_1)
    df[f'prob_{class_0_str}'] = 0.0
    df[f'prob_{class_1_str}'] = 0.0
    
    ### Create dataset and dataloader
    dataset = InferenceBinaryDataset(
        df=df,
        data_dir=data_dir,
        transform=transform,
        is_train=(args.dataset == 'train')
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

    ### Run prediction
    prediction(
        model=model,
        dataloader=dataloader,
        df=df,
        class_0=args.class_0,
        class_1=args.class_1,
        label_col=label_col,
        device=device,
        dataset_type=args.dataset,
        ckpt_path=ckpt_path
    )


if __name__ == '__main__':
    main() 