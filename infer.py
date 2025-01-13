"""Inference script for Pap Smear Cell Classification"""

import argparse
import pandas as pd
from pathlib import Path


import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import timm
from train import PapSmearClassifier
from utils.pytorch_utils import seed_worker, SeedAll


###------ Prediction Function -------###
def prediction(model, dataloader, df, idx_to_class, label_col, device,
               dataset_type, ckpt_path=None, pretrained_dir=None):
    """Run prediction on the dataset and export results
    
    Args:
        model (nn.Module): The model to use for prediction
        dataloader (DataLoader): DataLoader containing the dataset
        df (pd.DataFrame): DataFrame to store predictions
        idx_to_class (dict): Mapping from index to class names
        label_col (str): Column name for predictions
        device (torch.device): Device to run prediction on
        dataset_type (str): Either 'train' or 'test'
        ckpt_path (Path, optional): Path to checkpoint. Defaults to None
        pretrained_dir (Path, optional): Directory for pretrained model. Defaults to None
    """
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
            pred_path = ckpt_path.parent / "train_predictions.csv"
        else:
            pred_path = pretrained_dir / f"train_predictions_PreTrained_{model.model_name}.csv"
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
            raise ValueError(f"Error loading image {img_path}: {e}")
            
        return image, idx


###------ Main Function -------###
def main():
    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['train', 'test'], default='test',
                        help='Choose between training or test dataset (default: test)')
    parser.add_argument('--infer_mode', type=str, choices=['prediction', 'extract_features'],
                        default='prediction', help='Inference mode (default: prediction)')
    parser.add_argument('--extract_mode', type=str, choices=['pooled', 'pooled_all', 'classifier_token'], 
                        default='pooled', help='Extract mode (default: pooled). '
                       'pooled: extract pooled features from the last layer. '
                       'pooled_all: extract all features from the last layer included classifier token.'
                       'classifier_token: extract features from the classifier token.'
                       )
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
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()


    ### Create random generator collection and set seed for reproducibility
    rng = SeedAll(args.random_seed)

    ### Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Create model path based on configuration
    if args.load_ckpt:
        ckpt_path = Path(args.load_ckpt)
        print(f"Loading checkpoint from {ckpt_path=}")
    else:
        if args.dataset == 'test' and args.infer_mode == 'prediction':
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
    
    ### Create probability columns only for prediction mode
    if args.infer_mode == 'prediction':
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
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=rng.torch_generator
    )

    ### Run inference based on mode
    if args.infer_mode == 'prediction':
        prediction(
            model=model,
            dataloader=dataloader,
            df=df,
            idx_to_class=idx_to_class,
            label_col=label_col,
            device=device,
            dataset_type=args.dataset,
            ckpt_path=ckpt_path if args.load_ckpt else None,
            pretrained_dir=pretrained_dir if not args.load_ckpt else None
        )

    elif args.infer_mode == 'extract_features':
        ### ------------------------------- Preprocessing extract mode ------------------------------- ###
        ### Extract pooled features based on architecture
        if args.extract_mode == 'pooled':
            ### Set the classifier to Identity
            ### https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva.py#L553
            ### Will be normalized by the layer norm
            model.model.reset_classifier(0)

            ### The extract function is the forward function
            extract_func = model

        ### Extract all features from the last layer included classifier token
        ### In this code (https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva.py#L688),
        ### the global pooling operation skips the classifier token. Therefore, we need to extract
        ### features from the last layer that includes the classifier token.
        ### Is not normalized by the layer norm
        elif args.extract_mode == 'pooled_all':
            def extract_func(x):
                features = model.model.forward_features(x) ### (batch_size, token_len, num_features)
                return features.mean(dim=1) ### (batch_size, num_features)

        ### Extract features from the classifier token
        ### Is not normalized by the layer norm
        elif args.extract_mode == 'classifier_token':
            def extract_func(x):
                features = model.model.forward_features(x) ### (batch_size, token_len, num_features)
                return features[:, 0, :] ### (batch_size, num_features)

        ### ------------------------------- Extracting features ------------------------------- ###
        with torch.no_grad():
            for images, indices in tqdm(dataloader, desc='Extracting features', ncols=120):
                images = images.to(device)
                
                ### Get features
                features = extract_func(images) ### (batch_size, num_features)
                features = features.detach().clone().cpu().numpy()
            
                ### Store features in the dataframe based on indices
                df_features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
                df.loc[indices.tolist(), df_features.columns] = df_features.values

        ### ------------------------------- Export features ------------------------------- ###
        if args.load_ckpt:
            ### Create feature directory
            feature_dir = Path("extracted_features") / args.model_name / ckpt_path.stem
            feature_dir.mkdir(parents=True, exist_ok=True)

            ### Create feature file path
            file_path = feature_dir / f"{args.dataset}_{args.extract_mode}_features_ckpt_{ckpt_path.stem}.csv"
        else:
            ### Create feature directory
            feature_dir = Path("extracted_features") / args.model_name / "PreTrained"
            feature_dir.mkdir(parents=True, exist_ok=True)

            ### Create feature file path
            file_path = feature_dir / f"{args.dataset}_{args.extract_mode}_features_PreTrained_{args.model_name}.csv"
        
        df.to_csv(file_path, index=False, columns=['image_name'] + [f'feature_{i}' for i in range(features.shape[1])])
        print(f"\nFeatures saved to {file_path=}")



if __name__ == '__main__':
    ### Run python infer.py --help for more information
    ### Check the predict.sh & extract_features.sh files for example usage
    main()
