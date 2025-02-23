# Pap Smear Cell Classification Challenge (PS3C)

This is the repository for the  [PS3C challenge](https://www.kaggle.com/competitions/pap-smear-cell-classification-challenge/overview) of team **Khoa_NT**.

![Overview](assets/Overview_khoa.png)


## Data
Download the data follow this [instructions](https://www.kaggle.com/competitions/pap-smear-cell-classification-challenge/data) and put it in the `data/` folder.
The data should be organized as follows:
```
dataset
│
├── isbi2025-ps3c-test-dataset
│   ├── isbi2025_ps3c_test_image_00001.png
│   ├── isbi2025_ps3c_test_image_00002.png
│   ├── ...
│   ├── isbi2025_ps3c_test_image_18158.png
│   └── isbi2025_ps3c_test_image_18159.png
│
├── isbi2025-ps3c-train-dataset
│   ├── bothcells
│   ├── healthy
│   ├── rubbish
│   └── unhealthy
│
└── pap-smear-cell-classification-challenge
    ├── isbi2025-ps3c-test-dataset.csv
    └── isbi2025-ps3c-train-dataset.csv
```


## Training
### Multi-class classification
```bash
./train.sh
```
Will train the model and save the best model to `ckpt/` folder.

### Binary classification
```bash
./train_binary.sh
```
Will train the model and save the best model to `ckpt/` folder.
Check `train_binary.sh` for example usage.


### Features Classification

Using the extracted features to train the classifier.

#### Training
```bash
./train_features_classifier.sh
```

#### Inference
```bash
./infer_features_classifier.sh
```

## Submission
### Multi-class classification
```bash
./predict.sh
```
Will predict on the test set and save the results for submission to `submission/` folder.
Also, will save the predicted results to the folder containing the ckpt file.
Check `predict.sh` for example usage.

### Binary classification (☀️/🌙)
```bash
./predict_binary.sh
```
Will predict on the test set and save the results for submission to `submission/` folder.
Also, will save the predicted results to the folder containing the ckpt file.
Check `predict_binary.sh` for example usage.

**Note:** After predicting all binary models, run `merge_binary_predictions.py` to merge the results in `submission/` folder. 
For example, If we train the 1st model with **rubbish** vs **healthy_unhealthy_bothcells**, the predicted **healthy_unhealthy_bothcells** of the 1st model will be replaced by the prediction **(healthy vs unhealthy_bothcells)**  of the 2nd model.
+ Check `merge_predictions.sh` & `merge_binary_predictions.py` for example usage.
+ Check the [Binary Classification Result](#binary-classification-result) below to see the merged result.

## Analysis
### Predicting on training set
```bash
./predict_train.sh
```
Will predict on the training set and save the results to the folder containing the ckpt file.
If no ckpt file is provided, will use the pretrained model and save the results to the folder with the name of the pretrained model.

### Extracting features from the model
```bash
./extract_features.sh
```
Will extract features from the model and save the results to `extracted_features/{model_name}` folder.
If no ckpt file is provided, will use the pretrained model and save the results to the folder with the name of the pretrained model.

Extract modes:
+ pooled: Extract features from the pooled output of the model. Usually pooled without the classifier token.
+ pooled_all: Extract features from all outputs of the model. Including the classifier token.
+ classifier_token: Extract features from the classifier token.

#### Extract features [Google Drive link (Temporary)](https://drive.google.com/drive/u/5/folders/1tFFUHJ8rU1nnDzqKT7Fnzs2BJ9P6egPW):
+ eva02_base_patch14_448: Features from the model overfit on the training set.
+ PreTrained_eva02_base_patch14_448: Features from the ImageNet pretrained model.


### Binary Classification Result
| Name | Model 1 | Model 2 | Model 3 |
|:----:|:-------:|:-------:|:-------:|
| rubbish   | ☀️ | - | ☀️ |
| healthy   | 🌙 | ☀️ | 🌙 |
| unhealthy | 🌙 | 🌙 | - |
| bothcells | 🌙 | 🌙 | - |


| Model Combination | Test Result | Comment |
|:-----------------:|:-----------:|:---------|
| 1 + 2             | 0.84982     |  |
| 3 + 2             | 0.85025     |  |
| 3                 | 0.84127     | 🤔 Get 0.84 with only submitting rubbish & healthy (without unhealthy & bothcells) |

### Importance-based Feature Selection
```bash
./importance_based_feature_selection/mask.sh
```
Will create a masking table based on the feature importances of machine learning models, including Logistic Regression, Gradient Boosting, and Random Forest. This masking table will then be used to perform importance-based feature selection.

## TODO
- [x] Extracting features from the model
- [x] Add batch size for prediction script
- [x] Use [WandB](https://wandb.ai/site) to log training and prediction
- [x] Binary classification strategy: Rubbish vs (Healthy/Unhealthy) -> Healthy vs Unhealthy
- [x] Training with the training set based on csv files (Idea 1)
- [x] Add class weights to the loss function (Idea 2)
- [x] Add SAE strategy: training and extracting SAE features from the extracted features (Classifier token, Pooled, Pooled_all)
- [x] Add training and predicting with features (Idea 3)
- [ ] Add Validation set / Cross-validation
- [ ] Docker

## Ideas
- [x] **Idea 1:** Remove confusing data by identifying the file names of False Positives (FP) and False Negatives (FN) and excluding them from the training set.
- [x] **Idea 2:** Since classifying unhealthy cases is key to success, assign a higher loss weight to misclassifications of unhealthy cases.
- [x] **Idea 3:** Create the mask to prune the extracted features (e.g., Eva02 or SAE's features). Then train the classifier with the pruned features.

## Docker
Khoa's environment is available in [docker/khoa/khoa.md](docker/khoa/khoa.md).
