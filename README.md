# isbi2025_ps3c
Pap Smear Cell Classification Challenge (PS3C) 


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


## Submission
### Multi-class classification
```bash
./predict.sh
```
Will predict on the test set and save the results for submission to `submission/` folder.
Also, will save the predicted results to the folder containing the ckpt file.
Check `predict.sh` for example usage.

### Binary classification
```bash
./predict_binary.sh
```
Will predict on the test set and save the results for submission to `submission/` folder.
Also, will save the predicted results to the folder containing the ckpt file.
Can run on multiple models at once and merge the results.
Check `predict_binary.sh` for example usage.

After predicting all binary models, run `merge_binary_predictions.py` to merge the results.
Check `merge_predictions.sh` & `merge_binary_predictions.py` for example usage.


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

#### Extract features:
+ eva02_base_patch14_448: [Google Drive link (Temporary)](https://drive.google.com/drive/u/5/folders/1tFFUHJ8rU1nnDzqKT7Fnzs2BJ9P6egPW)




## TODO
- [x] Extracting features from the model
- [x] Add batch size for prediction script
- [x] Use [WandB](https://wandb.ai/site) to log training and prediction
- [x] Binary classification strategy: Rubbish vs (Healthy/Unhealthy) -> Healthy vs Unhealthy
- [ ] Add Validation set / Cross-validation
- [ ] Docker

## Ideas
- **Idea 1:** Remove confusing data by identifying the file names of False Positives (FP) and False Negatives (FN) and excluding them from the training set.

- **Idea 2:** Since classifying unhealthy cases is key to success, assign a higher loss weight to misclassifications of unhealthy cases.

## Docker
Khoa's environment is available in [docker/khoa/khoa.md](docker/khoa/khoa.md).
