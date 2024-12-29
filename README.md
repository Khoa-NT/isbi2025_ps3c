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
```bash
./train.sh
```
Will train the model and save the best model to `ckpt/` folder.


## Submission
```bash
./predict.sh
```
Will predict on the test set and save the results for submission to `submission/` folder.
Also, will save the predicted results to the folder containing the ckpt file.

## Analysis
### Predicting on training set
```bash
./predict_train.sh
```
Will predict on the training set and save the results to the folder containing the ckpt file.
If no ckpt file is provided, will use the pretrained model and save the results to the folder with the name of the pretrained model.



## TODO
- [ ] Add Validation set / Cross-validation
- [ ] Use [WandB](https://wandb.ai/site) to log training and prediction
- [ ] Add batch size for prediction script


## Docker
Khoa's environment is available in [docker/khoa/khoa.md](docker/khoa/khoa.md).
