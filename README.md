# isbi2025_ps3c
Pap Smear Cell Classification Challenge (PS3C) 


## Training
```bash
./train.sh
```
Will train the model and save the best model to `ckpt/` folder.


## Predicting
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