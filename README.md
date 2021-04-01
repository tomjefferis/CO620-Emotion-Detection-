# CO620 Emotion Detection Python
 Python files, for emotion detection using ECG, GSR and Eye Tracking, Results are as classification reports and graphs for each model. To view on Github: [Here](https://github.com/tomjefferis/CO620-Emotion-Detection-/)
 
 ## Introduction
 
There are working examples of the emotion detection working with the processed data as well as my own feature extraction. emotiondetection.ipynb is the main jupyter notebook used, the mlDataTesting.ipynb is for testing. 

### Files / folders
#### Python files: 
hyperparameter, kbestfeatures, results and performance metrics are used for the final report. The file tenpointaverage was a test for extracting 10 point average for ecg and gsr features. binaryclass was used to change labels for valence and arousal classification 

#### Data: 
raw data is all in the raw data folder, processed data was all the data with extracted features used in the report, the main files to look at is the completedata.csv and completedatanoeye.csv these files contain the features used in all ML models, the labels for four class classification are included in these datasets, binary labels are taken from the binary dataset (valence) and binaryarous dataset (arousal). The files with NS designation were not used since they were the not standardised datasets. 

#### Results: 
had results for initial classification however results in the report come from the emotiondetection.ipynb (Deep learning) and from results.py (SVM) Both files will need changing according to the dataset that wants to be tested. all datasets are in the processed data folder. Leave one out method is in the emotiondetection notebook at the bottom.

#### tfModel: 
was the saved model from deep learning testing and not used in report

#### venv:
Mac M1 tensorflow environment used when using macs with M1 soc 

