# CO620 Emotion Detection Python
 Python files, for emotion detection using ECG, GSR and Eye Tracking, Results are as classification reports and graphs for each model.
 
 ## Introduction
 
There are working examples of the emotion detection working with the processed data as well as my own feature extraction. emotiondetection.ipynb is the main jupyter notebook used, the mlDataTesting.ipynb is for testing. 

### Files / folders
#### Python files: 
hyperparameter, kbestfeatures, results and performance metrics are used for the final report. The file tenpointaverage was a test for extracting 10 point average for ecg and gsr features. binaryclass was used to change labels for valence and arousal classification 

#### Data: 
raw data is all in the raw data folder, processed data was all the data with extracted features used in the report. The files with NS designation were not used. 

#### Results: 
had results for initial classification however results in the report come from the emotiondetection.ipynb and from results.py

#### tfModel: 
was the saved model from deep learning testing

#### venv:
Mac M1 tensorflow environment used when using macs with M1 soc 

