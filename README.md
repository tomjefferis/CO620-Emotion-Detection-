# CO620_Python
 Python files, for emotion detection using ECG, GSR and Eye Tracking, Results are as classification reports and graphs for each model.
 
 ## Introduction
 
 There are working examples of the emotion detection working with the processed data as well as my own feature extraction. Deep learning at the end of the notebook however the data isnt enough to train a network like this, and the more simple models score highly anyway. Results below are for ECG and GSR combined, ECG, GSR and eye tracking combined as well as a 10 point average for the ECG and GSR signals. 

## Results
### Non standardised ECG-GSR:
#### Perfomance metric:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/RFC%20ecg-gsr/lowres/ns/benchmark_models_performance.png)
#### Decision boundaries:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/RFC%20ecg-gsr/lowres/ns/visual_classifier_decisions.png)
### Standardised ECG-GSR:
#### Perfomance metric:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/RFC%20ecg-gsr/lowres/s/benchmark_models_performance.png)
#### Decision boundaries:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/RFC%20ecg-gsr/lowres/s/visual_classifier_decisions.png)
### Standardised ECG-GSR-Eye-Tracking:
#### Perfomance metric:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/Shallow%20classifiers/lowres/benchmark_models_performance.png)
#### Decision boundaries:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/Shallow%20classifiers/lowres/visual_classifier_decisions.png)
### 10 Point Average For ECG and GSR:
#### Perfomance metric:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/10%20point%20average/lowres/1/benchmark_models_performance.png)
#### Decision boundaries:
![Non standardised ECG-GSR](https://github.com/tomjefferis/CO620-Emotion-Detection-/blob/main/results/10%20point%20average/lowres/1/visual_classifier_decisions.png)
