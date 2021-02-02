import pickle
import os
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk2
import statistics
import heartpy as hp
import numpy as np
import scipy as sci
import pysiology as pyd

data = []
# Getting all the files in the data directory
files = os.listdir("./raw_data/ecg-gsr-labels/")
removed = ['101_PreProcessed_GSR_ECG.dat', '102_PreProcessed_GSR_ECG.dat', '103_PreProcessed_GSR_ECG.dat',
           '115_PreProcessed_GSR_ECG.dat', '118_PreProcessed_GSR_ECG.dat', '121_PreProcessed_GSR_ECG.dat',
           '119_PreProcessed_GSR_ECG.dat', '130_PreProcessed_GSR_ECG.dat']

for item in removed:
    files.remove(item)

for i in range(len(files)):
    infile = open("./raw_data/ecg-gsr-labels/" + files[i], 'rb')
    data.append(pickle.load(infile))
    infile.close()

# Extracting all data (labels, ecg and gsr data) into seperate arrays.

completeLabels = []
completeEcg = []
completeGsr = []

# Iterate over all files
for i in range(len(data)):
    del data[i]['Data'][0]
    del data[i]['Labels'][0]
    features = data[i]['Data']
    labels = data[i]['Labels']
    # Iterate over all examples in file
    for x in range(len(labels)):
        completeLabels.append(labels[x])
        completeEcg.append(features[x][:][:, 1])
        completeGsr.append(features[x][:][:, 0])

labelslen = len(completeLabels)
ecglen = len(completeEcg)
gsrlen = len(completeGsr)
print(f"Completed:{labelslen} lables, {ecglen} ECG inputs, {gsrlen} GSR inputs")


def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])


def nkExtractECG(signal, samplerate):
    processed = pyd.electrocardiography.analyzeECG(signal, samplerate)
    # cleaned = nk2.ecg_clean(signal,sampling_rate=samplerate)
    # processed, info = nk2.ecg_process(cleaned,sampling_rate=samplerate)
    # compECG = nk2.ecg_intervalrelated(processed)
    # Append all features extracted from pysiology library
    temp = []
    try:
        pnn5020 = processed['pnn50pnn20']
    except:
        pnn5020 = np.NaN
    temp.append(processed['ibi'])
    temp.append(processed['bpm'])
    temp.append(processed['sdnn'])
    temp.append(processed['sdsd'])
    temp.append(processed['rmssd'])
    temp.append(processed['pnn50'])
    temp.append(processed['pnn20'])
    temp.append(pnn5020)
    temp.append(processed['frequencyAnalysis']['LF'])
    temp.append(processed['frequencyAnalysis']['HF'])
    temp.append(processed['frequencyAnalysis']['VLF'])
    temp.append(processed['frequencyAnalysisFiltered']['LF'])
    temp.append(processed['frequencyAnalysisFiltered']['HF'])
    temp.append(processed['frequencyAnalysisFiltered']['VLF'])

    # Append the maximumn, minimum and mean of the signal

    temp.append(max(signal))
    temp.append(min(signal))
    temp.append(statistics.mean(signal))

    columns = ["ibi","bpm","sdnn","sdsd","rmssd","pnn50","pnn20","pnn50pnn20","flf","fhf","fvlf","fflf","ffhf","ffvlf","max","min","mean"]
    x = pd.DataFrame([temp], columns=columns)


    return x



def extractECGfeatures(ecgfull, samplerate, labels):
    print("started ECG extraction")
    # Preprocess the data (filter, find peaks, etc.)
    completeFrame = pd.DataFrame([])
    tempFrame = pd.DataFrame([])
    for index, ecg in enumerate(ecgfull):  # loops through all 312 data items from the extracted items
        chunksize = (len(ecg) // 100) * 10
        splitecg = chunks(ecg, chunksize)
        if len(splitecg[
                   -1]) < 5000:  # removes the last item if its less than 5s length that is different sized causes problems
            del splitecg[-1]

        for item in splitecg:
            tems = pd.DataFrame(nkExtractECG(item, samplerate))
            tempFrame = pd.concat([tempFrame, tems], axis=0)
        tempFrame.fillna(tempFrame.mean(), inplace=True)
        tempFrame.fillna(0, inplace=True)
        #t2Frame = pd.DataFrame([])
        t2Frame = tempFrame.unstack().to_frame().T
        t2Frame.columns = t2Frame.columns.map('{0[0]}'.format)
        t2Frame = t2Frame.rename(columns=renamer())
        t2Frame["label"] = [labels[index]]
        completeFrame = pd.concat([completeFrame, t2Frame], axis=0)
        tempFrame = pd.DataFrame([])
    # completeFrame.drop(completeFrame.columns[0], axis=1,inplace=True) #removes first empty column
    print("Completed ECG extraction")
    return completeFrame


def extractGSR(signal, samplerate):
    signal = hp.filter_signal(signal, 0.1, 1000)

    processed_gsr, infos = nk2.eda_process(signal,
                                           sampling_rate=samplerate)  # processes the GSR, currently only doing one item to make sure it works properly
    # plot = nk2.eda_plot(processed_gsr[:60000], sampling_rate=1000) #plots the signal on a graph
    gsr_dict = nk2.eda_findpeaks(processed_gsr)  # finding peaks, time of the peaks and magnitude of peaks

    peaks = sci.signal.find_peaks(signal)
    valleys = sci.signal.find_peaks(signal * (-1))
    numpeakssci = len(peaks[0]) / len(signal)
    numvalleys = len(valleys[0]) / len(signal)

    peaktime = gsr_dict['SCR_Peaks']  # time of peaks
    numpeaks = len(peaktime) / len(signal)  # number of peaks
    timebetween = []
    lastPeak = 0
    for peaks in peaktime:  # going through all the peaks in the exrtacted data
        if lastPeak != 0:
            timebetween.append(peaks - lastPeak)  # finding the time between the peaks
        else:
            lastPeak = peaks

    if len(timebetween) >= 1:
        meantbpeaks = statistics.mean(timebetween)  # mean time between peaks
        mediantbpeaks = statistics.median(timebetween)  # median time between peaks
        meanheightpeaks = statistics.mean(gsr_dict['SCR_Height'])  # mean magnitide of peaks
        medianheightpeaks = statistics.median(gsr_dict['SCR_Height'])  # median magnitide of peaks
        varheightpeaks = statistics.variance(gsr_dict['SCR_Height'])  # variance magnitide of peaks
    elif len(gsr_dict['SCR_Height']) >= 1:
        meantbpeaks = np.NaN  # mean time between peaks
        mediantbpeaks = np.NaN  # median time between peaks
        meanheightpeaks = statistics.mean(gsr_dict['SCR_Height'])  # mean magnitide of peaks
        medianheightpeaks = statistics.median(gsr_dict['SCR_Height'])  # median magnitide of peaks
        if len(gsr_dict['SCR_Height']) >= 2:
            varheightpeaks = statistics.variance(gsr_dict['SCR_Height'])  # variance magnitide of peaks
        else:
            varheightpeaks = np.NaN
    else:
        meantbpeaks = np.NaN  # mean time between peaks
        mediantbpeaks = np.NaN  # median time between peaks
        meanheightpeaks = np.NaN  # mean magnitide of peaks
        medianheightpeaks = np.NaN  # median magnitide of peaks
        varheightpeaks = np.NaN  # variance magnitide of peaks

    maxGSR = max(signal)
    minGSR = min(signal)
    meanGSR = statistics.mean(signal)
    peakratio = 0
    if numpeaks > 0: peakratio = numpeaks / len(signal)
    valleyratio = 0
    if numvalleys > 0: valleyratio = numvalleys / len(signal)

    d = {'max_GSR': maxGSR,
         'min_GSR': minGSR,
         'mean_GSR': meanGSR,
         'number_of_peaks': numpeaks,
         'number_of_peaks_SCIPY': numpeakssci,
         'number_of_valleys': numvalleys,
         'mean_time_between_peaks': meantbpeaks,
         'median_time_between_peaks': mediantbpeaks,
         'mean_height_of_peaks': meanheightpeaks,
         'median_height_of_peaks': medianheightpeaks,
         'variance_height_of_peaks': varheightpeaks,
         'ratio_of_peaks': peakratio,
         'ratio_of_valleys': valleyratio}
    # adding to dataframe to be stored as csv later
    return pd.DataFrame([d])


def extractGSRfeatures(gsrfull, samplerate, labels):
    print("started GSR extraction")
    # Preprocess the data (filter, find peaks, etc.)
    completeFrame = pd.DataFrame([])
    tempFrame = pd.DataFrame([])
    for index, gsr in enumerate(gsrfull):
        chunksize = (len(gsr) // 100) * 20
        splitgsr = chunks(gsr, chunksize)
        if len(splitgsr[
                   -1]) < 5000:  # removes the last item if its less than 20s causes problems
            del splitgsr[-1]

        label = labels[index]

        for item in splitgsr:
            temp = extractGSR(item, samplerate)
            tempFrame = pd.concat([tempFrame, temp], axis=0)
        tempFrame.fillna(tempFrame.mean(), inplace=True)
        tempFrame.fillna(0, inplace=True)
            # t2Frame = pd.DataFrame([])
        t2Frame = tempFrame.unstack().to_frame().T
        t2Frame.columns = t2Frame.columns.map('{0[0]}'.format)
        t2Frame = t2Frame.rename(columns=renamer())
        t2Frame["label"] = [label]
        completeFrame = pd.concat([completeFrame, t2Frame])
        tempFrame = pd.DataFrame([])

    # completeFrame.drop(completeFrame.columns[0], axis=1,inplace=True) #removes first empty column
    return completeFrame


def extractor(samplerate):
    ecg = extractECGfeatures(completeEcg, samplerate, completeLabels)
    gsr = extractGSRfeatures(completeGsr, samplerate, completeLabels)
    gsr.drop('label', axis=1, inplace=True)
    #gsr.drop(gsr.index[0], inplace=True)
    complete = pd.concat([ecg, gsr], axis=1)
    # complete = complete.fillna(complete.mean())
    complete.to_csv("processed_data/meganoNan.csv", index=False)


samplerate = 1000
extractor(samplerate)
