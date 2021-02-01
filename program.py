import threading
from multiprocessing import pool
import multiprocessing
import time
import logging
import pickle
import os
import pandas as pd
from os import listdir

data = []
# Getting all the files in the data directory
files = os.listdir("./raw_data/ecg-gsr-labels/")
removed = ['101_PreProcessed_GSR_ECG.dat', '102_PreProcessed_GSR_ECG.dat', '103_PreProcessed_GSR_ECG.dat', '115_PreProcessed_GSR_ECG.dat', '118_PreProcessed_GSR_ECG.dat', '121_PreProcessed_GSR_ECG.dat', '119_PreProcessed_GSR_ECG.dat', '130_PreProcessed_GSR_ECG.dat']

for item in removed:
    files.remove(item)

for i in range(len(files)):
    infile = open("./raw_data/ecg-gsr-labels/" + files[i],'rb')
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
    #Iterate over all examples in file
    for x in range(len(labels)):
        completeLabels.append(labels[x])
        completeEcg.append(features[x][:][:,1])
        completeGsr.append(features[x][:][:,0])


labelslen = len(completeLabels)
ecglen = len(completeEcg)
gsrlen = len(completeGsr)
print(f"Completed:{labelslen} lables, {ecglen} ECG inputs, {gsrlen} GSR inputs")


cores = 24 #define amount of cores you have
threads = []
startindex = 0
endindex = len(completeLabels)//cores #depending on dp might have to floor
incindex = endindex

def threadrunner(listitems):
    for item in range(listitems):
        print("started: " +item)
        dlDataset = pd.DataFrame()
        label = completeLabels[[item]]
        ecg = completeEcg[item]
        gsr = completeGsr[item]
        for point in range(len(ecg)):
            ecgPoint = ecg[point]
            gsrPoint = gsr[point]
            d = {"user": item, "label": label, "timestamp": point, "ecg": ecgPoint,"gsr":gsrPoint}
            d = pd.DataFrame(d, index=[item])
            dlDataset = pd.concat([dlDataset,d])
        print("Completed: " +item)
        dlDataset.to_csv(f"processed_data/dlData/{item}.csv", index=False)

for core in range(cores):
    threads.append(list(range(startindex,endindex)))
    startindex = endindex
    endindex = endindex + incindex


with multiprocessing.Pool(processes=cores) as pool:
    result = pool.map_async(threadrunner, threads)
