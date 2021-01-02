# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:59:19 2020

@author: naufa
"""


import pandas as pd
import numpy as np

raw_data = pd.read_csv("B737_train.csv", delimiter = ",", low_memory=False)
processed_data = pd.DataFrame()
processed_data["timestep"] = raw_data.timestep
processed_data["time"] = raw_data.time
processed_data["callsign"] = raw_data.callsign
processed_data["latitude"] = raw_data.lat
processed_data["longitude"] = raw_data.lon
processed_data["baroaltitude"] = raw_data.baroaltitude
processed_data["velocity"] = raw_data.velocity
processed_data["heading"] = raw_data.heading

timesteps = processed_data["timestep"]

dataset = []
temp = []

for idx in range(len(timesteps) - 1):
    if timesteps.get(idx) == 0:
        temp = []
        temp.append([processed_data.iloc[idx].heading,processed_data.iloc[idx].velocity, processed_data.iloc[idx].baroaltitude]) 
    elif timesteps.get(idx + 1) == 0:
        
        temp = np.array(temp)
        x1 = ((temp[:,0] - temp[:,1])**2)
        x2 = ((temp[:,0] - temp[:,2])**2)
        x3 = ((temp[:,1] - temp[:,2])**2)
        
        r = x1 + x2 + x3
        dist = np.sqrt(r)
        
        dataset.append(dist)
    else:
        temp.append([processed_data.iloc[idx].heading,processed_data.iloc[idx].velocity, processed_data.iloc[idx].baroaltitude])

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import multiprocessing

scaler = MinMaxScaler()
linear_regression = LinearRegression()


def plot(args):
    data, trend_line, name = args
    fig = plt.figure()
    plt.plot(data)
    plt.plot(trend_line, "r--", color = 'red')
    plt.title(name)
    fig.savefig("plot train/"+name+".jpg")

def CariSelisih(data, idx):
    name = "Flight "
    data = scaler.fit_transform(data.reshape(-1,1))
    x = np.arange(0, len(data))
    y = data
    new_x= np.linspace(0,len(x),84)
    new_y = np.interp(new_x, x, y.reshape(-1))
    new_x = np.arange(len(new_y))
    new_x = new_x.reshape(-1,1)
    

    model = linear_regression.fit(new_x, new_y)
    trend_line = model.predict(new_x)
    r2 = r2_score(new_y, trend_line)
    print(r2)
    
    label = 0
    str_label = "Normal"
    if r2 < 0.9858520305549033:
        label = 1
        str_label = "Abnormal"
    
    name = name + str(idx) + "-" + str_label
    plt.plot(new_y)
    plt.plot(trend_line, "r--", color = 'red')
    plt.title(name)
    plt.savefig("plot train/"+name+".jpg")
    plt.close()

    new_y = np.append(new_y, label)
    return new_y
    
from joblib import Parallel, delayed
import os

n_cores = os.cpu_count()

parallel = Parallel(n_jobs=n_cores)
results = parallel(delayed(CariSelisih)(dataset[d], d) for d in range(len(dataset)))

np.savetxt("dataset.csv", results, delimiter = ",")

