# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:38:29 2023

@author: mikf
"""

import os 
import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
model_path = 'models/QLS'
model_indicator = '_QLS_surrogate_model.pickle'

files = []
IPs = []
for file in os.listdir(model_path):
    if model_indicator in file:
        IP = float(file.split(model_indicator)[0])
        files.append(file)
        IPs.append(IP)

def get_r2(data, prediction):
    residuals = data - prediction
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data-np.mean(data))**2) 
    r2_power = 1 - (ss_res / ss_tot)
    return r2_power

IP_item = 0
IP = IPs[IP_item]
path = os.path.join(model_path, files[IP_item])
with open(path, 'rb') as f:
    dic = pickle.load(f)

input_channel_names = dic['input_channel_names']
output_channel_names = dic['output_channel_names']
out_item = 1
output_channel = output_channel_names[out_item]
df = dic['df']

# # plot df
plt.figure()
r2 = get_r2(df[output_channel], df[output_channel+'_fit'])
plt.plot(df[output_channel], df[output_channel+'_fit'], '.')
plt.title(f'{output_channel} simulation vs QLS IP={IP} r2={r2:.3f}')

# predict 
class QLSModel(object):
    def __init__(self, model, input_scaler, output_scaler):
        self.model, self.input_scaler, self.output_scaler = model.getMetaModel(), input_scaler, output_scaler
        
    def predict(self, RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed):
        inps = np.asarray([RP, D, HTrans, HHub_Ratio, WaterDepth, WaveHeight, WavePeriod, WindSpeed])
        inps_scaled = self.input_scaler.transform(inps.reshape(1, -1))
        scaled_output = self.model(inps_scaled)
        output = self.output_scaler.inverse_transform(scaled_output).ravel()
        return output

qlsm = QLSModel(dic['models'][out_item], dic['input_scaler'], dic['output_scalers'][output_channel])
res = qlsm.predict(RP=11, D=214, HTrans=13, HHub_Ratio=0.7, WaterDepth=32, WaveHeight=3.5, WavePeriod=6, WindSpeed=9)
print(res)

