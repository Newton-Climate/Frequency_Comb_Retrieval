import numpy as np
import matplotlib as plt
import netCDF4 # to read netCDF files
import h5py
from scipy.ndimage.filters import gaussian_filter1d # Convolution for instrument lineshape
from scipy.interpolate import interp1d # interpolation for regridding
from hapi import * # import HITRAN database functions


filename = 'testdata_2.h5'
file = h5py.File(filename)

temperature_dataset = 'Temperature_K'; # K
temperature = file.get(temperature_dataset)
temperature = np.array(temperature)

pressure_dataset = 'Pressure_mbar'; # MBa
pressure = file.get(pressure_dataset)
pressure = np.array(pressure)


pathlength_dataset = 'path_m'; # m
pathlength = file.get(pathlength_dataset)
pathlength = np.array(pathlength)

frequency_dataset = 'Freq_Hz'
frequency = file.get(frequency_dataset)
frequency = np.array(frequency)
c = 299792458;
frequency = c/frequency*1e9;



FC_dataset = 'DCSdata_Hz'
FC = file.get(FC_dataset)
FC = np.array(FC)



"""
# get the CH4 cross-sections
fetch('CH4_S',6,1,xmin,xmax)
nu_CH4,sw_CH4 = getColumns('CH4_S',['nu','sw'])
nu_, cs_ch4 = absorptionCoefficient_Voigt(SourceTables='CH4_S', WavenumberRange=[xmin,xmax],Environment={'p':p_,'T':T_},IntensityThreshold=1e-27)
"""
