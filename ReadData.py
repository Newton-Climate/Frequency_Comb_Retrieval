from hapi import * # import HITRAN database functions
import numpy as np
import h5py
from scipy.interpolate import interp1d # interpolation for regridding

# Next tep is to compute the dry air mole fraction from the pressure (probably using hypsometric equation)


def GetCrossSections( min_wavelength ,max_wavelength):
    """
Download the cross-sections for CH4, H2O, and CO2 by accessing the HITRAN database. 
Note that hapy.py should be in this directory.

inputs:
min_wavelength (double): the minimum part of the wavelength grid in nm
max_wavelength (double): the maximum wavelength in the spectral grid

outputs:
N/A
"""
    
    
    fetch('CH4_S' ,6 ,1 ,min_wavelength ,max_wavelength)
    fetch('CO2_S',2,1,min_wavelength, max_wavelength)
    fetch('H2O_S',1,1,min_wavelength,max_wavelength)
    return 0


class CombData:
    def ReadDatafile( self ,filename):

        file = h5py.File(filename)

        def GetDataField(field_name):
            data = file.get(field_name)
            data = np.array(data)
            return data

        self.temperature = GetDataField('Temperature_K')
        self.pressure = GetDataField('Pressure_mbar') / 1013 # convert from mbar to atm
        self.pathlength = GetDataField('path_m') * 100.0 # convert from m to cm        
        frequency = GetDataField('Freq_Hz')
        c = 299792458 * 100 # speed of light
#        frequency = np.flip(frequency) # need to reverse the order to low -> high
#        self.frequency = c/frequency*1e9; # convert to nanometers
        self.wavenumber_grid = frequency / c # convert to wave nubmers from hz
        self.FC = GetDataField('DCSdata_Hz')
#        self.FC = np.flip(self.FC, axis = 0)
        return self

    def __init__(self, filename):
        self.filename = filename
        self.ReadDatafile(filename)

        

class Measurement(CombData):

    def GetMeasurement(self  ,measurement_number ,dataset_object ):
        self.pressure = dataset_object.pressure[measurement_number]
        self.FC = dataset_object.FC[ : ,measurement_number ]
        self.temperature = dataset_object.temperature[ measurement_number ] # degrees K
        self.spectral_grid = dataset_object.wavenumber_grid
        self.min_wavelength = self.spectral_grid[0]
        self.max_wavelength = self.spectral_grid[-1]
        self.spectral_resolution = self.spectral_grid[1] - self.spectral_grid[0]
        self.pathlength = dataset_object.pathlength
        return self

    def GetVCD(self ,VMR_H2O = None ,specific_humidity = None):
        Rd = 287.04 # specific gas constant for dry air
        R_universal = 8.314472;
        Na = 6.0221415e23;
        dz = self.pathlength
        temperature = self.temperature
        pressure = self.pressure
        HumidityToVMR = 1.6068


        if VMR_H2O == None:
            print('No a priori humidity value was given. Assuming 0 percent humidity by default \n')
            VMR_H2O = 0.0

        try:
            VMR_H2O = specific_humidity * HumidityToVMR # convert from specific humidity to volume mixing ratio
            print('Specific Humidity was given. Converting to Volume Mixing Ratio \n')
            
        except:
            print('Water vapor Volume Mixing Ratio was given. Computing dry air mole fraction \n')
            
#        rho_n =  pressure *(1- VMR_H2O ) * 100. /(R_universal*temperature)*Na/10000.0
        rho_n =  pressure *(1- VMR_H2O ) * 100. /(R_universal*temperature)*Na/10000.0
        self.VCD = rho_n * dz
        return self


    def __init__(self, measurement_number, dataset_object):
        self.GetMeasurement( measurement_number ,dataset_object)
        self.GetVCD()

class HitranSpectra:
    
    def ComputeCrossSections(self):
        temperature_ = self.temperature
        pressure_ = self.pressure
        min_wavelength =   self.min_wavelength # convert to cm^-1
        max_wavelength = self.max_wavelength # convert to bbcm^-1
        wavenumber_resolution = self.spectral_resolution


        self.grid, self.CH4 = absorptionCoefficient_Voigt(SourceTables='CH4_S', WavenumberRange=[ min_wavelength ,max_wavelength ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        nu_, self.CO2 = absorptionCoefficient_Voigt(SourceTables='CO2_S', WavenumberRange=[ min_wavelength ,max_wavelength ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        nu_, self.H2O = absorptionCoefficient_Voigt(SourceTables='H2O_S', WavenumberRange=[ min_wavelength ,max_wavelength ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        return self

    def GetSolarSpectrum(self):
        sun = np.loadtxt('solar_merged_20160127_600_26316_000.out')
        c = 299792458 * 100 # speed of light
        solar_grid = 1e7 / sun[:,0]
        f_solar = interp1d(solar_grid, sun[:,1])

        solar_transmission = f_solar(self.grid) # will need to modify this from nm to cm^-1
        return solar_transmission


    def __init__(self ,dataset_object):

        self.max_wavelength = dataset_object.max_wavelength + 1
        self.min_wavelength = dataset_object.min_wavelength - 1
        self.spectral_resolution = dataset_object.spectral_resolution
        self.pressure = dataset_object.pressure
        self.temperature = dataset_object.temperature
        self.ComputeCrossSections()
        self.solar_spectrum = self.GetSolarSpectrum()
#        self.grid = np.flip(self.grid)
        




