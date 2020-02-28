from hapi import * # import HITRAN database functions
import numpy as np
import h5py

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
    fetch('CO2_S',6,1,min_wavelength, max_wavelength)
    fetch('H2O_S',6,1,min_wavelength,max_wavelength)
    return 0


class CombData:
    def ReadDatafile( self ,filename):

        file = h5py.File(filename)

        def GetDataField(field_name):
            data = file.get(field_name)
            data = np.array(data)
            return data

        self.temperature = GetDataField('Temperature_K')
        self.pressure = GetDataField('Pressure_mbar')
        self.pathlength = GetDataField('path_m')
        frequency = GetDataField('Freq_Hz')
        c = 299792458;
        frequency = np.flip(frequency) # need to reverse the order to low -> high
        self.frequency = c/frequency*1e9;
        FC = GetDataField('DCSdata_Hz')
        self.FC = np.flip(FC, axis = 1)
        return self

    def __init__(self, filename):
        self.filename = filename
        self.ReadDatafile(filename)

        

class Measurement(CombData):

    def GetMeasurement(self  ,measurement_number ,dataset_object ):
        self.pressure = dataset_object.pressure[measurement_number]
        self.FC = dataset_object.FC[ measurement_number ]
        self.temperature = dataset_object.temperature[ measurement_number ]
        self.spectral_grid = dataset_object.frequency
        self.min_wavelength = self.spectral_grid[0]
        self.max_wavelength = self.spectral_grid[-1]
        self.pathlength = dataset_object.pathlength
        return self

    def GetVCD(self):
        Rd = 287.04 # specific gas constant for dry air
        R_universal = 8.314472;
        Na = 6.0221415e23;
        dz = self.pathlength
        temperature = self.temperature
        pressure = self.pressure

        rho_n =  pressure/(R_universal*temperature)*Na/10000.0
        vcd = rho_n * dz
        return vcd


    def __init__(self, measurement_number, dataset_object):
        self.GetMeasurement( measurement_number ,dataset_object)
        self.VCD = self.GetVCD()

class HitranSpectra:
    
    def ComputeCrossSections(self):
        temperature_ = self.temperature
        pressure_ = self.pressure
        min_wavelength = self.min_wavelength
        max_wavelength = self.max_wavelength
        self.grid, self.CH4 = absorptionCoefficient_Voigt(SourceTables='CH4_S', WavenumberRange=[ min_wavelength ,max_wavelength ] ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        nu_, self.CO2 = absorptionCoefficient_Voigt(SourceTables='CO2_S', WavenumberRange=[ min_wavelength ,max_wavelength ] ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        nu_, self.H2O = absorptionCoefficient_Voigt(SourceTables='H2O_S', WavenumberRange=[ min_wavelength ,max_wavelength ] ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)


    def __init__(self ,dataset_object):

        self.max_wavelength = dataset_object.max_wavelength + 1
        self.min_wavelength = dataset_object.min_wavelength - 1
        self.pressure = dataset_object.pressure
        self.temperature = dataset_object.temperature
        self.ComputeCrossSections()
        



file = 'testdata_2.h5'
data = CombData(file)
current_data = Measurement( 100 , data)
GetCrossSections( current_data.min_wavelength ,current_data.max_wavelength)
spectra = HitranSpectra(current_data)

