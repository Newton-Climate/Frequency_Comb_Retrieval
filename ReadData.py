from hapi import * # import HITRAN database functions
import numpy as np
import h5py
from scipy.interpolate import interp1d # interpolation for regridding

def GetCrossSections( min_wavenumber ,max_wavenumber):

    """
Download the cross-sections for CH4, H2O, and CO2 by accessing the HITRAN database. 
Note that hapy.py should be in this directory.

inputs:
1. min_wavenumber: Float64 number with minimum wavenumber in range (cm^-1)
max_wavenumber: Float64 number denoting max wavenumber in range (cm^-1)
outputs:
Downloaded cross-section files
"""
    
    
    fetch('CH4_S' ,6 ,1 ,min_wavenumber ,max_wavenumber)
    fetch('CO2_S',2,1,min_wavenumber, max_wavenumber)
    fetch('H2O_S',1,1,min_wavenumber,max_wavenumber)
    return 0
# end of function GetCrossSections


class CombData:

    '''
class that contains the fields from frequency comb datafile
initialize as:
object = CombData( 'filename.h5')
'''
    
    def ReadDatafile( self ,filename):

        file = h5py.File(filename)

        # define a function to read each field in file
        def GetDataField(field_name):
            data = file.get(field_name)
            data = np.array(data)
            return data
        # end of function ReadDataField

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
    # end of method ReadDataFile


    def __init__(self, filename):
        self.filename = filename
        self.ReadDatafile(filename)
    # end of function FreqComb.__init__
# end of class CombData

        

class Measurement(CombData):
    '''
A Class the contains the fields for inversion the ith measurement

initialize as:
object = Measurement( mesurement_number ,FrequencyComb_object)

inputs:
1. measurement_object: Int that corresponds to the specfic measurement being requested
2. FrequencyComb_object: the object that contains the entire frequency comb data
'''

    def GetMeasurement(self  ,measurement_number ,dataset_object ):
        '''
Reads data from FrequencyComb_object and assigns fields 
'''
        
        self.pressure = dataset_object.pressure[measurement_number]
        self.FC = dataset_object.FC[ : ,measurement_number ]
        self.temperature = dataset_object.temperature[ measurement_number ] # degrees K
        self.spectral_grid = dataset_object.wavenumber_grid
        self.min_wavenumber = self.spectral_grid[0]
        self.max_wavenumber = self.spectral_grid[-1]
        self.spectral_resolution = self.spectral_grid[1] - self.spectral_grid[0]
        self.pathlength = dataset_object.pathlength
        return self
    # end of method GetMeasurement

    def GetVCD(self ,VMR_H2O = None ,specific_humidity = None):
        '''
Calculates VCD from self and FrequencyComb_object
'''
        
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
    # end of method GetVCD


    def __init__(self, measurement_number, dataset_object):
        self.GetMeasurement( measurement_number ,dataset_object)
        self.GetVCD()
    # end of method __init__
# end of class Measurement 
        

class HitranSpectra:
    '''
Accesses downloaded header and cross-section fileds from HITRAN and calculates spectra/cross-sections
initialize as:
hitran_object = HitranSpectra(measurement_object)
Inputs:
1. measurement_object: the Measurement class that contains the ith measurement object
this was initialized as measurement_object = Measurement( i, CombData_object)

'''
    
    
    def ComputeCrossSections(self):
        '''
computs the cross-sections with broadening and line-mixing
'''
        
        temperature_ = self.temperature
        pressure_ = self.pressure
        min_wavenumber =   self.min_wavenumber # convert to cm^-1
        max_wavenumber = self.max_wavenumber # convert to bbcm^-1
        wavenumber_resolution = self.spectral_resolution


        self.grid, self.CH4 = absorptionCoefficient_Voigt(SourceTables='CH4_S', WavenumberRange=[ min_wavenumber ,max_wavenumber ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        nu_, self.CO2 = absorptionCoefficient_Voigt(SourceTables='CO2_S', WavenumberRange=[ min_wavenumber ,max_wavenumber ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        nu_, self.H2O = absorptionCoefficient_Voigt(SourceTables='H2O_S', WavenumberRange=[ min_wavenumber ,max_wavenumber ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-27)
        return self
    # end of method ComputeCrossSections

    def GetSolarSpectrum(self):
        '''
initializes and saves solar spectra object and corresponding spectral grid
'''
        
        sun = np.loadtxt('solar_merged_20160127_600_26316_000.out')
        c = 299792458 * 100 # speed of light
        solar_grid = 1e7 / sun[:,0]
        f_solar = interp1d(solar_grid, sun[:,1])

        solar_transmission = f_solar(self.grid) # will need to modify this from nm to cm^-1
        return solar_transmission
    # end of method GetSolarSpectrum


    def __init__(self ,dataset_object):

        self.max_wavenumber = dataset_object.max_wavenumber + 1
        self.min_wavenumber = dataset_object.min_wavenumber - 1
        self.spectral_resolution = dataset_object.spectral_resolution
        self.pressure = dataset_object.pressure
        self.temperature = dataset_object.temperature
        self.ComputeCrossSections()
#        self.solar_spectrum = self.GetSolarSpectrum()
    # end of method __init__
# end of class HitranSpectra


        




