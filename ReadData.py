from hapi import * # import HITRAN database functions
import numpy as np
import h5py
import multiprocessing as mp
import pdb




import numpy as np

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
    
    
    fetch('CH4_S' ,6 ,1 ,min_wavenumber ,max_wavenumber) # 12CH4
#    fetch('CH4_S' ,6 ,2 ,min_wavenumber ,max_wavenumber) # 13CH4
    fetch('CO2_S',2,1,min_wavenumber, max_wavenumber) # 12CO2
#    fetch('CO2_S',2,2,min_wavenumber, max_wavenumber) # 12CO2
    fetch('H2O_S',1,1,min_wavenumber,max_wavenumber) # H2O
#    fetch('H2O_S',1,4,min_wavenumber,max_wavenumber) # H2O
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
        self.pressure = GetDataField('Pressure_mbar')
        self.pathlength = GetDataField('path_m') * 100.0 # convert from m to cm
        self.time = GetDataField('LocalTime')
        frequency = GetDataField('Freq_Hz')
        c = 299792458 * 100 # speed of light
#        frequency = np.flip(frequency) # need to reverse the order to low -> high
#        self.frequency = c/frequency*1e9; # convert to nanometers
        self.wavenumber_grid = frequency / c # convert to wave nubmers from hz
        self.min_wavenumber = self.wavenumber_grid[0]
        self.max_wavenumber = self.wavenumber_grid[-1]
        self.FC = GetDataField('DCSdata_Hz')
        self.num_measurements = self.FC.shape[1]
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
        self.spectral_grid = dataset_object.wavenumber_grid
        self.time = dataset_object.time[ measurement_number ]

        # subset Frequency Comb data to user-defined spectral range
        window_indexes = np.array( np.where( (self.spectral_grid > self.min_wavenumber) & (self.spectral_grid < self.max_wavenumber) ))
        # Specify index of min and max wavenumber range in array
        left_index ,right_index = window_indexes[ 0, 0] ,window_indexes[ 0, -1]

        self.FC = dataset_object.FC[ left_index : right_index ,measurement_number ] # subset FC to user-defined wavenumber range
        self.spectral_grid = self.spectral_grid[left_index : right_index]

        
        self.temperature = dataset_object.temperature[ measurement_number ] # degrees K
        self.spectral_resolution = self.spectral_grid[1] - self.spectral_grid[0]
        self.pathlength = dataset_object.pathlength
        return self
    # end of method GetMeasurement

    def GetVCD(self ,VMR_H2O = None ,specific_humidity = None):
        '''
Calculates VCD from self and FrequencyComb_object
'''
        
        Rd = 287.04 # specific gas constant for dry air
        R_universal = 8.314472; # joules / moles / K
        Na = 6.0221415e23; # molecules / moles
        dz = self.pathlength
        pressure = self.pressure
        temperature = self.temperature



        if VMR_H2O == None:
            VMR_H2O = 0.0

        #rho_n =  pressure *(1- VMR_H2O ) * 100. /(R_universal*temperature)*Na/10000.0
        rho_n = pressure *(1- VMR_H2O ) /(R_universal*temperature)*Na/1.0e4
        self.rho_n = rho_n
        self.VCD = rho_n * dz
        return self
    # end of method GetVCD




    def __init__(self, measurement_number ,min_wavenumber ,max_wavenumber ,dataset_object, legendre_polynomial_degree = 20):
        self.max_wavenumber = max_wavenumber
        self.min_wavenumber = min_wavenumber
        self.GetMeasurement( measurement_number ,dataset_object)
        self.GetVCD()
        self.legendre_polynomial_degree = legendre_polynomial_degree
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
        pressure_ = self.pressure / 1013.25 # convert to atmospheres 
        min_wavenumber =   self.min_wavenumber # convert to cm^-1
        max_wavenumber = self.max_wavenumber # convert to bbcm^-1
        wavenumber_resolution = self.spectral_resolution


        self.grid, self.CH4 = absorptionCoefficient_Voigt(SourceTables='CH4_S', WavenumberRange=[ min_wavenumber ,max_wavenumber ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-30)
        nu_, self.CO2 = absorptionCoefficient_Voigt(SourceTables='CO2_S', WavenumberRange=[ min_wavenumber ,max_wavenumber ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-30)
        nu_, self.H2O = absorptionCoefficient_Voigt(SourceTables='H2O_S', WavenumberRange=[ min_wavenumber ,max_wavenumber ] ,WavenumberStep = wavenumber_resolution ,Environment={'p':pressure_ ,'T':temperature_},IntensityThreshold=1e-30)
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

