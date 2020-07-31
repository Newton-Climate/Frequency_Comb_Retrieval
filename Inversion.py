from numpy.linalg import inv, norm
from numpy import dot
import numpy as np
from ForwardModel import *
from ReadData import *


num_species = 4
_H2O_index ,_CH4_index ,_CO2_index ,_HDO_index = [i for i in range(num_species)]
_temperature_index = num_species
_windspeed_index = num_species + 1

def CalculateChi2( modelled_measurement ,measurement ,state_vector):
    degrees_of_freedom = measurement.size - state_vector.size
    diff =  modelled_measurement - measurement
    error = norm(diff)
    chi2 = error / np.var(diff) /  degrees_of_freedom
    return chi2



def LinearInversion( state_vector ,measurement_object ,hitran_object, FM=False):

    k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=True)
    ans = inv(k.T.dot(k)).dot(k.T).dot( np.log(measurement_object.FC) )

   
    modelled_measurement = k.dot(ans)

    if FM == False:
        return ans
    else:
        return ans, modelled_measurement, measurement_object.FC, measurement_object.spectral_grid
    
    
    

def TestLinearInversion( test_state_vector ,true_state_vector ,measurement_object ,hitran_object):
    
    f_true ,transmission ,evaluated_polynomial = ForwardModel(true_state_vector ,measurement_object, hitran_object)
    k = ComputeJacobian( test_state_vector ,measurement_object ,hitran_object ,linear=True)

    ans = inv(k.T.dot(k)).dot(k.T).dot(np.log(f_true))
#    f_test ,transmission ,evaluated_polynomial = ForwardModel(ans ,current_data ,spectra)
    return ArrayToDict(ans) 


def NonlinearInversion( initial_guess ,measurement_object ,hitran_object, flags = None, FM = False):

    measurement = np.log( measurement_object.FC )
    state_vector = initial_guess.copy()

    # initialize variables for while-loop
    tolerence = 0.0001 # stopping criterium 
    rel_diff = 1 # relative difference for stopping criterium
    i = 0 # iteration number
    current_modelled_measurement = ForwardModel( state_vector ,measurement_object ,hitran_object, flags = flags)
    

    while i < 10 and rel_diff > tolerence:

        

        k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,flags = flags ,linear=False)
        

        # Evaluate Gauss-Newton Algorithm 
        state_vector = state_vector + inv(k.T.dot(k)).dot(k.T).dot( measurement - current_modelled_measurement )
  
        # reassign iteration variables
        old_modelled_measurement = current_modelled_measurement 
        current_modelled_measurement = ForwardModel( state_vector ,measurement_object ,hitran_object, flags= flags)
        i += 1
        print('relative difference for iteration ', i ,' is ', rel_diff)
        rel_diff = np.abs((norm( current_modelled_measurement - measurement) - norm(old_modelled_measurement - measurement)) / norm(old_modelled_measurement - measurement))

        
    if FM == False:
        return state_vector
    else:
        return state_vector ,current_modelled_measurement ,measurement, measurement_object.spectral_grid 

def FitSpectra( state_vector ,measurement_object ,hitran_object ,flags = None, linear = True, FM = False):
    if linear:
        return LinearInversion( state_vector ,measurement_object, hitran_object, FM = FM)
    else:
        return NonlinearInversion( state_vector ,measurement_object ,hitran_object, flags = flags ,FM=FM)
    
class TestRun:
    def ForwardModelTest( self, measurement_object, hitran_object, test_spectra = False ):
        f_true = ForwardModel(self.truth, measurement_object, hitran_object, flags = self.control_flags)
        measurement_object.FC = np.exp( f_true )

        if test_spectra:
            print("Testing Hit08")
            hitran_object = self.ch4_test_spectra
            
        result = FitSpectra( self.guess, measurement_object, hitran_object, flags = self.test_flags, linear = False)
        f_test = ForwardModel( result, measurement_object, hitran_object, flags = self.test_flags)
        return result, f_true, f_test, measurement_object.spectral_grid
    
        
        
    def __init__(self, guess, truth, dataset_object, control_flags, test_flags):
        self.guess, self.truth = guess, truth
        self.control_flags, self.test_flags = control_flags, test_flags
        ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
        co2_min_wavenumber, co2_max_wavenumber = 6180, 6250
        hdo_min_wavenumber, hdo_max_wavenumber = 6310, 6380
        measurement_index = 5 # arbitrary index as will be overwritten 

        # instnatiate the Measurement objects
        ch4_data = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,dataset_object)
        co2_data = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,dataset_object)
        hdo_data = Measurement(measurement_index, hdo_min_wavenumber, hdo_max_wavenumber, dataset_object)

        # instantiate the hitran objects
        co2_spectra = HitranSpectra( co2_data)
        ch4_spectra = HitranSpectra(ch4_data, use_hitran08 = control_flags['use_hitran08'])
        hdo_spectra = HitranSpectra(hdo_data)
        if control_flags['use_hitran08'] != test_flags['use_hitran08']:
            self.ch4_test_spectra = HitranSpectra(ch4_data, use_hitran08 = test_flags['use_hitran08'])

        # commence the test-runs
        self.co2_result, self.co2_true, self.co2_modelled, self.co2_grid = self.ForwardModelTest(co2_data, co2_spectra)
        self.ch4_result, self.ch4_true, self.ch4_modelled, self.ch4_grid = self.ForwardModelTest(ch4_data, ch4_spectra, test_spectra = True)
        self.hdo_result, self.hdo_true, self.hdo_modelled, self.hdo_grid = self.ForwardModelTest(hdo_data, hdo_spectra)

        # assign to output vector
        concentrations = np.zeros(num_species)
        concentrations[_H2O_index] = self.co2_result[_H2O_index]
        concentrations[_CO2_index] = self.co2_result[ _CO2_index]
        concentrations[ _CH4_index] = self.ch4_result[ _CH4_index]
        concentrations[_HDO_index] = self.hdo_result[ _HDO_index]
        self.concentrations = concentrations
        return None



        


def InvertAllData( initial_guess ,dataset_object, hitran_object):
    num_measurements = dataset_object.num_measurements
    ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
    co2_min_wavenumber, co2_max_wavenumber = 6180, 6250
    hdo_min_wavenumber ,hdo_max_wavenumber = 6300 , 6370 



    H2O_con = np.empty(( num_measurements ))
    CH4_con = np.empty(( num_measurements ))
    CO2_con = np.empty(( num_measurements ))


    for measurement_index in range(num_measurements):
        #print(measurement_index)


        ch4_measurement_object = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,dataset_object)
        co2_measurement_object = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,dataset_object)
        ch4_spectra_object = HitranSpectra( ch4_measurement_object)
        co2_spectra_object = HitranSpectra( co2_measurement_object )

        ch4_state_vector  = FitSpectra( initial_guess ,ch4_measurement_object ,ch4_spectra_object)
        co2_state_vector  = FitSpectra( initial_guess ,co2_measurement_object ,co2_spectra_object)


        H2O_con[ measurement_index ] = co2_state_vector[ 'VMR_H2O' ]
        CH4_con[ measurement_index ] = ch4_state_vector[ 'VMR_CH4' ]
        CO2_con[ measurement_index ] = co2_state_vector[ 'VMR_CO2' ]

    # end of for loop
    return H2O_con, CH4_con ,CO2_con 
# end of function InvertAllData
 
def MakeInversionFunction( dataset_object, initial_guess ,flags = None ,linear = True ,FM = True):
    num_measurements = dataset_object.num_measurements
    ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
    co2_min_wavenumber, co2_max_wavenumber = 6180, 6250
    hdo_min_wavenumber, hdo_max_wavenumber = 6310, 6370
    global Invert
    

    def Invert( measurement_index):
#        print(measurement_index)
        output_vector = np.empty( num_species + 3)

        
        #Get the measurements and sub-set to corresponding wavenumber regions        
        co2_measurement_object = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,dataset_object)

        
        co2_spectra_object = HitranSpectra( co2_measurement_object )
        co2_state_vector  = FitSpectra( initial_guess ,co2_measurement_object ,co2_spectra_object , flags = None, linear = False, FM = FM)
        dataset_object.temperature[measurement_index] = co2_state_vector[0][_temperature_index]
        hdo_measurement_object = Measurement( measurement_index ,hdo_min_wavenumber ,hdo_max_wavenumber ,dataset_object)
        ch4_measurement_object = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,dataset_object)

        #Calculate the line intensities and shapes
        ch4_spectra_object = HitranSpectra( ch4_measurement_object, use_hitran08 = flags['use_hitran08'])        
        hdo_spectra_object = HitranSpectra( hdo_measurement_object )

        # Perform linear inversions
        ch4_state_vector = FitSpectra( initial_guess ,ch4_measurement_object ,ch4_spectra_object, flags = flags, linear = linear, FM = FM)
        hdo_state_vector  = FitSpectra( initial_guess ,hdo_measurement_object ,hdo_spectra_object ,flags = flags ,linear = linear,FM = FM)

        #output_vector[ _H2O_index ] = co2_state_vector[ _H2O_index ]
        #output_vector[ _CH4_index ] = ch4_state_vector[ _CH4_index ]
        #output_vector[ _CO2_index ] = co2_state_vector[ _CO2_index ]
        #output_vector[ _HDO_index ] = hdo_state_vector[ _HDO_index ]
        #output_vector[ num_species ] = ch4_measurement_object.VCD # put that in the output just in case
        #output_vector[ num_species + 1] = ch4_measurement_object.time
        return ch4_state_vector, co2_state_vector, hdo_state_vector 
    return Invert

def InvertParallel( inversion_function, num_measurements, num_threads = 4):
    pool = mp.Pool(num_threads)
    result = pool.map( inversion_function ,[i for i in range(num_measurements )])

    # order results by time stamp
    try:
        result = sorted(result ,key = lambda x: x[-1])
        print('Done sorting output')
        return np.array(result)
    except:
        return result

