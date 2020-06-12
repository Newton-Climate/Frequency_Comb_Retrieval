from numpy.linalg import inv, norm
from numpy import dot
import numpy as np
from ForwardModel import *
from ReadData import *


num_species = 4
_H2O_index ,_CH4_index ,_CO2_index ,_HDO_index = [i for i in range(num_species)]
_temperature_index = num_species

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


def NonlinearInversion( initial_guess ,measurement_object ,hitran_object, FM = False):

    measurement = np.log( measurement_object.FC )
    state_vector = initial_guess

    # initialize variables for while-loop
    tolerence = 0.0001 # stopping criterium 
    rel_diff = 1 # relative difference for stopping criterium
    i = 0 # iteration number
    current_modelled_measurement = ForwardModel( state_vector ,measurement_object ,hitran_object)
    

    while i < 10 and rel_diff > tolerence:

        
        print('relative difference for iteration ', i ,' is ', rel_diff)
        k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=False)
        

        # Evaluate Gauss-Newton Algorithm 
        state_vector = state_vector + inv(k.T.dot(k)).dot(k.T).dot( measurement - current_modelled_measurement )

        # reassign iteration variables
        old_modelled_measurement = current_modelled_measurement 
        current_modelled_measurement = ForwardModel( state_vector ,measurement_object ,hitran_object)
        i += 1
        print(state_vector[4])
        rel_diff = np.abs((norm( current_modelled_measurement - measurement) - norm(old_modelled_measurement - measurement)) / norm(old_modelled_measurement - measurement))

        
    if FM == False:
        return state_vector
    else:
        return state_vector ,current_modelled_measurement ,measurement, measurement_object.spectral_grid 

def FitSpectra( state_vector ,measurement_object ,hitran_object, linear = True, FM = False):
    if linear:
        return LinearInversion( state_vector ,measurement_object, hitran_object, FM = FM)
    else:
        return NonlinearInversion( state_vector ,measurement_object ,hitran_object ,FM=FM)
    
        


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
 
def MakeInversionFunction( dataset_object, initial_guess ,linear = True ,FM = True):
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
        co2_state_vector  = FitSpectra( initial_guess ,co2_measurement_object ,co2_spectra_object , linear = False, FM = FM)
        dataset_object.temperature[measurement_index] = co2_state_vector[0][_temperature_index]
        hdo_measurement_object = Measurement( measurement_index ,hdo_min_wavenumber ,hdo_max_wavenumber ,dataset_object)
        ch4_measurement_object = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,dataset_object)

        #Calculate the line intensities and shapes
        ch4_spectra_object = HitranSpectra( ch4_measurement_object)        
        hdo_spectra_object = HitranSpectra( hdo_measurement_object )

        # Perform linear inversions
        ch4_state_vector = FitSpectra( initial_guess ,ch4_measurement_object ,ch4_spectra_object, linear = linear, FM = FM)
        hdo_state_vector  = FitSpectra( initial_guess ,hdo_measurement_object ,hdo_spectra_object ,linear = linear,FM = FM)

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

