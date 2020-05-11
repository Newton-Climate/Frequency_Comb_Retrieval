from numpy.linalg import inv, norm
from numpy import dot
import numpy as np
from ForwardModel import *
from ReadData import *



def CalculateChi2( modelled_measurement ,measurement ,state_vector):
    degrees_of_freedom = measurement.size - state_vector.size
    diff =  modelled_measurement - measurement
    error = norm(diff)
    chi2 = error / np.var(diff) /  degrees_of_freedom
    return chi2



def LinearInversion( state_vector ,measurement_object ,hitran_object):

    k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=True)
    ans = inv(k.T.dot(k)).dot(k.T).dot( np.log(measurement_object.FC) )
#    pdb.set_trace()
   
    modelled_measurement = np.exp (k.dot(ans))


    return ans

def TestLinearInversion( test_state_vector ,true_state_vector ,measurement_object ,hitran_object):
    
    f_true ,transmission ,evaluated_polynomial = ForwardModel(true_state_vector ,measurement_object, hitran_object)
    k = ComputeJacobian( test_state_vector ,measurement_object ,hitran_object ,linear=True)

    ans = inv(k.T.dot(k)).dot(k.T).dot(np.log(f_true))
#    f_test ,transmission ,evaluated_polynomial = ForwardModel(ans ,current_data ,spectra)
    return ArrayToDict(ans) 


def NonlinearInversion( initial_guess ,measurement_object ,hitran_object):

    measurement = measurement_object.FC
    state_vector = initial_guess
    chi2_difference = 1
    chi2_old = 0
    i = 0
    state_vector = DictToArray(state_vector)    
    while i <1:

        modelled_measurement ,transmission ,evaluated_polynomial = ForwardModel( state_vector ,measurement_object ,hitran_object)
        k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=False)
        

        #evaluate the equation from Rogers 2000
        dx = state_vector + inv(k.T.dot(k)).dot(k.T).dot( measurement - modelled_measurement )
        state_vector = dx
        chi2_new = CalculateChi2( modelled_measurement ,measurement ,state_vector)
        print(chi2_new)
        i += 1
        print(i, '\n')

        chi2_old = chi2_new
        chi2_difference = np.abs( chi2_old - chi2_new )
    return state_vector ,chi2_difference
        
def InvertAllData( initial_guess ,dataset_object, hitran_object):
    num_measurements = dataset_object.num_measurements
    ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
    co2_min_wavenumber, co2_max_wavenumber = 6180, 6250



    H2O_con = np.empty(( num_measurements ))
    CH4_con = np.empty(( num_measurements ))
    CO2_con = np.empty(( num_measurements ))


    for measurement_index in range(num_measurements):
        print(measurement_index)


        ch4_measurement_object = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,dataset_object)
        co2_measurement_object = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,dataset_object)
        ch4_spectra_object = HitranSpectra( ch4_measurement_object)
        co2_spectra_object = HitranSpectra( co2_measurement_object )

        ch4_state_vector  = LinearInversion( initial_guess ,ch4_measurement_object ,ch4_spectra_object)
        co2_state_vector  = LinearInversion( initial_guess ,co2_measurement_object ,co2_spectra_object)


        H2O_con[ measurement_index ] = co2_state_vector[ 'VMR_H2O' ]
        CH4_con[ measurement_index ] = ch4_state_vector[ 'VMR_CH4' ]
        CO2_con[ measurement_index ] = co2_state_vector[ 'VMR_CO2' ]

    # end of for loop
    return H2O_con, CH4_con ,CO2_con 
# end of function InvertAllData
 
def MakeInversionFunction( dataset_object, initial_guess):
    num_measurements = dataset_object.num_measurements
    ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
    co2_min_wavenumber, co2_max_wavenumber = 6180, 6250
    global Invert
    

    def Invert( measurement_index):
        print(measurement_index)
        output_vector = np.empty( num_species + 2)

        ch4_measurement_object = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,dataset_object)
        co2_measurement_object = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,dataset_object)
        ch4_spectra_object = HitranSpectra( ch4_measurement_object)
        co2_spectra_object = HitranSpectra( co2_measurement_object )

        ch4_state_vector  = LinearInversion( initial_guess ,ch4_measurement_object ,ch4_spectra_object)
        co2_state_vector  = LinearInversion( initial_guess ,co2_measurement_object ,co2_spectra_object)

        # assign to output vector
        output_vector[ H2O_index ] = co2_state_vector[ H2O_index ]
        output_vector[ CH4_index ] = ch4_state_vector[ CH4_index ]
        output_vector[ CO2_index ] = co2_state_vector[ CO2_index ]
        output_vector[ num_species ] = ch4_measurement_object.VCD # put that in the output just in case
        output_vector[ num_species + 1] = ch4_measurement_object.time
        return output_vector
    return Invert

def InvertParallel( inversion_function, num_measurements):
    pool = mp.Pool(4)
    result = pool.map( inversion_function ,[i for i in range(num_measurements )])

    # order results by time stamp
    try:
        result = sorted(result ,key = lambda x: x[-1])
        print('Done sorting output')
        return np.array(result)
    except:
        return result

    
    


    


ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
co2_min_wavenumber, co2_max_wavenumber = 6180, 6250    
file = 'testdata_2.h5'
data = CombData(file)
legendre_polynomial_degree = 20
vcd = 1


guess = {
    'VMR_H2O' : 0.02 * vcd,
    'VMR_CH4' : 3000e-9 * vcd,
    'VMR_CO2' : 400e-6 * vcd,
    'shape_parameter' : np.ones( legendre_polynomial_degree +1 )
    }

truth = {
    'VMR_H2O' : 0.02 * vcd,
    'VMR_CH4' : 2000e-9 * vcd,
    'VMR_CO2' : 400e-6 * vcd,
    'shape_parameter' : np.ones( legendre_polynomial_degree + 1 )*0.1
    }

truth = DictToArray(truth)
measurement_index = 50
ch4_data = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,data)
current_data = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,data)


try:
    spectra = HitranSpectra( current_data)
except:
    GetCrossSections( data.min_wavenumber ,data.max_wavenumber)
    spectra = HitranSpectra( current_data, legendre_polynomial_degree = 40)
ch4_spectra = HitranSpectra(ch4_data)

co2_ans = LinearInversion( truth ,current_data ,spectra)
ch4_ans = LinearInversion( truth ,ch4_data ,ch4_spectra)


#np.savez('fitting_data' ,co2_obs = current_data.FC, ch4_obs = ch4_data.FC, co2_grid = #current_data.spectral_grid, ch4_grid = ch4_data.spectral_grid, co2_k = k_co2, ch4_k = k_ch4, #ch4_ans = ch4_ans, co2_ans = co2_ans)


#H2O, CH4, CO2 = InvertAllData( truth ,data ,spectra)
#np.savez('CH4_linear_inversion', CO2 = CO2, H2O = H2O, CH4 = CH4)
global my_func
my_func = MakeInversionFunction( data, truth)
result = InvertParallel( my_func, data.num_measurements)

