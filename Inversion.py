from numpy.linalg import inv, norm
from numpy import dot
from ForwardModel import *




guess = {
    'VMR_H2O' : 0.2,
    'VMR_CH4' : 3000e-9,
    'VMR_CO2' : 400e-6,
    'shape_parameter' : np.ones( legendre_polynomial_degree )
    }

truth = {
    'VMR_H2O' : 0.2,
    'VMR_CH4' : 2000e-9,
    'VMR_CO2' : 500e-6,
    'shape_parameter' : np.ones( legendre_polynomial_degree )
    }


def CalculateChi2( modelled_measurement ,measurement ,state_vector):
    degrees_of_freedom = measurement.size - state_vector.size
    chi2 = norm( (modelled_measurement - measurement) ) / degrees_of_freedom
    return chi2

def LinearInversion( state_vector ,measurement_object ,hitran_object):

    k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=True)
    ans = inv(k.T.dot(k)).dot(k.T).dot(np.log( measurement_object.FC ))
   
    modelled_measurement ,transmission ,evaluated_polynomial = ForwardModel( ans ,measurement_object ,hitran_object)
    chi2 = CalculateChi2( measurement_object.FC , modelled_measurement ,ans)
    ans = ArrayToDict(ans)
    return ans, chi2

def TestLinearInversion( state_vector ,measurement_object ,hitran_object):
    f ,transmission ,evaluated_polynomial = ForwardModel(state_vector ,current_data ,spectra)
    k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=True)
    ans = inv(k.T.dot(k)).dot(k.T).dot(np.log( f ))
    return ArrayToDict(ans) , f


def NonlinearInversion( initial_guess ,measurement_object ,hitran_object):

    measurement = measurement_object.FC
    state_vector = initial_guess
    chi2_difference = 1
    chi2_old = 0
    i = 0
    
    while chi2_difference > 1 and i < 10:
        modelled_measurement ,transmission ,evaluated_polynomial = ForwardModel( state_vector ,measurement_object ,hitran_object)
        k = ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=False)
        

        #evaluate the equation from Rogers 2000
        dx = state_vector - inv(k.T.dot(k)).dot(k.T).dot( measurement - modelled_measurement )
        state_vector = dx
        chi2_new = CalculateChi2( modelled_measurement ,measurement ,state_vector)
        print(chi2_new)
        i += 1
        print(i, '\n')

        chi2_old = chi2_new
        chi2_difference = np.abs( chi2_old - chi2_new )
    return state_vector ,chi2_difference
        
        
    



file = 'testdata_2.h5'
data = CombData(file)
current_data = Measurement( 1000 , data)
GetCrossSections( current_data.min_wavelength ,current_data.max_wavelength)
spectra = HitranSpectra( current_data)

ans, f = TestLinearInversion( truth, current_data ,spectra)
k = ComputeJacobian( truth ,current_data, spectra, linear = False)
f, transmission ,polynomial_grid = ForwardModel( truth ,current_data ,spectra)
x = inv(k.T.dot(k)).dot(k.T).dot(np.log( f))
