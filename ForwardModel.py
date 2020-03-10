from ReadData import *
from scipy.interpolate import interp1d
from scipy.special import legendre
import pdb

def CalculateTransmission(state_vector ,measurement_object ,hitran_object):
    cross_sections = hitran_object
    if type(state_vector) == np.ndarray:
        state_vector = ArrayToDict( state_vector )
    if type(state_vector) == dict:
        VMR_CH4 = state_vector[ 'VMR_CH4' ]
        VMR_CO2 = state_vector[ 'VMR_CO2' ]
        VMR_H2O = state_vector[ 'VMR_H2O' ]

        
        VCD_dry = measurement_object.VCD
    optical_depth = VCD_dry * (VMR_CH4 * cross_sections.CH4 + VMR_H2O * cross_sections.H2O + VMR_CO2 * cross_sections.CO2)
    solar_spectrum = hitran_object.solar_spectrum
    transmission = solar_spectrum * np.exp( -optical_depth)
    return transmission, optical_depth

def DownSampleInstrument( input_spectral_grid ,input_spectra ,output_spectral_grid ):
    
    """
down-sample the instrument grid to the HITRAN instrument, because the instrument is too high-res

inputs:
dataset_object: the object containing the instrument measurement (measurement class)
hitran_object: the object containing the HITRAN cross-sections (hitran class)

output:
downsampled_spectra: the downscaled instrument spectra
downsampled_grid: the lower resolution spectral grid
"""

    finterp = interp1d( input_spectral_grid ,input_spectra ,kind='cubic')
    return finterp( output_spectral_grid)


def GenerateLegendrePolynomial(degree):
    coefficients = legendre(degree ,monic= True)
    polynomial = np.poly1d(coefficients)
    derivative = np.polyder(polynomial)
    return polynomial ,derivative 

x = {
    'VMR_H2O' : 0.01,
    'VMR_CH4' : 1700e-9,
    'VMR_CO2' : 400e-6
    }

def ArrayToDict( input_vector ):
    output_dict = {
        'VMR_H2O' : input_vector[0],
        'VMR_CH4' : input_vector[1],
    'VMR_CO2' : input_vector[2]
    }
    return output_dict


def ForwardModel(state_vector ,measurement_object ,hitran_object):
    
    transmission, optical_depth = CalculateTransmission( state_vector ,measurement_object ,hitran_object)
    instrument_spectra = DownSampleInstrument( hitran_object.grid ,transmission ,measurement_object.spectral_grid )
    legendre_polynomial ,polynomial_derivative = GenerateLegendrePolynomial(5)


    def MakePolynomialGrid( polynomial_function ,input_grid ,min_wavenumber, max_wavenumber):
        window_indexes = np.array( np.where( (input_grid > min_wavenumber) & (input_grid < max_wavenumber) ) )
        window_indexes = window_indexes[0,:]
        left_window_index ,right_window_index = window_indexes[0] ,window_indexes[-1]
        
        shifted_grid = input_grid - np.mean(input_grid)
        output_values = np.ones( shifted_grid.shape )
        output_values[ left_window_index : right_window_index ] = polynomial_function( shifted_grid[ left_window_index : right_window_index ] )
        return output_values

    evaluated_polynomial = MakePolynomialGrid( legendre_polynomial ,measurement_object.spectral_grid ,5952.4 ,6230.5 )
    
    irr = instrument_spectra * evaluated_polynomial
    return irr

def DictToArray( input_dictionary ):

    output_vector = [ input_dictionary[key] for key in input_dictionary ]
    output_vector = np.array(output_vector)
    return output_vector
        
    



def ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=True):
    """
Analytically comptue the jacobian of the forward model

inputs:
1. State Vector
2. measurement_object
3. hitran_object

outputs:
Jacobain: the jacobain of hte forward model
"""

    f = ForwardModel( state_vector ,measurement_object ,hitran_object)
    if type(state_vector) == dict:
        state_vector = DictToArray( state_vector )

    jacobian = np.empty( (f.size ,len(state_vector)) )
    print(state_vector) # need to convert to a array

    if linear:
        f, g = CalculateTransmission(state_vector ,measurement_object ,hitran_object)
        vcd = measurement_object.VCD
        CO2_cross_sections = hitran_object.CO2
        H2O_cross_sections = hitran_object.H2O
        CH4_cross_sections = hitran_object.CH4
        print(f * vcd * H2O_cross_sections - f*vcd*CO2_cross_sections)
        
        jacobian[: ,0] = DownSampleInstrument(hitran_object.grid , f * vcd * H2O_cross_sections ,measurement_object.spectral_grid)
        jacobian[: ,1] = DownSampleInstrument(hitran_object.grid ,CH4_cross_sections * vcd * f,measurement_object.spectral_grid)
        jacobian[: ,2] = DownSampleInstrument(hitran_object.grid ,CO2_cross_sections * vcd * f,measurement_object.spectral_grid)
#        pdb.set_trace()
	
    else: #calculate the derivative using finite differencing (Euler's method)

        def CalculateDerivative( perturbation_index ,state_vector):


            x_0 = state_vector.copy()
            dx = state_vector.copy()
            dx[ perturbation_index ] = 1.001 * dx[ perturbation_index ]
            df = ForwardModel( dx ,measurement_object ,hitran_object )
            df_dx = (df - f) / ( dx[ perturbation_index ] - x_0[ perturbation_index ])
            state_vector = ArrayToDict( state_vector )
            return df_dx
        

        for i in range( len( state_vector ) ):
            jacobian[ : ,i ] = CalculateDerivative( i ,state_vector)
            # end of for-loop

    return jacobian
# end of function CalculateJacobian


    
f = ForwardModel(x ,current_data ,spectra)
k = ComputeJacobian( x ,current_data ,spectra ,linear=True)
from numpy.linalg import inv
from numpy import dot

ans = inv(k.T.dot(k)).dot(k.T).dot(np.log(f))
t, od = CalculateTransmission( x ,current_data,spectra)
import matplotlib.pyplot as plt
plt.plot(spectra.grid ,t)
plt.xlabel(r'$cm^{-1}$')
plt.ylabel('Transmission')
plt.savefig('test.pdf')
