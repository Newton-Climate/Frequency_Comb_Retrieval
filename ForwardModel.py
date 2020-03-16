from ReadData import *
from scipy.interpolate import interp1d
from scipy.special import legendre
import pdb

VMR_H2O_index ,VMR_CH4_index ,VMR_CO2_index ,shape_parameter_index = 0 ,1 ,2 ,3
legendre_polynomial_degree = 5


def CalculateTransmission(state_vector ,measurement_object ,hitran_object):
    cross_sections = hitran_object
    if type(state_vector) == np.ndarray:
        VMR_CH4 = state_vector[ VMR_CH4_index ]
        VMR_CO2 = state_vector[ VMR_CO2_index ]
        VMR_H2O = state_vector[ VMR_H2O_index ]
        shape_parameter = state_vector[ shape_parameter_index ]


        
    if type(state_vector) == dict:
        VMR_CH4 = state_vector[ 'VMR_CH4' ]
        VMR_CO2 = state_vector[ 'VMR_CO2' ]
        VMR_H2O = state_vector[ 'VMR_H2O' ]


        
    VCD_dry = measurement_object.VCD
    optical_depth = VCD_dry * (VMR_CH4 * cross_sections.CH4 + VMR_H2O * cross_sections.H2O + VMR_CO2 * cross_sections.CO2)
    solar_spectrum = hitran_object.solar_spectrum
    transmission = np.exp( -optical_depth)
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



def MakePolynomialGrid( polynomial_degree ,shape_parameter ,input_grid ,min_wavenumber, max_wavenumber):
    window_indexes = np.array( np.where( (input_grid > min_wavenumber) & (input_grid < max_wavenumber) ) )
    window_indexes = window_indexes[0,:]
    left_window_index ,right_window_index = window_indexes[0] ,window_indexes[-1]

    def GenerateLegendrePolynomial(degree_ ,shape_parameter_):
        coefficients = legendre(degree_ ,monic= False)
        coefficients = shape_parameter_ * coefficients
        polynomial = np.poly1d(coefficients)
        derivative = np.polyder(polynomial)
        return polynomial 

    polynomial_function = GenerateLegendrePolynomial( polynomial_degree ,shape_parameter )
    shifted_grid = input_grid - np.mean(input_grid)
    output_values = np.ones( shifted_grid.shape )
    output_values[ left_window_index : right_window_index ] = polynomial_function( shifted_grid[ left_window_index : right_window_index ] )
    return output_values



def ArrayToDict( input_vector ):
    output_dict = {
        'VMR_H2O' : input_vector[ VMR_H2O_index],
        'VMR_CH4' : input_vector[ VMR_CH4_index ],
        'VMR_CO2' : input_vector[ VMR_CO2_index ],
        'shape_parameter' : input_vector[ shape_parameter_index ]
    }
    return output_dict


def ForwardModel(state_vector ,measurement_object ,hitran_object):
    
    transmission, optical_depth = CalculateTransmission( state_vector ,measurement_object ,hitran_object)
    instrument_spectra = DownSampleInstrument( hitran_object.grid ,transmission ,measurement_object.spectral_grid )
    
    try:
        shape_parameter = state_vector['shape_parameter']
    except:
        shape_parameter = state_vector[ shape_parameter_index ]
        degree = 5




    evaluated_polynomial = MakePolynomialGrid( legendre_polynomial_degree ,shape_parameter ,measurement_object.spectral_grid ,5952.4 ,6230.5 )
    
    irr = instrument_spectra #* evaluated_polynomial
    return irr ,transmission ,evaluated_polynomial

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

    f ,transmission ,evaluated_polynomial = ForwardModel( state_vector ,measurement_object ,hitran_object )
    if type(state_vector) == dict:
        state_vector = DictToArray( state_vector )

    jacobian = np.empty( (f.size ,len(state_vector)) )


    shape_parameter = state_vector[ shape_parameter_index ]
    evaluated_polynomial = MakePolynomialGrid( legendre_polynomial_degree ,shape_parameter ,hitran_object.grid ,5952.4 ,6230.5 )

    if linear:

        vcd = measurement_object.VCD
        CO2_cross_sections = hitran_object.CO2
        H2O_cross_sections = hitran_object.H2O
        CH4_cross_sections = hitran_object.CH4

        
        jacobian[: ,0] = DownSampleInstrument(hitran_object.grid , -1*transmission * vcd * H2O_cross_sections ,measurement_object.spectral_grid)
        jacobian[: ,1] = DownSampleInstrument(hitran_object.grid ,-1 * CH4_cross_sections * vcd * transmission,measurement_object.spectral_grid)
        jacobian[: ,2] = DownSampleInstrument(hitran_object.grid ,-1 * CO2_cross_sections * vcd * transmission,measurement_object.spectral_grid)
#        jacobian[:,3] = np.ones(measurement_object.spectral_grid.shape)
        jacobian[:,3] = DownSampleInstrument(hitran_object.grid ,transmission * evaluated_polynomial ,measurement_object.spectral_grid) 
#        pdb.set_trace()
	
    else: #calculate the derivative using finite differencing (Euler's method)

        def CalculateDerivative( perturbation_index ,state_vector):


            x_0 = state_vector.copy()
            dx = state_vector.copy()
            dx[ perturbation_index ] = 1.01 * dx[ perturbation_index ]
            df ,transmission ,evaluated_polynomial = ForwardModel( dx ,measurement_object ,hitran_object )
#            pdb.set_trace()
            df_dx = (df - f) / ( dx[ perturbation_index ] - x_0[ perturbation_index ])
            state_vector = ArrayToDict( state_vector )
            return df_dx
        

        for state_vector_index in range( len( state_vector )  ):
            jacobian[ : , state_vector_index ] = CalculateDerivative( state_vector_index ,state_vector)
            # end of for-loop

            # Evaluate the shape parameter jacobian explicitly, because finite-difference is unstable 
#        jacobian[ : ,shape_parameter_index ] = DownSampleInstrument( hitran_object.grid ,transmission * evaluated_polynomial ,measurement_object.spectral_grid )

    return jacobian
# end of function CalculateJacobian


