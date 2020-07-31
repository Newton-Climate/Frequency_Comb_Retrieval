import numpy as np
from scipy.interpolate import interp1d
from scipy.special import legendre

import pdb
from ReadData import *

# define index

num_species = 4
_H2O_index ,_CH4_index ,_CO2_index ,_HDO_index = [i for i in range(num_species)]
_temperature_index = num_species
_windspeed_index = num_species + 1

def DictToArray( input):
    output = np.hstack( (input[key] for key in input) )
    return output


def CalculateTransmission(state_vector ,measurement_object ,hitran_object):
    '''
Calculate the extinction from Beer's Law.

inputs:
1. state_vector: The state vector containing the concentrations. can be either a dict or array.
2. measurement_object: Measurement class that contains the observations from frequency comb
3. hitran_object: HitranSpectra class that contains the hitran cross-sections

Outputs:
1. optical depth from Beer's law: exp(-tau)
2. optical depth: tau = VCD * (VMR_gas_i * gas_cross_sections_i) for each gas i
'''
    
    cross_sections = hitran_object
    
    if type(state_vector) == np.ndarray:
        VMR_CH4 = state_vector[ _CH4_index ]
        VMR_CO2 = state_vector[ _CO2_index ]
        VMR_H2O = state_vector[ _H2O_index ]
        #VMR_13CO2 = state_vector[ _13CO2_index ]
        VMR_HDO = state_vector[ _HDO_index ]
        
    if type(state_vector) == dict:
        VMR_CH4 = state_vector[ 'VMR_CH4' ]
        VMR_CO2 = state_vector[ 'VMR_CO2' ]
        VMR_H2O = state_vector[ 'VMR_H2O' ]
        #VMR_13CO2 = state_vector[ 'VMR_13CO2' ]
        VMR_HDO = state_vector[ 'VMR_HDO' ]


        
    VCD_dry = measurement_object.VCD
    optical_depth = VCD_dry * (VMR_CH4 * cross_sections._CH4 + VMR_H2O * cross_sections._H2O + VMR_CO2 * cross_sections._CO2)
    optical_depth_isotopes = VCD_dry * (VMR_HDO * cross_sections._HDO)
    transmission = np.exp( -optical_depth - optical_depth_isotopes )
    return transmission
# end of function CalculateTransmission


def DownSampleInstrument( input_spectral_grid ,input_spectra ,output_spectral_grid ):
    '''

Function for downsampling hitran grid to instrument grid.
Interpolates x, y points and evaluates at new grid x_new

Inputs:
1. input_spectral_grid: (n,) np.array that contains the x points 
2. input_spectra: (n,) np.array containing cross-sections 
3. output_spectral_grid (p,) np.array that contains the instrument grid
    
outputs:
1. output_spectral_grid: the downsampled spectra
'''

    finterp = interp1d( input_spectral_grid ,input_spectra ,kind='cubic')
    output_spectra = finterp( output_spectral_grid)
    return output_spectra
# end of function DownSampleInstrument



def CalcPolynomialTerm( max_legendre_polynomial_degree ,shape_parameters , wavenumber_grid_length ):
    '''
Makes the Legendre Polynomial Array used in ForwardModel and its corresponding Jacobian. 
Evaluates polynomial and derivative only over the given window (between min and max_wavenumber).
Fills in all other outside spectral window with ones.

inputs:
1. polynomial_degree: int that gives the degree of the legendre polynomial
2. input_grid: np.array containing wavelengths that the polynomial will be evaluated on
3. min_wavenumber: Float64 the first boundary of the polynomial evaluation window
4. max_wavenumber: Float64: The last wavenumber of the polynomial evaluation window

Outputs:
1. polynomial_values_out: np.array containing evalauted polynomialb
2. derivative_values_out: np.array that contains the evaluated polynomial derivative 
    '''

    # define the domain of Legendre polynomial
    x = np.linspace(-1 ,1 ,wavenumber_grid_length )
    polynomial_term = np.zeros( x.shape )

    for degree in range( max_legendre_polynomial_degree+1):


        p = legendre( degree)
        polynomial_term += shape_parameters[ degree ] * p(x)
    return polynomial_term
# end of function EvaluatePolynomial


def ArrayToDict( input_vector ):
    output_dict = {
        'VMR_H2O' : input_vector[ _H2O_index],
        'VMR_CH4' : input_vector[ _CH4_index ],
        'VMR_CO2' : input_vector[ _CO2_index ],
        'VMR_HDO' : input_vector[ _HDO_index],
        #'VMR_13CO2' : input_vector[ _13CO2_index ],
        'shape_parameter' : input_vector[ num_species :]
    }
    return output_dict

def DopplerShift( relative_speed ,spectral_grid):
    c = 299792458 * 100 # speed of light in cm/s

    # calculate shift
    shift1 = (relative_speed/c) * spectral_grid.copy()
    shift2 = -(relative_speed/c) * spectral_grid.copy()
    
    coef1 = np.ones( spectral_grid.shape) + shift1
    coef2 = np.ones( spectral_grid.shape) + shift2
    spectral_grid_out = coef1 * coef2 * spectral_grid
    return spectral_grid_out


def ForwardModel(state_vector ,measurement_object ,hitran_object, flags = None):
    '''
Evaluate the forward model: f = log(i) + p(lambda)
i = exp( - tau) from Beer's Law
p(lambda) = legendre polynomial

inputs:
1. state_vector: a dict or array containing the system state
2. measurement_object: Measurement class that contains the observations from frequency comb
3. hitran_object: HitranSpectra class that contains the hitran cross-sections

outputs:
1. f_out: evaluated forward model
2. transmission: (just for debugging)
3. evaluated_polynomial: just for debugging
'''
    
    transmission = CalculateTransmission( state_vector ,measurement_object ,hitran_object)
    spectral_grid = measurement_object.spectral_grid.copy()
    if flags['fit_windspeed']:
        spectral_grid = DopplerShift( state_vector[ _windspeed_index] , spectral_grid)

    transmission = DownSampleInstrument( hitran_object.grid ,transmission ,spectral_grid)

    if flags['fit_windspeed']:
        shape_parameter_index = [i for i in range(num_species + 2,len(state_vector))]
    else:
        shape_parameter_index = [i for i in range(num_species + 1, len(state_vector))]
        
    shape_parameter = state_vector[ shape_parameter_index ]

    legendre_polynomial_degree = len( shape_parameter ) - 1
    polynomial_term = CalcPolynomialTerm( legendre_polynomial_degree ,shape_parameter , len(hitran_object.grid))

    # Define the legendre polynomial grid 
    polynomial_term = DownSampleInstrument( hitran_object.grid ,polynomial_term ,measurement_object.spectral_grid )
    f_out = np.log(transmission) + polynomial_term
    return f_out

def TemperatureJacobian( state_vector, measurement_object ,original_run, flags = None):
    T0 = measurement_object.temperature
    T1 = T0 + 5

    x1 = state_vector.copy()
    x1[_temperature_index] = T1

    # calculate the new cross-sections given the perturbed temperature
    new_hitran_object = HitranSpectra( measurement_object ,temperature = T1, use_hitran08 = flags['use_hitran08'])
    
    f0 = original_run
    f1 = ForwardModel( x1, measurement_object , new_hitran_object, flags = flags)
    dfdT = (f1 - f0) / (T1 - T0)

    return dfdT
    



def ComputeJacobian( state_vector ,measurement_object ,hitran_object ,flags = None ,linear=True):
    '''
Calculate the jacobian of the system state either analytically or numerically.

inputs:
1. state_vector: np.array or dict containing the system state (VMR and shape parameters).
measurement_object: Measurement class that contains the observations from frequency comb
hitran_object: HitranSpectra class that contains the hitran cross-sections
Linear=True: Boul that tells function to calculate analytically or numerically

outputs:
jacobian: np.array that contains the jacobian that should be (num_spectral_points X num_state_vector_elements)
'''

    # run the forward model once for reference
    base_run = ForwardModel( state_vector, measurement_object ,hitran_object, flags)        
    jacobian = np.empty( (measurement_object.spectral_grid.size ,len(state_vector)) ) # allocate memory for output jacobian
    
    # index for shape parameters for legendre
    if flags['fit_windspeed']:
        shape_parameter_index = [i for i in range(num_species + 2, len(state_vector))]
    else:
        shape_parameter_index = [i for i in range(num_species + 1, len(state_vector))]
        shape_parameter = state_vector[ shape_parameter_index ] 

    
    # Calculate jacobian analytically 
    if linear:

#        evaluated_polynomial = CalcPolynomialTerm( legendre_polynomial_degree ,shape_parameter , len(hitran_object.grid) )

        # assign variable names for jacobian calculation 
        vcd = measurement_object.VCD        
        _CO2_cross_sections = hitran_object._CO2
        _H2O_cross_sections = hitran_object._H2O
        _CH4_cross_sections = hitran_object._CH4
        #_13CO2_cross_sections = hitran_object._13CO2
        _HDO_cross_sections = hitran_object._HDO

        # fill final jacobian with intensity jacobian
        jacobian[:, _H2O_index] = DownSampleInstrument(hitran_object.grid , -1 * _H2O_cross_sections * vcd,measurement_object.spectral_grid)
        jacobian[: ,_CH4_index] = DownSampleInstrument(hitran_object.grid ,-1 * _CH4_cross_sections * vcd ,measurement_object.spectral_grid)
        jacobian[: ,_CO2_index] = DownSampleInstrument(hitran_object.grid ,-1 * _CO2_cross_sections * vcd ,measurement_object.spectral_grid)
        jacobian[:, _HDO_index] = DownSampleInstrument(hitran_object.grid , -1 * _HDO_cross_sections * vcd,measurement_object.spectral_grid)
        #jacobian[: ,_13CO2_index] = DownSampleInstrument(hitran_object.grid ,-1 * _13CO2_cross_sections * vcd ,measurement_object.spectral_grid)
        jacobian[ : , _temperature_index ] = TemperatureJacobian( state_vector ,measurement_object ,base_run)
        

        # Fill final jacobian with polynomial jacobian
        i=0
        x = np.linspace( -1, 1, hitran_object.grid.size)
        for index in shape_parameter_index:
            p = legendre(i)
            polynomial_jacobian = p(x)

            jacobian[ : ,index ] = DownSampleInstrument( hitran_object.grid , polynomial_jacobian, measurement_object.spectral_grid )
            i += 1
        # end of for-loop
    # end of linear if-conditional

	
    else: #calculate the derivative using finite differencing (Euler's method)
        
        
        # Define finite difference function
        def CalculateDerivative( perturbation_index ,state_vector):
            
            x0 = state_vector.copy()
            x1 = state_vector.copy()
            x1[ perturbation_index ] = 1.01 * x1[ perturbation_index ]
            
            if x0[perturbation_index] <= 1e-9:
                x1[perturbation_index] = 1e-3
                
            dx = x1[perturbation_index] - x0[perturbation_index]
            f0 = base_run
            f1 = ForwardModel( x1 ,measurement_object ,hitran_object, flags )
            
            df_dx = (f1 - f0) / dx

            return df_dx
    # end of function CalculateDerivative

    # loop through each state vector element and calculate derivative via finite difference
        for state_vector_index in range( len( state_vector)  ):
#            print(len(state_vector))
            if state_vector_index != _temperature_index:
                jacobian[ : , state_vector_index ] = CalculateDerivative( state_vector_index ,state_vector)
            elif state_vector_index == _temperature_index:
                jacobian[ : , _temperature_index ] = TemperatureJacobian( state_vector ,measurement_object ,base_run, flags = flags)
                # end of for-loop
#    pdb.set_trace()
    return jacobian
# end of function CalculateJacobian
