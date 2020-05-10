import numpy as np
from scipy.interpolate import interp1d
from scipy.special import legendre

import pdb
#from inversion import TestLinearInversion

# define index

num_species = 3
H2O_index ,CH4_index ,CO2_index = 0 ,1 ,2 

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
        VMR_CH4 = state_vector[ CH4_index ]
        VMR_CO2 = state_vector[ CO2_index ]
        VMR_H2O = state_vector[ H2O_index ]
        shape_parameter_index = [i for i in range(num_species , len(state_vector))]
        shape_parameter = state_vector[ shape_parameter_index ]


        
    if type(state_vector) == dict:
        VMR_CH4 = state_vector[ 'VMR_CH4' ]
        VMR_CO2 = state_vector[ 'VMR_CO2' ]
        VMR_H2O = state_vector[ 'VMR_H2O' ]


        
    VCD_dry = measurement_object.VCD
    optical_depth = VCD_dry * (VMR_CH4 * cross_sections.CH4 + VMR_H2O * cross_sections.H2O + VMR_CO2 * cross_sections.CO2)
    transmission = np.exp( -optical_depth)
    return transmission, optical_depth
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
        'VMR_H2O' : input_vector[ H2O_index],
        'VMR_CH4' : input_vector[ CH4_index ],
        'VMR_CO2' : input_vector[ CO2_index ],
        'shape_parameter' : input_vector[ num_species :]
    }
    return output_dict


def ForwardModel(state_vector ,measurement_object ,hitran_object):
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
    

    
    transmission, optical_depth = CalculateTransmission( state_vector ,measurement_object ,hitran_object)
    transmission = DownSampleInstrument( hitran_object.grid ,transmission ,measurement_object.spectral_grid )
    
    shape_parameter_index = [i for i in range(num_species ,len(state_vector))]
    shape_parameter = state_vector[ shape_parameter_index ]

    polynomial_term = CalcPolynomialTerm( legendre_polynomial_degree ,shape_parameter , len(hitran_object.grid))
    polynomial_term = DownSampleInstrument( hitran_object.grid ,polynomial_term ,measurement_object.spectral_grid )
#    pdb.set_trace()

    
    f_out = np.log(transmission) + polynomial_term
    return f_out, transmission ,polynomial_term

        
    



def ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=True):
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


#    f ,transmission ,evaluated_polynomial = ForwardModel( state_vector ,measurement_object ,hitran_object )

        
    jacobian = np.empty( (measurement_object.spectral_grid.size ,len(state_vector)) ) # allocate memory for output jacobian
    # index for shape parameters for legendre
    shape_parameter_index = [i for i in range(num_species , len(state_vector))]
    shape_parameter = state_vector[ shape_parameter_index ] 

    
    # Calculate jacobian analytically 
    if linear:

#        evaluated_polynomial = CalcPolynomialTerm( legendre_polynomial_degree ,shape_parameter , len(hitran_object.grid) )

        # assign variable names for jacobian calculation 
        vcd = measurement_object.VCD        
        CO2_cross_sections = hitran_object.CO2
        H2O_cross_sections = hitran_object.H2O
        CH4_cross_sections = hitran_object.CH4

        # fill final jacobian with intensity jacobian
        jacobian[:, H2O_index] = DownSampleInstrument(hitran_object.grid , -1 * vcd * H2O_cross_sections ,measurement_object.spectral_grid)
        jacobian[: ,CH4_index] = DownSampleInstrument(hitran_object.grid ,-1 * CH4_cross_sections * vcd ,measurement_object.spectral_grid)
        jacobian[: ,CO2_index] = DownSampleInstrument(hitran_object.grid ,-1 * CO2_cross_sections * vcd ,measurement_object.spectral_grid)

        # Fill final jacobian with polynomial jacobian
        i=0
        x = np.linspace( -1, 1, hitran_object.grid.size)
        for index in shape_parameter_index:
            p = legendre(i)
            polynomial_jacobian = p(x)
#            pdb.set_trace()
            jacobian[ : ,index ] = DownSampleInstrument( hitran_object.grid , polynomial_jacobian, measurement_object.spectral_grid )
            i += 1
        # end of for-loop
    # end of linear if-conditional

	
    else: #calculate the derivative using finite differencing (Euler's method)

        # Define finite difference function
        def CalculateDerivative( perturbation_index ,state_vector):
            
            x_0 = state_vector.copy()
            dx = state_vector.copy()
            dx[ perturbation_index ] = 1.01 * dx[ perturbation_index ]
            df ,transmission ,evaluated_polynomial = ForwardModel( dx ,measurement_object ,hitran_object )

            df_dx = (df - f) / ( dx[ perturbation_index ] - x_0[ perturbation_index ])



            return df_dx
    # end of function CalculateDerivative

    # loop through each state vector element and calculate derivative via finite difference
        for state_vector_index in range( len( state_vector)  ):
#            print(len(state_vector))
#            print(state_vector_index)
            jacobian[ : , state_vector_index ] = CalculateDerivative( state_vector_index ,state_vector)
        # end of for-loop
        
    return jacobian
# end of function CalculateJacobian
