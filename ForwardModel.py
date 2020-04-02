from ReadData import *
from scipy.interpolate import interp1d
from scipy.special import legendre
from numpy.linalg import inv
import pdb


legendre_polynomial_degree = 20
num_species = 3
VMR_H2O_index ,VMR_CH4_index ,VMR_CO2_index ,shape_parameter_index = 0 ,1 ,2 ,[i for i in range(3, num_species + legendre_polynomial_degree + 1)]


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



def MakePolynomialGrid( polynomial_degree ,input_grid ,min_wavenumber, max_wavenumber):
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

    # Find index of corresponding spectral window
    window_indexes = np.array( np.where( (input_grid > min_wavenumber) & (input_grid < max_wavenumber) ))
    window_indexes = window_indexes[0,:]
    left_window_index ,right_window_index = window_indexes[0] ,window_indexes[-1]
    
    def GetPolynomialJacobian(_coefficients ,_spectral_grid):
        '''
calculate the polynomial jacobian with respect to the shape parameter.
p(lambda) = sum_i alpha_i * a_i * lambda^i for i = 0:n
partial p / partial alpha = sum_i a_i * lambda^i for i=0:n
n = polynomial_degree, lambda = wavenumber, a_i = ith legendre coefficient, alpha_i = ith shape parameter
'''


        num_polynomial_coefficients = np.shape(_coefficients)[0]
        num_spectral_points  = np.shape( _spectral_grid )[0]
        derivative_array = np.zeros( ( num_spectral_points , num_polynomial_coefficients ))

        # evaluate 
        for n in range( num_polynomial_coefficients ):
            degree = num_polynomial_coefficients -n - 1 # -1 comes from 0 indexing        
            derivative_array[ :, n ] = _coefficients[n] * _spectral_grid ** degree
        # end of for-loop
        
        return derivative_array
# end of function 


    polynomial_function = legendre( polynomial_degree ) # generate polynomial object
    legendre_coefficients = polynomial_function.coefficients
    
    shifted_grid = input_grid - np.mean(input_grid) # center spectral grid at 0
    # Evaluate Jacobian of polymnomial with respect to shape parameter
    polynomial_derivative_values = GetPolynomialJacobian( legendre_coefficients ,shifted_grid)
#    pdb.set_trace()
    
    # initialize output arrays
    polynomial_values_out = np.zeros( shifted_grid.shape)
    derivative_values_out = np.zeros( (np.size(shifted_grid) , np.size(legendre_coefficients) )) 
    polynomial_values_out[ left_window_index : right_window_index ] = polynomial_function( shifted_grid[ left_window_index : right_window_index ] )

    derivative_values_out[ left_window_index : right_window_index ,: ] = polynomial_derivative_values[ left_window_index : right_window_index ,: ]
#    derivative_values_out = polynomial_derivative_values


    return polynomial_values_out , derivative_values_out
# end of function EvaluatePolynomial


def ArrayToDict( input_vector ):
    output_dict = {
        'VMR_H2O' : input_vector[ VMR_H2O_index],
        'VMR_CH4' : input_vector[ VMR_CH4_index ],
        'VMR_CO2' : input_vector[ VMR_CO2_index ],
        'shape_parameter' : input_vector[ shape_parameter_index ]
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
    extinction = DownSampleInstrument( hitran_object.grid ,transmission ,measurement_object.spectral_grid )
    
    try:
        shape_parameter = state_vector['shape_parameter']
    except:
        shape_parameter = state_vector[ shape_parameter_index ]
        
    evaluated_polynomial, evaluated_derivative  = MakePolynomialGrid( legendre_polynomial_degree ,measurement_object.spectral_grid ,5952.4 ,6230.5 )
    
    f_out = np.log(extinction) + evaluated_polynomial
    return f_out ,transmission ,evaluated_polynomial

def DictToArray( input_dictionary ):
    output_vector = np.empty(( num_species + legendre_polynomial_degree + 1 ))

    output_vector[0] = input_dictionary[ 'VMR_H2O']
    output_vector[1] = input_dictionary[ 'VMR_CH4']
    output_vector[2] = input_dictionary[ 'VMR_CO2']
    output_vector[3:] = input_dictionary[ 'shape_parameter']
    return output_vector
        
    



def ComputeJacobian( state_vector ,measurement_object ,hitran_object ,linear=True):
    '''
Calculate the jacobian of the system state either analytically or numerically.

inputs:
1. state_vector: np.array or dict containing the system state (VMR and shape parameters).
measurement_object: Measurement class that contains the observations from frequency comb
hitran_object: HitranSpectra class that contains the hitran cross-sections
Linear=True: Boul that tells function to calculate analytically or numerically

outputs:b
jacobian: np.array that contains the jacobian that should be (num_spectral_points X num_state_vector_elements)
'''
    




    f ,transmission ,evaluated_polynomial = ForwardModel( state_vector ,measurement_object ,hitran_object )

    if type(state_vector) == dict:
        state_vector = DictToArray( state_vector )
        
    jacobian = np.empty( (f.size ,len(state_vector)) ) # allocate memory for output jacobian
    # index for shape parameters for legendre 
    shape_parameter = state_vector[ shape_parameter_index ] 
    evaluated_poblynomial ,evaluated_polynomial_derivative = MakePolynomialGrid( legendre_polynomial_degree ,hitran_object.grid ,5952.4 ,6230.5 ) # evaluate legendre polynomial
    
    # Calculate jacobian analytically 
    if linear:

        # assign variable names for jacobian calculation 
        vcd = measurement_object.VCD
        CO2_cross_sections = hitran_object.CO2
        H2O_cross_sections = hitran_object.H2O
        CH4_cross_sections = hitran_object.CH4

        # fill final jacobian with intensity jacobian
        jacobian[: ,0] = DownSampleInstrument(hitran_object.grid , -1*transmission * vcd * H2O_cross_sections ,measurement_object.spectral_grid)
        jacobian[: ,1] = DownSampleInstrument(hitran_object.grid ,-1 * CH4_cross_sections * vcd * transmission,measurement_object.spectral_grid)
        jacobian[: ,2] = DownSampleInstrument(hitran_object.grid ,-1 * CO2_cross_sections * vcd * transmission,measurement_object.spectral_grid)

        # Fill final jacobian with polynomial jacobian
        i=0        
        for index in shape_parameter_index:
#            pdb.set_trace()
            jacobian[ : ,index ] = DownSampleInstrument( hitran_object.grid , evaluated_polynomial_derivative[ :, i ], measurement_object.spectral_grid )
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
            state_vector = ArrayToDict( state_vector )
            return df_dx
    # end of function CalculateDerivative

    # loop through each state vector element and calculate derivative via finite difference
        for state_vector_index in range( len( state_vector )  ):
            jacobian[ : , state_vector_index ] = CalculateDerivative( state_vector_index ,state_vector)
        # end of for-loop
        
    return jacobian
# end of function CalculateJacobian






truth = {
    'VMR_H2O' : 0.2,
    'VMR_CH4' : 2000e-9,
    'VMR_CO2' : 500e-6,
    'shape_parameter' : 2*np.ones( legendre_polynomial_degree + 1)
    }


file = 'testdata_2.h5'
data = CombData(file)
current_data = Measurement( 1000 , data)
GetCrossSections( current_data.min_wavenumber ,current_data.max_wavenumber)
spectra = HitranSpectra( current_data)

k = ComputeJacobian( truth ,current_data, spectra, linear = True)
f, transmission ,polynomial_grid = ForwardModel( truth ,current_data ,spectra)
x = inv(k.T.dot(k)).dot(k.T).dot(f)
