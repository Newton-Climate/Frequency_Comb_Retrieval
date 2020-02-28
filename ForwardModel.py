from ReadData import *
from scipy.interpolate import interp1d

def CalculateTransmission(state_vector ,measurement_object ,hitran_object):
    cross_sections = hitran_object
    VMR_CH4 = state_vector[ 'VMR_CH4' ]
    VMR_CO2 = state_vector[ 'VMR_CO2' ]
    VMR_H2O = state_vector[ 'VMR_H2O' ]
    VCD_dry = measurement_object.VCD
    optical_depth = VCD_dry * (VMR_CH4 * cross_sections.CH4 + VMR_H2O * cross_sections.H2O + VMR_CO2 * cross_sections.CO2)
    transmission = np.exp(optical_depth)
    return transmission, optical_depth

def DownSampleInstrument( input_spectral_grid ,input_spectra,output_spectral_grid ):
    
    """
down-sample the instrument grid to the HITRAN instrument, because the instrument is too high-res

inputs:
dataset_object: the object containing the instrument measurement (measurement class)
hitran_object: the object containing the HITRAN cross-sections (hitran class)

output:
downsampled_spectra: the downscaled instrument spectra
downsampled_grid: the lower resolution spectral grid
"""
    print(input_spectral_grid.shape)
    print(input_spectra.shape)
    finterp = interp1d( input_spectral_grid ,input_spectra ,kind='linear')
    return finterp( output_spectral_grid)

def GenerateLegendrePolynomial(degree):
    coefficients = legendre(degree)
    polynomial = np.poly1d(coefficients)
    derivative = np.polyder(polynomial)
    return polynomial ,derivative 
    



x = {
    'VMR_H2O' : 0.01,
    'VMR_CH4' : 1700e-9,
    'VMR_CO2' : 400e-6
    }

def ForwardModel(state_vector ,measurement_object ,hitran_object):
    
    transmission, optical_depth = CalculateTransmission( state_vector ,measurement_object ,hitran_object)
    instrument_spectra = DownSampleInstrument( hitran_object.grid ,transmission ,measurement_object.spectral_grid )
    legendre_polynomial ,polynomial_derivative = GenerateLegendrePolynomial(15)

    # shift the spectral grid to be centered at 0, because polynomials are calculated over an even interval
    shifted_grid = measurement_object.spectral_grid - np.mean( measurement_object.spectral_grid )
    irr = transmission * legendre_polynomial( shifted_grid )
    return irr
    
