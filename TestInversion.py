from ReadData import *
from ForwardModel import *
from Inversion import *


ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
co2_min_wavenumber, co2_max_wavenumber = 6180, 6250
hdo_min_wavenumber, hdo_max_wavenumber = 6310, 6380
file = 'testdata_2.h5'
data = CombData(file ,take_time_mean = True)
legendre_polynomial_degree = 5
vcd = 1


guess = {
    'VMR_H2O' : 0.02 * vcd,
    'VMR_CH4' : 3000e-9 * vcd,
    'VMR_CO2' : 400e-6 * vcd,
    'VMR_HDO' : 0.01 * vcd,
    'temperature' : 290,
    'shape_parameter' : 0.01 * np.ones( legendre_polynomial_degree +1 )
    }

truth = {
    'VMR_H2O' : 0.02 * vcd,
    'VMR_CH4' : 2000e-9 * vcd,
    'VMR_CO2' : 400e-6 * vcd,
    'VMR_HDO' : 0.001 * vcd,
    'temperature' : 295,
    'shape_parameter' : 0 * np.ones( legendre_polynomial_degree + 1 )
    }

truth = DictToArray(truth)
guess = DictToArray(guess)

measurement_index = 5
#ch4_data = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,data)
current_data = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,data)
#hdo_data = Measurement(measurement_index, hdo_min_wavenumber, hdo_max_wavenumber, data)


try:
    spectra = HitranSpectra( current_data)
except:
    GetCrossSections( data.min_wavenumber ,data.max_wavenumber)
    spectra = HitranSpectra( current_data)

control_flags ={
    'fit_windspeed' : False,
    'use_hitran08' : True
}

    
test_flags = {
    'fit_windspeed' : False,
    'use_hitran08' : False
    }


output = TestRun(guess, truth, data, control_flags, test_flags)
