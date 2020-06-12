from ReadData import *
from ForwardModel import *
from Inversion import *


ch4_min_wavenumber, ch4_max_wavenumber = 6055, 6120
co2_min_wavenumber, co2_max_wavenumber = 6180, 6250
hdo_min_wavenumber, hdo_max_wavenumber = 6310, 6380
file = 'testdata_2.h5'
data = CombData(file ,take_time_mean = True)
legendre_polynomial_degree = 40
vcd = 1


guess = {
    'VMR_H2O' : 0.02 * vcd,
    'VMR_CH4' : 3000e-9 * vcd,
    'VMR_CO2' : 400e-6 * vcd,
    'VMR_HDO' : 0.01 * vcd,
    'temperature' : 290,
    'shape_parameter' : np.ones( legendre_polynomial_degree +1 )
    }

truth = {
    'VMR_H2O' : 0.02 * vcd,
    'VMR_CH4' : 2000e-9 * vcd,
    'VMR_CO2' : 400e-6 * vcd,
    'VMR_HDO' : 0.001 * vcd,
    'temperature' : 295,
    'shape_parameter' : np.ones( legendre_polynomial_degree + 1 )*0.1
    }

truth = DictToArray(truth)


measurement_index = 5
#ch4_data = Measurement( measurement_index ,ch4_min_wavenumber ,ch4_max_wavenumber ,data)
current_data = Measurement( measurement_index ,co2_min_wavenumber ,co2_max_wavenumber ,data)
#hdo_data = Measurement(measurement_index, hdo_min_wavenumber, hdo_max_wavenumber, data)


try:
    spectra = HitranSpectra( current_data)
except:
    GetCrossSections( data.min_wavenumber ,data.max_wavenumber)
    spectra = HitranSpectra( current_data)
#ch4_spectra = HitranSpectra(ch4_data)
#hdo_spectra = HitranSpectra( hdo_data)




#co2_ans, co2_model = LinearInversion( truth ,current_data ,spectra)
#ch4_ans, ch4_model = LinearInversion( truth ,ch4_data ,ch4_spectra)
#hdo_ans, hdo_model = LinearInversion(truth, hdo_data, hdo_spectra)


#np.savez('fitting_data' ,co2_obs = current_data.FC, ch4_obs = ch4_data.FC, co2_grid = current_data.spectral_grid, ch4_grid = ch4_data.spectral_grid, co2_model = co2_model, ch4_model = ch4_model, #ch4_ans = ch4_ans, co2_ans = co2_ans, hdo_ans = hdo_ans, hdo_obs = hdo_data.FC, hdo_grid = hdo_data.spectral_grid, hdo_model = hdo_model)


#H2O, CH4, CO2 = InvertAllData( truth ,data ,spectra)
#np.savez('CH4_linear_inversion', CO2 = CO2, H2O = H2O, CH4 = CH4)
#global my_func
linear_func = MakeInversionFunction( data, truth ,linear = True , FM = True)
nonlinear_func = MakeInversionFunction( data, truth ,linear = False, FM = True)

ch4_l, co2_l, hdo_l = linear_func( measurement_index)
np.savez('mixed_fit.npz', ch4_m = ch4_l, co2_m = co2_l, hdo_m = hdo_l)
ch4_n, co2_n, hdo_n = nonlinear_func(measurement_index)
np.savez('nonlinear_fit.npz', co2_n = co2_n, ch4_n = ch4_n, hdo_n = hdo_n, co2_l = co2_l, ch4_l = ch4_l, hdo_l = hdo_l)
         

