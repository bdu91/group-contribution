from decimal import Decimal
import numpy as np
import pandas as pd

def e_water(T):
    """
    Calculate water dielectric constant as a function of temperature
    :param T: temperature in Kelvin
    :return: water dielectric constant
    """
    T = float(T) - 273.15
    e = Decimal(-5.6537368714291)/(10**7)*(Decimal(T)**3)\
    + Decimal(0.000758946363457904)*(Decimal(T)**2)\
    + Decimal(-0.393783271271903)*Decimal(T)\
    + Decimal(87.8281016042781)
    return float(e)

def X_Born(T):
    """
    Calculate Born function X as a function of temperature
    :param T: temperature in Kelvin
    :return: Born function X
    """
    T = float(T) - 273.15
    X =Decimal(-1.49310076942083)/(10**12)*(Decimal(T)**6)\
    + Decimal(7.92173114230276)/(10**10)*(Decimal(T)**5)\
    + Decimal(-1.74864248279802)/(10**7)*(Decimal(T)**4)\
    + Decimal(0.0000201001581377973)*(Decimal(T)**3)\
    + Decimal(-0.00134688577363809)*(Decimal(T)**2)\
    + Decimal(0.0386506331596876)*Decimal(T)\
    + Decimal(-3.46586139615551)
    return float(X/Decimal(10000000.0))

def Y_Born(T):
    """
    Calculate Born function Y as a function of temperature
    :param T: temperature in Kelvin
    :return: Born function Y
    """
    T = float(T) - 273.15
    Y = Decimal(-2.47803408177797)/(10**11)*(Decimal(T)**5)\
    + Decimal(9.89176185473189)/(10**9)*(Decimal(T)**4)\
    + Decimal(-1.84384507398707)/(10**6)*(Decimal(T)**3)\
    + Decimal(0.000106928450229463)*(Decimal(T)**2)\
    + Decimal(-0.0335095833911023)*Decimal(T)\
    + Decimal(-5.01257018716577)
    return float(Y/Decimal(100000.0))

#import data of metal ions needed for temperature transform
metal_thermo_data = pd.read_csv('data/metal_thermo_data.csv')

def metal_T_transform_dG_f(metal_type, T):
    """
    Calculate the difference in formation energy between target temperature T and reference temperature T_r, note c2 and w have the scale of 10^-4 and 10^-5

    The specifc formulation and data can be found in: Shock, Everett L., and Harold C. Helgeson. "Calculation of the thermodynamic and transport properties
    of aqueous species at high pressures and temperatures: Correlation algorithms for ionic species and equation of state predictions to 5 kb and 1000 C."
    Geochimica et Cosmochimica Acta 52.8 (1988): 2009-2036.

    :param metal_type: type of metal ion
    :param T: Temperature in K
    :return: dG_f_T - dG_f_T_r
    """
    T_r = 298.15
    Theta = 228
    
    cur_metal_thermo_data = metal_thermo_data[metal_thermo_data['metal_type'] == metal_type]

    c1 = float(cur_metal_thermo_data['c1'].tolist()[0])
    c2 = float(cur_metal_thermo_data['c2'].tolist()[0])
    w = float(cur_metal_thermo_data['w'].tolist()[0])
    dS_f = float(cur_metal_thermo_data['dS'].tolist()[0])
    S = float(cur_metal_thermo_data['S'].tolist()[0])

    Entropy_part = -dS_f * (T - T_r)
    c1_part = -c1 * (T * np.log(T/T_r) - T + T_r)
    c2_part = -c2 * np.power(10.,4) * ((1/(T - Theta) - 1/(T_r - Theta))*(Theta - T)/Theta - T/np.square(Theta) * np.log(T_r * (T - Theta)/T/(T_r - Theta)))
    w_part = w * np.power(10.,5) * (Y_Born(T_r) * (T - T_r) + 1/e_water(T) - 1/e_water(T_r))
    delta_dG = Entropy_part + c1_part + c2_part + w_part
    return delta_dG

def metal_T_transform_dH_f(metal_type, T):
    """
    Calculate the difference in formation enthalpy between target temperature T and reference temperature T_r, note c2 and w have the scale of 10^-4 and 10^-5

    The specifc formulation and data can be found in: Shock, Everett L., and Harold C. Helgeson. "Calculation of the thermodynamic and transport properties
    of aqueous species at high pressures and temperatures: Correlation algorithms for ionic species and equation of state predictions to 5 kb and 1000 C."
    Geochimica et Cosmochimica Acta 52.8 (1988): 2009-2036.

    :param metal_type: type of metal ion
    :param T: Temperature in K
    :return: dH_f_T - dH_f_T_r
    """
    T_r = 298.15
    Theta = 228

    cur_metal_thermo_data = metal_thermo_data[metal_thermo_data['metal_type'] == metal_type]

    c1 = float(cur_metal_thermo_data['c1'].tolist()[0])
    c2 = float(cur_metal_thermo_data['c2'].tolist()[0])
    w = float(cur_metal_thermo_data['w'].tolist()[0])

    c1_part = c1*(T-T_r)
    c2_part = -c2 * np.power(10.,4) * (1/(T - Theta) - 1/(T_r - Theta))
    w_part = w * np.power(10.,5) * (T*Y_Born(T) - T_r*Y_Born(T_r) + 1/e_water(T) - 1/e_water(T_r))
    delta_dH = c1_part + c2_part + w_part
    
    return delta_dH