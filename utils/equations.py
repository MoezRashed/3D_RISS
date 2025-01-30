import math
import numpy as np

def pitch_calc(fx , fy , fz, a_od):

    return math.atan2(-(fy - a_od), math.sqrt(fx**2 + fz**2))

def roll_calc(fx, fz, v_od, azimuth):

    return math.atan2(-(fx + (v_od * azimuth)), fz)

# rate of change in yaw
def yaw_rate(pitch, roll, wz, we, ve, r, lat, alt):

    yaw = (np.cos(pitch) * np.cos(roll) * wz) - (we * np.sin(lat)) -((ve * np.tan(lat)) / (r + alt))

    return yaw

def transform_to_navigation_frame(yaw, pitch, v_od):
    
    V_e = -np.sin(yaw) * np.cos(pitch) * v_od
    V_n =  np.cos(yaw) * np.cos(pitch) * v_od
    V_u =  np.sin(pitch) * v_od

    return [V_e, V_n, V_u]

# def vehicle_stoppage():

# def online_caibration():




