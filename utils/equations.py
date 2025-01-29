import math
import numpy as np

def pitch(fx , fy , fz, a_od):
    return math.atan2(-(fy-a_od), math.sqrt(fx**2 + fz**2))

def roll(fx, fz, v_od, azimuth):
    return math.atan2(-(fx + (v_od * azimuth)), fz)

# def yaw(p, r, ):

# def vehicle_stoppage():

# def online_caibration():




