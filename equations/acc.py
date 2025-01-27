import math
import numpy as np

def acc_to_angle(fx , fy , fz):

    pitch = math.atan2(-fy, math.sqrt(fx**2 + fz**2))
    roll  = math.atan2(-fx, fz)

    return pitch, roll






# def vehicle_stoppage():


