import math
import numpy as np

def pitch(fx , fy , fz):
    return math.atan2(-fy, math.sqrt(fx**2 + fz**2))

def roll(fx, fz):
    return math.atan2(-fx, fz)





# def vehicle_stoppage():


