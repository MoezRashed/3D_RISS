import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils.plot import *
from utils.data_manipulation import *
from utils.equations import *

def process_imu(imu_data, imu_name, odo, gt, radius, we, dt):

    # Initialize state containers
    estimated_v_n   = []
    estimated_v_e   = []
    estimated_v_up  = []
    estimated_lat   = []
    estimated_long  = []
    estimated_alt   = []
    estimated_yaw   = []
    estimated_pitch = []
    estimated_roll  = []
    
    # Initial conditions from ground truth
    velocity  = [gt[7][0], gt[8][0], gt[6][0]]  # [ve, vn, vu]
    alt       = gt[0][0]
    lat       = gt[1][0]
    long      = gt[2][0]
    yaw       = gt[5][0]
    pitch     = gt[3][0]
    roll      = gt[4][0]
    v_od_prev = 0

    for i in range(len(imu_data[6])):
        # Sensor measurements
        fx, fy, fz = imu_data[0][i], imu_data[1][i], imu_data[2][i]
        wz         = imu_data[5][i]
        v_od       = odo[0][i]

        # ---- Bias Handling ----
        # (Add your bias calibration logic here if needed)
        # fx -= fx_bias_value, etc.

        # Mechanization equations
        acc_od = (v_od - v_od_prev) / dt
        roll   = roll_calc(fx, fz, v_od, wz)
        pitch  = pitch_calc(fx, fy, fz, acc_od)
        
        # Yaw update
        yaw_rate_value = yaw_rate(pitch, roll, wz, we, velocity[0], radius, lat, alt)
        yaw           += yaw_rate_value * dt
        yaw            = yaw % (2 * np.pi)

        # Velocity transformation
        velocity = transform_to_navigation_frame(yaw, pitch, v_od)
        
        # Position update
        delta_lat  = (velocity[1] / (radius + alt)) * dt
        delta_long = (velocity[0] / ((radius + alt) * np.cos(lat))) * dt
        delta_alt  = velocity[2] * dt
        
        # Update states
        lat      += delta_lat
        long     += delta_long
        alt      += delta_alt
        v_od_prev = v_od

        # Store all states
        estimated_v_n.append(velocity[1])
        estimated_v_e.append(velocity[0])
        estimated_v_up.append(velocity[2])
        estimated_lat.append(lat)
        estimated_long.append(long)
        estimated_alt.append(alt)
        estimated_yaw.append(yaw)
        estimated_pitch.append(pitch)
        estimated_roll.append(roll)

    return [
        estimated_v_n, estimated_v_e, estimated_v_up,
        estimated_lat, estimated_long, estimated_alt,
        estimated_yaw, estimated_pitch, estimated_roll
    ]