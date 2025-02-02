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

    # Initalize bias lists
    fx_bias          = []
    fy_bias          = []
    fz_bias          = []
    wz_bias          = []
    
    g                = 9.80577
    acc_od           = 0.0
    fx_bias_value    = 0.0
    fy_bias_value    = 0.0
    fz_bias_value    = 0.0
    wz_bias_value    = 0.0
    bias_calibrated  = False 
    stationary_counter = 0
    non_stationary_counter = 0

    for i in range(len(imu_data[6])):
        # Sensor measurements
        fx, fy, fz = imu_data[0][i], imu_data[1][i], imu_data[2][i]
        wz         = imu_data[5][i]
        v_od       = odo[0][i]

        # ---- Start of TPI Optimization ----
        if imu_name == "TPI":

            if v_od == 0:
                
                # Collect sensor data for bias calibration
                fx_current_value = -(np.cos(pitch) * np.sin(roll) * g) - fx
                fy_current_value = (np.sin(pitch) * g) - fy
                fz_current_value = (np.cos(pitch) * np.cos(roll) * g) -fz
                wz_current_value = (we * np.sin(lat)) - wz

                
                fx_bias.append(fx_current_value)
                fy_bias.append(fx_current_value)
                fz_bias.append(fx_current_value)
                wz_bias.append(wz_current_value)
                stationary_counter += 1

                # Calibrate biases after 10 consecutive stationary timesteps
                if stationary_counter >= 1 and not bias_calibrated:
                    fx_bias_value = np.median(fx_bias)
                    fy_bias_value = np.median(fy_bias)
                    fz_bias_value = np.median(fz_bias)
                    wz_bias_value = np.median(wz_bias)

                    bias_calibrated = True

                # Freeze mechanization during stoppage
                estimated_lat.append(lat)
                estimated_long.append(long)
                estimated_alt.append(alt)
                estimated_yaw.append(yaw)
                estimated_pitch.append(pitch)
                estimated_roll.append(roll)
                estimated_v_n.append(velocity[1])
                estimated_v_e.append(velocity[0])
                estimated_v_up.append(velocity[2])
                continue
            else:
                # Reset counter and calibration flag when vehicle moves
                stationary_counter      = 0
                bias_calibrated         = False
                non_stationary_counter += 1
                # Clearing the lists
                fx_bias.clear()
                fy_bias.clear()
                fz_bias.clear()
                wz_bias.clear()

            if non_stationary_counter <= 150:
                # Update fx, fy, fz, wz values
                fx           = fx - (fx_bias_value)
                fy           = fy - (fy_bias_value)
                fz           = fz - (fz_bias_value)
                wz           = wz - (wz_bias_value)
        # ----- End of TPI Optimization ----- 

        # Mechanization equations
        acc_od = (v_od - v_od_prev) / dt
        # # Testing
        # acc_od = 0

        # Calculate Roll & Pitch
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