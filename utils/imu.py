import numpy                 as np
import matplotlib.pyplot     as plt
from scipy.io                import loadmat
from utils.plot              import *
from utils.data_manipulation import *
from utils.equations         import *

def process_imu(imu_data, imu_name, odo, gt, radius, we, dt, pos, vel):

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
    estimated_a_od  = []

    
    # Initial conditions from ground truth
    velocity        = [gt[7][0], gt[8][0], gt[6][0]]  # [ve, vn, vu]
    alt             = gt[0][0]
    lat             = gt[1][0]
    long            = gt[2][0]
    yaw             = gt[5][0]
    pitch           = gt[3][0]
    roll            = gt[4][0]
    v_od_prev       = 0

    # Reference Position Data
    alt_ref         = pos[0]
    lat_ref         = pos[1]
    long_ref        = pos[2]
    alt_ref_std     = pos[3]
    lat_ref_std     = pos[4]
    long_ref_std    = pos[5]

    # Reference Velocity Data
    ve_ref          = vel[0]
    vn_ref          = vel[1]
    vu_ref          = vel[2]

    # Initalize bias lists
    fx_bias         = []
    fy_bias         = []
    fz_bias         = []
    wz_bias         = []
    
    g               = 9.80577
    acc_od          = 0.0
    fx_bias_value   = 0.0
    fy_bias_value   = 0.0
    fz_bias_value   = 0.0
    wz_bias_value   = 0.0
    bias_calibrated        = False 
    stationary_counter     = 0
    non_stationary_counter = 0
    p_prev                 = 0

    for i in range(len(imu_data[6])):

        # Sensor measurements
        fx, fy, fz = imu_data[0][i], imu_data[1][i], imu_data[2][i]
        wz         = imu_data[5][i]
        v_od       = odo[0][i]

        # KVH
        bwz_corr_time = 3 * 3600
        bwz_stdv      = 0.01* np.pi/180

        a_od_corr_time = 0.1 * 3600
        a_od_stdv      = 10

        # ---- Start of TPI Optimization ----
        if imu_name == "TPI":

            bwz_corr_time = 4 * 3600
            bwz_stdv      = 0.01* np.pi/180

            a_od_corr_time = 0.1 * 3600
            a_od_stdv      = 10

            if v_od <= 0.0:
                non_stationary_counter = 0
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
                    fx_bias_value = np.mean(fx_bias)
                    fy_bias_value = np.mean(fy_bias)
                    fz_bias_value = np.mean(fz_bias)
                    wz_bias_value = np.mean(wz_bias)

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

        # # ----- End of TPI Optimization ----- 

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

        # First component --> VE
        # Second component--> VN
        # Third component --> VU

        velocity = transform_to_navigation_frame(yaw, pitch, v_od)
        
        # Position update
        delta_lat  = (velocity[1] / (radius + alt)) * dt
        delta_long = (velocity[0] / ((radius + alt) * np.cos(lat))) * dt
        delta_alt  = velocity[2] * dt
        
        # Update states before kalman filter 
        lat      += delta_lat
        long     += delta_long
        alt      += delta_alt
        v_od_prev = v_od
        p_prev    = pitch

        # I need the first 6 values for the Z calculation from IMU_Calculations
        IMU_calculations = [lat, long, alt, velocity[0], velocity[1], velocity[2], yaw, wz, acc_od]
        # imported the gps_pos & gps_vel data before the loop, they include _ref in variable name, pass it to process_imu, and use it with (i) per timestep.
        
        # Kalman filter comes here 

        # Update states again after Kalman Filter to implement loosely coupled

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
        estimated_a_od.append(acc_od)

    return [
        estimated_v_n, estimated_v_e, estimated_v_up,
        estimated_lat, estimated_long, estimated_alt,
        estimated_yaw, estimated_pitch, estimated_roll,
        estimated_a_od
    ]