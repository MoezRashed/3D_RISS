import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils.plot import *
from utils.data_manipulation import *
from utils.equations import *


def main():

    # Load the configuration structure
    config()

    # Constants
    stationary_counter = 0            # count timesteps where v_od is 0
    we                 = 7.292115e-5  # Earth's rotation rate (rad/s)
    radius             = 6378137.0    # Earth's radius (m)
    dt                 = 1            # Timestep
    

    # Load IMU data from file
    data_KVH   = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/NovAtel/1.RAWIMU.mat'                          , simplify_cells=True) #High-end unit
    data_TPI   = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/TPI/1.TPI_data_denoised_LOD6_interpolated2.mat', simplify_cells=True) #Low-end unit

    # Load the ground truth from file
    data_GT    = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/NovAtel/3.INSPVA_Reference.mat'                , simplify_cells=True)

    # Load the Odometer data from file
    data_OD    = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/OBDII_data/CarChip_Speed.mat'                  , simplify_cells=True)
    time_OD    = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/OBDII_data/odo_second.mat'                     , simplify_cells=True)

    # Load the ground truth data
    gt         = load_gt(data_GT)

    # Load the sensor data from the IMU units
    TPI        = load_imu(data_TPI)
    KVH        = load_imu(data_KVH)

    # Load the Odometer data
    odo        = load_odometer(data_OD, time_OD)

    # Calculating start time and end time
    start_time = max(TPI[6][0], KVH[6][0], gt[9][0], odo[1][0])
    end_time   = min(TPI[6][-1], KVH[6][-1], gt[9][-1], odo[1][-1])

    # Trimming data
    TPI        = trim_data(TPI, start_time, end_time, 6)
    KVH        = trim_data(KVH, start_time, end_time, 6)
    odo        = trim_data(odo, start_time, end_time, 1)
    gt         = trim_data(gt , start_time, end_time, 9)
   
    # Down sample data to 1 Hz [to match odo]
    gt         = downsample_by_mean(gt , 5)
    TPI        = downsample_by_mean(TPI, 20)
    KVH        = downsample_by_mean(KVH, 200)

    # Load the First timestep of gt as ground truth
    alt_gt, lat_gt, long_gt, ve_gt, yaw_gt, pitch_gt, roll_gt, vn_gt, vu_gt = gt[0][0], gt[1][0], gt[2][0], gt[7][0], gt[5][0], gt[3][0], gt[4][0], gt[8][0], gt[6][0]
    # Initialize lists to store estimated states
    estimated_lat    = []
    estimated_long   = []
    estimated_alt    = []
    estimated_yaw    = []
    estimated_pitch  = []
    estimated_roll   = []
    estimated_v_n    = []
    estimated_v_e    = []
    estimated_v_up   = []
    velocity         = [ve_gt,vn_gt,vu_gt]

    # KVH // TPI loop
    for i in range(len(KVH[6])):

        if i == 0:
            alt       = alt_gt
            lat       = lat_gt
            long      = long_gt
            ve        = ve_gt
            yaw       = yaw_gt
            pitch     = pitch_gt
            roll      = roll_gt
            v_od_prev = 0
       
        # Load instances into variables
        fx , fy , fz  = KVH[0][i], KVH[1][i], KVH[2][i]
        wz            = KVH[5][i]
        v_od          = odo[0][i]

        # if v_od == 0:

        #     stationary_counter += 1
        #     if stationary_counter >= 1:  # Changing the timesteps affects it alot
        #         estimated_lat.append(lat)
        #         estimated_long.append(long)
        #         estimated_alt.append(alt)
        #         estimated_yaw.append(yaw)
        #         estimated_roll.append(roll)
        #         estimated_pitch.append(pitch)
        #         estimated_v_n.append(velocity[1])
        #         estimated_v_e.append(velocity[0])
        #         estimated_v_up.append(velocity[2])
        #         continue
        
        # Resetting the stationary_counter
        stationary_counter = 0

        # Calculate acceleration odometer
        acc_od        = (v_od - v_od_prev) / dt
        # Calculate pitch & roll 
        roll          = roll_calc(fx, fz, v_od, wz)
        pitch         = pitch_calc(fx, fy, fz, acc_od)
        # Calculate Rate of change of yaw
        yaw_rate_value= yaw_rate(pitch, roll, wz, we, velocity[0], radius, lat, alt)
        yaw          += yaw_rate_value * dt
        yaw           = yaw + (yaw < 0) * 2 * np.pi + (yaw >= 2 * np.pi) * (-2 * np.pi)
        
        # Transform from body frame to local level frame
        velocity      = transform_to_navigation_frame(yaw, pitch, v_od)
        
        # Update latitude, longitude, and altitude
        delta_lat     = (velocity[1] / (radius  + alt)) * dt  
        delta_long    = (velocity[0] / ((radius + alt)  * np.cos(lat))) * dt
        delta_alt     = (velocity[2]) * dt  
        # Update lat, long & alt
        lat  += delta_lat  
        long += delta_long  
        alt  += delta_alt 
        # Update V_od
        v_od_prev = v_od

        # Populating the lists
        estimated_lat.append(lat)
        estimated_long.append(long)
        estimated_alt.append(alt)
        estimated_yaw.append(yaw)
        estimated_roll.append(roll)
        estimated_pitch.append(pitch)
        estimated_v_n.append(velocity[1])
        estimated_v_e.append(velocity[0])
        estimated_v_up.append(velocity[2])

        logging.info(f"  Timestep {i}:")
        logging.info(f"  Roll: {np.degrees(roll):.2f}°, Pitch: {np.degrees(pitch):.2f}°, Yaw: {np.degrees(yaw):.2f}°")
        logging.info(f"  Velocity (E, N, U)                : {velocity}")
        logging.info(f"  Acceleration                      : {acc_od:.2f} m/s²")
        logging.info(f"  Position Estimate (Lat, Long, Alt): ({np.degrees(lat):.6f}°, {np.degrees(long):.6f}°, {alt:.2f} m)")
        logging.info(f"  Position Actual   (Lat, Long, Alt): ({np.degrees(gt[1][i]):.6f}°, {np.degrees(gt[2][i]):.6f}°, {gt[0][i]:.2f} m)")


    # Convert ground truth and estimated data to degrees for plotting
    estimated_lat_deg    = np.degrees(np.array(estimated_lat))
    estimated_long_deg   = np.degrees(np.array(estimated_long))
    groundtruth_lat_deg  = np.degrees(gt[1])
    groundtruth_long_deg = np.degrees(gt[2])

    # Append states to a list called estimated_states
    estimated_states     = [estimated_v_n, estimated_v_e, estimated_v_up, estimated_lat, estimated_long, estimated_alt, estimated_yaw, estimated_pitch, estimated_roll]

    delta_pn, delta_pe, delta_ph = delta_position_errors(gt, estimated_states, radius, radius)

    logging.info(f" RMSE Position North: {rmse(delta_pn)}, RMSE Position East: {rmse(delta_pe)}, RMSE Position Horizontal: {rmse(delta_ph)},")
    logging.info(f" Max Error Position North: {max_error(delta_pn)}, Max Error Position East: {max_error(delta_pe)}, Max Error Position Horizontal: {max_error(delta_ph)},")
    logging.info(f" Total Horizontal Distance: {total_horizontal_distane(gt, radius, radius)}")

    plt.figure(figsize=(12, 8))

    # Plot ground truth vs. estimated trajectory
    plt.plot(groundtruth_long_deg, groundtruth_lat_deg, 
            label='Ground Truth', color='green', linestyle='--', linewidth=2)
    plt.plot(estimated_long_deg, estimated_lat_deg, 
            label='Estimated', color='red', linewidth=1.5)

    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Ground Truth vs. Estimated Position')
    plt.legend()
    plt.grid(True)

    plt.savefig('trajectory_comparison_HighEnd.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()    