import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils.plot import *
from utils.data_manipulation import *
from utils.equations import roll, pitch


def main():

    config()

    # Load IMU data from file
    data_KVH = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/NovAtel/1.RAWIMU.mat'                          , simplify_cells=True) #High-end unit
    data_TPI = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/TPI/1.TPI_data_denoised_LOD6_interpolated2.mat', simplify_cells=True) #Low-end unit

    # Load the ground truth from file
    data_GT  = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/NovAtel/3.INSPVA_Reference.mat'                , simplify_cells=True)

    # Load the Odometer data from file
    data_OD  = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/OBDII_data/CarChip_Speed.mat'                  , simplify_cells=True)
    time_OD  = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/OBDII_data/odo_second.mat'                     , simplify_cells=True)

    # Load the sensor data from the IMU units
    TPI      = load_imu(data_TPI)
    KVH      = load_imu(data_KVH)
    # Load the Odometer data
    odo      = load_odometer(data_OD, time_OD)
    # Load the ground truth data
    gt       = load_gt(data_GT)

    # Calculating start time and end time
    start_time = max(TPI[6][0], KVH[6][0], gt[9][0], odo[1][0])
    end_time   = min(TPI[6][-1], KVH[6][-1], gt[9][-1], odo[1][-1])
    print(len(TPI[6]))
    # Trimming data
    TPI      = trim_data(TPI, start_time, end_time, 6)
    KVH      = trim_data(KVH, start_time, end_time, 6)
    odo      = trim_data(odo, start_time, end_time, 1)
    gt       = trim_data(gt , start_time, end_time, 9)
   
    print(len(TPI[6]))
    # Down sample data to 1 Hz
    TPI      = downsample_by_mean(TPI, 20)
    KVH      = downsample_by_mean(KVH, 200)
    gt       = downsample_by_mean(gt , 5)

    print((KVH[6]))
    print((TPI[6]))
    print((odo[1]))
    print((gt[9]))

    # For loop over the timesteps where you will calculate the yaw and get the velocites 
    
    

    # Calculate the pitch and roll 

    # # Plot the sensor data of the TPI IMU unit
    # plot_sensor_data(fx_TPI, fy_TPI, fz_TPI, timestamp_TPI)
    # plot_sensor_data(wx_TPI, wy_TPI, wz_TPI, timestamp_TPI)

    # # Plot the sensor data of the KVH IMU unit
    # plot_sensor_data(fx_KVH, fy_KVH, fz_KVH, timestamp_KVH)
    # plot_sensor_data(wx_KVH, wy_KVH, wz_KVH, timestamp_KVH)

if __name__ == "__main__":
    main()    