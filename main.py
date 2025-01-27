import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils.plot import *
from utils.dataset import *
from equations.acc import acc_to_angle


def main():
    # Load data from the mat files
    data_KVH = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/NovAtel/1.RAWIMU.mat', simplify_cells=True) #High-end unit
    data_TPI = loadmat('/Users/moezrashed/Documents/Programming/Python/Project1/TPI/1.TPI_data_denoised_LOD6_interpolated2.mat', simplify_cells=True) #Low-end unit
    # Load the sensor data from the IMU units
    fx_TPI, fy_TPI, fz_TPI, wx_TPI, wy_TPI, wz_TPI, timestamp_TPI = load_imu(data_TPI)
    fx_KVH, fy_KVH, fz_KVH, wx_KVH, wy_KVH, wz_KVH, timestamp_KVH = load_imu(data_KVH)
    #plot the sensor data of the TPI IMU unit
    plot_sensor_data(fx_TPI, fy_TPI, fz_TPI, timestamp_TPI)
    plot_sensor_data(wx_TPI, wy_TPI, wz_TPI, timestamp_TPI)

    plot_sensor_data(fx_KVH, fy_KVH, fz_KVH, timestamp_KVH)
    plot_sensor_data(wx_KVH, wy_KVH, wz_KVH, timestamp_TPI)

if __name__ == "__main__":
    main()    