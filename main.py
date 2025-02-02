import numpy as np
import matplotlib.pyplot as plt
from scipy.io                import loadmat
from utils.plot              import *
from utils.data_manipulation import *
from utils.equations         import *
from utils.imu               import *


def main():

    # Load the configuration structure
    config()

    # Constants
    we         = 7.292115e-5  # Earth's rotation rate (rad/s)
    radius     = 6378137.0    # Earth's radius (m)
    dt         = 1            # Timestep

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

    # Process IMUs and get full state lists
    kvh_states = process_imu(KVH, "KVH", odo, gt, radius, we, dt)
    tpi_states = process_imu(TPI, "TPI", odo, gt, radius, we, dt)

    # Calculate errors for KVH
    delta_pn_kvh, delta_pe_kvh, delta_ph_kvh, delta_pu_kvh, delta_3d_kvh = delta_position_errors(
        gt, kvh_states, radius, radius
    )
    # Calculate errors for TPI
    delta_pn_tpi, delta_pe_tpi, delta_ph_tpi, delta_pu_tpi, delta_3d_tpi = delta_position_errors(
        gt, tpi_states, radius, radius
    )

    # Log KVH errors
    logging.info("\n=== KVH Performance ===")
    logging.info(f"RMSE North: {rmse(delta_pn_kvh):.3f} m")
    logging.info(f"RMSE East: {rmse(delta_pe_kvh):.3f} m")
    logging.info(f"RMSE Up: {rmse(delta_pu_kvh):.3f} m")
    logging.info(f"RMSE Horizontal: {rmse(delta_ph_kvh):.3f} m")
    logging.info(f"Max Error Position North: {max_error(delta_pn_kvh):.3f} m")
    logging.info(f"Max Error Position East: {max_error(delta_pe_kvh):.3f} m")
    logging.info(f"Max Error Position Up: {max_error(delta_pu_kvh):.3f} m")
    logging.info(f"3D RMSE: {rmse(delta_3d_kvh):.3f} m")

    # Log TPI errors
    logging.info("\n=== TPI Performance ===")
    logging.info(f"RMSE North: {rmse(delta_pn_tpi):.3f} m")
    logging.info(f"RMSE East: {rmse(delta_pe_tpi):.3f} m")
    logging.info(f"RMSE Up: {rmse(delta_pu_tpi):.3f} m")
    logging.info(f"RMSE Horizontal: {rmse(delta_ph_tpi):.3f} m")
    logging.info(f"Max Error Position North: {max_error(delta_pn_tpi):.3f} m")
    logging.info(f"Max Error Position East: {max_error(delta_pe_tpi):.3f} m")
    logging.info(f"Max Error Position Up: {max_error(delta_pu_tpi):.3f} m")
    logging.info(f"3D RMSE: {rmse(delta_3d_tpi):.3f} m")

    # Plot states for KVH
    plot_states(gt, kvh_states, "KVH")
    # Plot states for TPI
    plot_states(gt, tpi_states, "TPI")


if __name__ == "__main__":
    main()    