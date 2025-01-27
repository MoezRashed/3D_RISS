import scipy.io as sio
import logging

def change_KVH_sign(fy, wy):
    return -fy, -wy

def load_imu(path):

    # Shape for KVH: (500729, 1) sampled at 200hz
    # Shape for TPI: (50061, 1)  sampled at 20hz

    data         = path
    # Extract Accelerometer data (x, y, z components)
    accelerometer_data = data['f']
    fx_data      = accelerometer_data['x']  # Shape: (50061, 1)
    fy_data      = accelerometer_data['y']  # Shape: (50061, 1)
    fz_data      = accelerometer_data['z']  # Shape: (50061, 1)
    # Extract Gyro data (x, y, z components)
    gyro_data    = data['w']
    wx_data      = gyro_data['x']  
    wy_data      = gyro_data['y']  
    wz_data      = gyro_data['z']  
    # Convert to 1D arrays (optional, for easier analysis)
    time         = data['IMU_second'].flatten()  # Shape: (500729,1)
    fx_data      = fx_data.flatten()
    fy_data      = fy_data.flatten()
    fz_data      = fz_data.flatten()
    wx_data      = wx_data.flatten()
    wy_data      = wy_data.flatten()
    wz_data      = wz_data.flatten()

    logging.info("IMU Info:", data['IMU_info'][0])
    # if data.get('denoising_info') == None: logging.info("Denoising Info:", data['denoising_info'][0])

    return fx_data, fy_data, fz_data, wx_data, wy_data, wz_data, time 

# def downsample_data(x_data, y_data, z_data, time, factor):
