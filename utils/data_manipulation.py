import logging
import numpy as np
import scipy.io as sio
from collections import Counter

# For Debugging 
def count_repeats(data):
    count = Counter(data)
    for key, value in count.items():
        print(f"Element {key} is repeated {value} times.")
    
def config():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def change_KVH_sign(fy, wy):
    return -fy, -wy

def trim_data(data, start_time, end_time, time_index):

    start_idx = None
    end_idx = None

    # Get the start_idx: Slice whatever is before it
    for i in range(len(data[time_index])):
        if float(data[time_index][i]) >= start_time:
            start_idx = i
            break
    # Get the end_idx: Slice whatever is after it
    for i in range(len(data[time_index])-1, -1, -1):
        if float(data[time_index][i]) <= end_time:
            end_idx = i
            break
    # Corner case
    if start_idx is None or end_idx is None:
        return [[] for _ in range(len(data))]

    trimmed_data = [d[start_idx:end_idx + 1] for d in data]

    return trimmed_data


def load_imu(path):

    # Shape for KVH: (500729, 1) sampled at 200hz
    # Shape for TPI: (50061, 1)  sampled at 20hz

    data               = path
    accelerometer_data = data['f']
    gyro_data          = data['w']

    # Extract Accelerometer data (x, y, z components)
    fx_data      = accelerometer_data['x'].flatten()
    fy_data      = accelerometer_data['y'].flatten()
    fz_data      = accelerometer_data['z'].flatten()

    # Extract Gyro data          (x, y, z components)
    wx_data      = gyro_data['x'].flatten()  
    wy_data      = gyro_data['y'].flatten() 
    wz_data      = gyro_data['z'].flatten()  

    # Extract time data
    time         = data['IMU_second'].flatten()

    logging.info(f"First timestamp at: {time[0]}, Last timestamp at: {time[-1]}")

    if 'NovAtel' in path: 
        change_KVH_sign(fy_data, fz_data)

    processed_data = [fx_data, fy_data, fz_data, wx_data, wy_data, wz_data, time]
    
    return processed_data

def load_odometer(data, time):

    # Shape for Odometer: (2503, 1) sampled at 1hz

    data = data['CarChip_Speed'].flatten()
    time = time['odo_second'].flatten()

    logging.info(f"First timestamp at: {time[0]}, Last timestamp at: {time[-1]}")

    processed_data = [data, time]
   
    return processed_data

def load_gt(path):

    # Shape for Ground truth: (12517, 1) sampled at 5hz

    data      = path
    time      = data['INS_second'].flatten()

    altitude  = data['INS_Alt'].flatten()
    latitude  = np.deg2rad(data['INS_Lat'].flatten()  )
    longtiude = np.deg2rad(data['INS_Long'].flatten() )
   
    pitch     = np.deg2rad(data['INS_Pitch'].flatten())
    roll      = np.deg2rad(data['INS_Roll'].flatten() )
    azimuth   = np.deg2rad(data['INS_Azi'].flatten()  )
    
    v_up      = data['INS_vu'].flatten()
    v_e       = data['INS_ve'].flatten()
    v_n       = data['INS_vn'].flatten()

    logging.info(f"First timestamp at: {time[0]}, Last timestamp at: {time[-1]}")

    processed_data = [altitude, latitude, longtiude, pitch, roll, azimuth , v_up, v_e, v_n, time]

    return processed_data

def downsample_by_mean(data, target_freq):
    
    downsampled_data = []
    
    # Take the mean of every 'step' number of points [Target_Freq]
    for d in data:
        downsampled_d = [np.mean(d[i:i + target_freq]) for i in range(0, len(d), target_freq)]
        downsampled_data.append(downsampled_d)
    
    time = downsampled_data[-1]

    floored_time = np.array(np.floor(time).astype(int)) 

    # To mitigate if the last time does not have a full chunk
    if len(floored_time) > 1 and floored_time[-1] == floored_time[-2]: floored_time[-1] += 1

    downsampled_data[-1] = floored_time

    logging.info(f"First timestamp at: {downsampled_data[-1][0]}, Last timestamp at: {downsampled_data[-1][-1]}")
    
    return downsampled_data