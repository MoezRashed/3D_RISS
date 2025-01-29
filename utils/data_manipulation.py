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

    for i in range(len(data[time_index])):
        if float(data[time_index][i]) >= start_time:
            start_idx = i
            break

    for i in range(len(data[time_index])-1, -1, -1):
        if float(data[time_index][i]) <= end_time:
            end_idx = i
            break

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
    #     processed_data = downsample(processed_data, 6 , 200)
    # else:
    #     processed_data = downsample(processed_data, 6 , 20)

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
    # downsample(processed_data, 9, 5)

    return processed_data

def downsample(data, time_index, factor):

    # Initialize empty lists for downsampled data
    downsampled_data = [[] for _ in range(len(data))]  
    time_data = data[time_index]  

    # Step 1: Find the first full timestep
    first_full_time_idx = None
    first_time_step = int(time_data[0])  # Convert first timestamp to an integer
    
    for i, t in enumerate(time_data):
        if int(t) > first_time_step:  # Find when a new full timestep begins
            first_full_time_idx = i
            break

    if first_full_time_idx is None:
        first_full_time_idx = 0  # If no new timestep is found, start from the beginning

    # Step 2: Start downsampling from this index using a fixed factor
    for i in range(first_full_time_idx, len(time_data), factor):
        next_i = min(i + factor, len(time_data))  # Avoid out-of-bounds indexing
        
        for j, component in enumerate(data):
            chunk = component[i:next_i]  # Extract the current chunk
            
            if j == time_index:
                downsampled_data[j].append(time_data[i])  # Take the first time value in chunk
            else:
                downsampled_data[j].append(np.mean(chunk))  # Compute mean for numerical data

    return downsampled_data

def downsample_by_mean(data, target_freq):
    
    downsampled_data = []
    
    for d in data:
        # Take the mean of every 'step' number of points
        downsampled_d = [np.mean(d[i:i + target_freq]) for i in range(0, len(d), target_freq)]
        downsampled_data.append(downsampled_d)
    
    time = downsampled_data[-1]  
    floored_time = np.array(np.floor(time).astype(int)) 
    if len(floored_time) > 1 and floored_time[-1] == floored_time[-2]: floored_time[-1] += 1
    downsampled_data[-1] = floored_time
    
    return downsampled_data