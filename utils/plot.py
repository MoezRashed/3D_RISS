import matplotlib.pyplot as plt 
import numpy as np

def plot_imu_data(x_data, y_data, z_data, time):
    plt.figure(figsize=(12, 6))
    plt.plot(time, x_data, label='X-axis')
    plt.plot(time, y_data, label='Y-axis')
    plt.plot(time, z_data, label='Z-axis')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration [m/s]')
    plt.title('IMU Sensor Data Over Time')
    plt.legend()
    plt.show()

def plot_gyro_data(x_data, y_data, z_data, time):
    plt.figure(figsize=(12, 6))
    plt.plot(time, np.rad2deg(x_data), label='X-axis')
    plt.plot(time, np.rad2deg(y_data), label='Y-axis')
    plt.plot(time, np.rad2deg(z_data), label='Z-axis')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Degrees')
    plt.title('IMU Sensor Data Over Time')
    plt.legend()
    plt.show()

def plot_states(gt, estimated_states, imu_name):
    """
    Plot ground truth vs. estimated states for a given IMU.
    Each velocity and orientation component is plotted in a separate subplot.
    
    Args:
        gt (list): Ground truth data.
        estimated_states (list): Estimated states from the IMU.
        imu_name (str): Name of the IMU (e.g., "KVH" or "TPI").
    """
    # Extract states
    v_n_est, v_e_est, v_up_est   = estimated_states[0], estimated_states[1], estimated_states[2]
    lat_est, lon_est, alt_est    = estimated_states[3], estimated_states[4], estimated_states[5]
    yaw_est, pitch_est, roll_est = estimated_states[6], estimated_states[7], estimated_states[8]

    # Time vector (based on the number of samples)
    time = np.arange(len(lat_est))

    # Create a figure with a 5x2 grid (10 subplots)
    plt.figure(figsize=(15, 20))

    # --------------------------
    # Position States
    # --------------------------

    # Latitude
    plt.subplot(5, 2, 1)
    plt.plot(time, np.degrees(gt[1][:len(lat_est)]), 'g--', label='Ground Truth')
    plt.plot(time, np.degrees(lat_est), 'r-', label=f'{imu_name} Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Latitude (deg)')
    plt.title('Latitude vs. Time')
    plt.legend()
    plt.grid(True)

    # Longitude
    plt.subplot(5, 2, 2)
    plt.plot(time, np.degrees(gt[2][:len(lon_est)]), 'g--', label='Ground Truth')
    plt.plot(time, np.degrees(lon_est), 'r-', label=f'{imu_name} Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Longitude (deg)')
    plt.title('Longitude vs. Time')
    plt.legend()
    plt.grid(True)

    # Altitude
    plt.subplot(5, 2, 3)
    plt.plot(time, gt[0][:len(alt_est)], 'g--', label='Ground Truth')
    plt.plot(time, alt_est, 'r-', label=f'{imu_name} Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs. Time')
    plt.legend()
    plt.grid(True)

    # --------------------------
    # Velocity States (Separate Plots)
    # --------------------------

    # North Velocity (V_N)
    plt.subplot(5, 2, 4)
    plt.plot(time, gt[8][:len(v_n_est)], 'g--', label='Ground Truth V_N')
    plt.plot(time, v_n_est, 'r-', label=f'{imu_name} Estimated V_N')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity North vs. Time')
    plt.legend()
    plt.grid(True)

    # East Velocity (V_E)
    plt.subplot(5, 2, 5)
    plt.plot(time, gt[7][:len(v_e_est)], 'g--', label='Ground Truth V_E')
    plt.plot(time, v_e_est, 'r-', label=f'{imu_name} Estimated V_E')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity East vs. Time')
    plt.legend()
    plt.grid(True)

    # Up Velocity (V_U)
    plt.subplot(5, 2, 6)
    plt.plot(time, gt[6][:len(v_up_est)], 'g--', label='Ground Truth V_U')
    plt.plot(time, v_up_est, 'r-', label=f'{imu_name} Estimated V_U')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Up vs. Time')
    plt.legend()
    plt.grid(True)

    # --------------------------
    # Orientation States (Separate Plots)
    # --------------------------

    # Yaw
    plt.subplot(5, 2, 7)
    plt.plot(time, np.degrees(gt[5][:len(yaw_est)]), 'g--', label='Ground Truth Yaw')
    plt.plot(time, np.degrees(yaw_est), 'r-', label=f'{imu_name} Estimated Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (deg)')
    plt.title('Yaw vs. Time')
    plt.legend()
    plt.grid(True)

    # Pitch
    plt.subplot(5, 2, 8)
    plt.plot(time, np.degrees(gt[3][:len(pitch_est)]), 'g--', label='Ground Truth Pitch')
    plt.plot(time, np.degrees(pitch_est), 'r-', label=f'{imu_name} Estimated Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (deg)')
    plt.title('Pitch vs. Time')
    plt.legend()
    plt.grid(True)

    # Roll
    plt.subplot(5, 2, 9)
    plt.plot(time, np.degrees(gt[4][:len(roll_est)]), 'g--', label='Ground Truth Roll')
    plt.plot(time, np.degrees(roll_est), 'r-', label=f'{imu_name} Estimated Roll')
    plt.xlabel('Time (s)')
    plt.ylabel('Roll (deg)')
    plt.title('Roll vs. Time')
    plt.legend()
    plt.grid(True)

    # --------------------------
    # Trajectory (Latitude vs. Longitude)
    # --------------------------

    plt.subplot(5, 2, 10)
    plt.plot(np.degrees(gt[2][:len(lon_est)]), np.degrees(gt[1][:len(lat_est)]), 'g--', label='Ground Truth')
    plt.plot(np.degrees(lon_est), np.degrees(lat_est), 'r-', label=f'{imu_name} Estimated')
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Trajectory (Latitude vs. Longitude)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{imu_name}_state_comparison.png', dpi=300)
    plt.show()


def plot_trajectory(gt, estimated_state , name):
    # Plotting (convert to degrees for visualization)
    plt.figure(figsize=(12, 8))
    plt.plot(np.degrees(gt[2]), np.degrees(gt[1]), 'g--', label='Ground Truth')
    plt.plot(np.degrees(estimated_state[4]), np.degrees(estimated_state[3]), 'r-', label=f'{name}Estimated')
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
