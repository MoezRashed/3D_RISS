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
    
    Args:
        gt (list): Ground truth data.
        estimated_states (list): Estimated states from the IMU.
        imu_name (str): Name of the IMU (e.g., "KVH" or "TPI").
    """
    # Extract states
    v_n_est, v_e_est, v_up_est = estimated_states[0], estimated_states[1], estimated_states[2]
    lat_est, lon_est, alt_est = estimated_states[3], estimated_states[4], estimated_states[5]
    yaw_est, pitch_est, roll_est = estimated_states[6], estimated_states[7], estimated_states[8]

    # Time vector (assuming 1 Hz data)
    time = np.arange(len(lat_est))

    # Plot Position States
    plt.figure(figsize=(15, 10))

    # Latitude
    plt.subplot(3, 2, 1)
    plt.plot(time, np.degrees(gt[1][:len(lat_est)]), 'g--', label='Ground Truth')
    plt.plot(time, np.degrees(lat_est), 'r-', label=f'{imu_name} Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Latitude (deg)')
    plt.title('Latitude vs. Time')
    plt.legend()
    plt.grid(True)

    # Longitude
    plt.subplot(3, 2, 2)
    plt.plot(time, np.degrees(gt[2][:len(lon_est)]), 'g--', label='Ground Truth')
    plt.plot(time, np.degrees(lon_est), 'r-', label=f'{imu_name} Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Longitude (deg)')
    plt.title('Longitude vs. Time')
    plt.legend()
    plt.grid(True)

    # Altitude
    plt.subplot(3, 2, 3)
    plt.plot(time, gt[0][:len(alt_est)], 'g--', label='Ground Truth')
    plt.plot(time, alt_est, 'r-', label=f'{imu_name} Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs. Time')
    plt.legend()
    plt.grid(True)

    # Velocity (North, East, Up)
    plt.subplot(3, 2, 4)
    plt.plot(time, gt[8][:len(v_n_est)], 'g--', label='Ground Truth V_N')
    plt.plot(time, v_n_est, 'r-', label=f'{imu_name} Estimated V_N')
    plt.plot(time, gt[7][:len(v_e_est)], 'b--', label='Ground Truth V_E')
    plt.plot(time, v_e_est, 'm-', label=f'{imu_name} Estimated V_E')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs. Time')
    plt.legend()
    plt.grid(True)

    # Orientation (Yaw, Pitch, Roll)
    plt.subplot(3, 2, 5)
    plt.plot(time, np.degrees(gt[5][:len(yaw_est)]), 'g--', label='Ground Truth Yaw')
    plt.plot(time, np.degrees(yaw_est), 'r-', label=f'{imu_name} Estimated Yaw')
    plt.plot(time, np.degrees(gt[3][:len(pitch_est)]), 'b--', label='Ground Truth Pitch')
    plt.plot(time, np.degrees(pitch_est), 'm-', label=f'{imu_name} Estimated Pitch')
    plt.plot(time, np.degrees(gt[4][:len(roll_est)]), 'k--', label='Ground Truth Roll')
    plt.plot(time, np.degrees(roll_est), 'c-', label=f'{imu_name} Estimated Roll')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation (deg)')
    plt.title('Orientation vs. Time')
    plt.legend()
    plt.grid(True)

    # Trajectory (Latitude vs. Longitude)
    plt.subplot(3, 2, 6)
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
