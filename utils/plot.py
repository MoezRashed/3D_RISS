import matplotlib.pyplot as plt 


def plot_sensor_data(x_data, y_data, z_data, time):
    plt.figure(figsize=(12, 6))
    plt.plot(time, x_data, label='X-axis')
    plt.plot(time, y_data, label='Y-axis')
    plt.plot(time, z_data, label='Z-axis')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sensor Value')
    plt.title('IMU Sensor Data Over Time')
    plt.legend()
    plt.show()