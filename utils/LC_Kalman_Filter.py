import numpy    as     np
import os
from   scipy.io import loadmat
from   utils    import data_manipulation

def H_matrix():
    H = np.zeros((6,9))
    for i in range (6):
        H[i,i] = 1
    return H

def sec(x):
    return (1 / np.cos(x))

def F_matrix(IMU_calculations, alpha, beta, pitch, roll):
   
    we     = 7.292115e-5  # Earth's rotation rate (rad/s)
    R      = 6378137.0    # Earth's radius (m)
    
    lat    = IMU_calculations[0]
    alt    = IMU_calculations[2]
    Ve     = IMU_calculations[3]
    Vn     = IMU_calculations[4]
    yaw    = IMU_calculations[6]
    acc_od = IMU_calculations[8]  
    
    F = np.array([
        # Row 0: lat dynamics (originally used -Vn/(R+alt))
        [ 0,
          0,
          -Vn / ((R + alt)**2),
          0,
          1/(R+alt),
          0,
          0,
          0,
          0 ],
          
        # Row 1: long dynamics (originally using Ve, lat, and alt)
        [ (Ve * np.sin(lat)) / ((R + alt) * (np.cos(lat) ** 2)),
          0,
          -Ve / (((R + alt)**2) * np.cos(lat)),
          1 / ((R + alt) * np.cos(lat)),
          0,
          0,
          0,
          0,
          0 ],
          
        # Row 2: alt dynamics (vertical velocity Vu)
        [ 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
        
        # Row 3: Ve dynamics (east velocity)
        # Original used yaw, pitch, and acceleration; with pitch=0 (cos(0)=1)
        [ 0,
          0,
          0,
          0,
          0,
          0,
          -np.cos(yaw)* np.cos(pitch) * acc_od,
          0,
          -np.sin(yaw)* np.cos(pitch) ],
        
        # Row 4: Vn dynamics (north velocity)
        [ 0,
          0,
          0,
          0,
          0,
          0,
          -np.sin(yaw)* np.cos(pitch) * acc_od,
          0,
          np.cos(yaw) * np.cos(pitch) ],
        
        # Row 5: Vu dynamics (vertical velocity); originally depended on pitch,
        [ 0, 0, 0, 0, 0, 0, 0, 0, np.sin(pitch) ],
        
        # Row 6: yaw dynamics
        [ -we * np.cos(lat) - ((Ve * ((1/np.cos(lat))**2)) / (R + alt)),
          0,
          (Ve * np.tan(lat)) / ((R + alt)**2),
          -np.tan(lat) / (R + alt),
          0,
          0,
          np.cos(pitch) * np.cos(roll),
          0,
          0 ],
        
        # Row 7: wz dynamics (assumed constant)
        [ 0, 0, 0, 0, 0, 0, 0, -beta, 0 ],
        
        # Row 8: acc_od dynamics (assumed constant)
        [ 0, 0, 0, 0, 0, 0, 0, 0, -alpha ]
    ])
    
    return F

def G_matrix(bwz_c, bwz_s, a_od_c, a_od_s):
    G = np.eye((9,9)) 
    G[8,8] = np.sqrt(2*(1/a_od_c)*(a_od_s **2))
    G[9,9] = np.sqrt(2*(1/bwz_c)*(bwz_s **2))
    return G

def Q_matrix():
    # Create a 9x9 zero matrix for Q
    Q = np.zeros((9, 9))
    Q[0, 0] = 1e-15       # lat
    Q[1, 1] = 1e-15       # long
    Q[2, 2] = 1e-15       # h
    Q[3, 3] = 0           # ve
    Q[4, 4] = 0           # vn
    Q[5, 5] = 0           # vu
    Q[6, 6] = 0           # A
    Q[7, 7] = 1e-5        # a_od
    Q[8, 8] = 0.001 * np.pi / 180  # Bz
    return Q

def P_matrix_initial():
    # Create a 9x9 identity matrix
    P = np.eye(9)
    P[0, 0] = 0.00157       # lat
    P[1, 1] = 0.00156       # long
    P[2, 2] = 100000        # h
    P[3, 3] = 160 * 10000 / 3600  # ve
    P[4, 4] = 160 * 10000 / 3600  # vn
    P[5, 5] = 160 * 10000 / 3600  # vu
    P[6, 6] = np.pi         # Yaaaaaaw
    P[7, 7] = 100           # a_od
    P[8, 8] = 10            # Bz
    return P

def R_matrix(alt_std,lat_std,long_std,alt,lat):

    # Lat_std, long_std are in meters
    M = 6367636
    lat_std_in_rad = lat_std / (M + alt)
    long_std_in_rad = long_std / ((M + alt) * np.cos(lat))

    # Create a 6x6 zero matrix for R
    R = np.zeros((6, 6))
    R[0, 0] = lat_std_in_rad ** 2
    R[1, 1] = long_std_in_rad ** 2
    R[2, 2] = alt_std ** 2
    R[3, 3] = (long_std * 5) ** 2
    R[4, 4] = (lat_std * 5) ** 2
    R[5, 5] = (alt_std * 10) ** 2

# def LC_Kalman_filter(Measurment_GNSS, R_Matrix_GNSS, IMU_Calculations):

    