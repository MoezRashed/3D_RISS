import math
import numpy as np

def pitch_calc(fx , fy , fz, a_od):

    return math.atan2(-(fy - a_od), math.sqrt(fx**2 + fz**2))

def roll_calc(fx, fz, v_od, azimuth):

    return math.atan2(-(fx + (v_od * azimuth)), fz)

def yaw_rate(pitch, roll, wz, we, ve, r, lat, alt):

    yaw = (np.cos(pitch) * np.cos(roll) * wz) - (we * np.sin(lat)) -((ve * np.tan(lat)) / (r + alt))

    return yaw

def transform_to_navigation_frame(yaw, pitch, v_od):
    
    V_e = -np.sin(yaw) * np.cos(pitch) * v_od
    V_n =  np.cos(yaw) * np.cos(pitch) * v_od
    V_u =  np.sin(pitch) * v_od

    return [V_e, V_n, V_u]



def total_horizontal_distane(gt , Rm , Rn):

    delta_ph         = 0
    for i in range(1, len(gt[-1])):

        delta_phi    = (gt[1][i] - gt[1][i-1])
        delta_lambda = (gt[2][i] - gt[2][i-1])
        delta_pn     = (Rm + gt[0][i]) * delta_phi
        delta_pe     = (Rn + gt[0][i]) * np.cos(gt[1][i]) * delta_lambda
        delta_ph     += np.sqrt(delta_pn**2 + delta_pe**2)
    
    return delta_ph


def delta_position_errors(gt,estimated, Rm, Rn):

    # Variable to store delta position horizontal
    delta_ph         = 0
    # List to store error across the trajectory
    delta_pu_traj    = []
    delta_pn_traj    = []
    delta_pe_traj    = []
    delta_ph_traj    = []
    delta_3d_traj    = []

    for i in range(len(gt[-1])):

        delta_alt    = (estimated[5][i] - gt[0][i])
        delta_phi    = (estimated[3][i] - gt[1][i])
        delta_lambda = (estimated[4][i] - gt[2][i])
        delta_pn     = (Rm + gt[0][i]) * delta_phi
        delta_pe     = (Rn + gt[0][i]) * np.cos(gt[1][i]) * delta_lambda
        delta_ph     = np.sqrt(delta_pn**2 + delta_pe**2)
        delta_3d     = np.sqrt(delta_pn**2 + delta_pe**2 + delta_alt**2)

        delta_pn_traj.append(delta_pn)
        delta_pe_traj.append(delta_pe)
        delta_ph_traj.append(delta_ph)
        delta_pu_traj.append(delta_alt)
        delta_3d_traj.append(delta_3d)

    return delta_pn_traj, delta_pe_traj, delta_ph_traj, delta_pu_traj,delta_3d_traj

def rmse(delta):
   
   return np.sqrt(np.mean(np.square(delta)))

def max_error(delta):
    return max(delta)

# def online_caibration():
# def radius_calculation():