import math
import numpy as np
import casadi as ca


#######################################################################
# Defining constants
#######################################################################

knot2ms = 0.514                                     # conversion constant for knots to m/s
rad2deg = 180 / math.pi                             # conversion constant for radians to degrees


#######################################################################
# Defining functions
#######################################################################

def ssa(angle):
    '''
    Smallest signed angle. Maps an angle in rad to the interval [-pi,pi)
    
        Parameters:
            psi (float): An heading angle in radians

        Returns:
            angle (float): Smallest difference between the angles
    
    '''

    angle = ca.fmod(angle + ca.pi, 2 * ca.pi) - ca.pi

    return angle

def J_rot(psi):
    '''
    Returns the Euler angle rotation matrix J(psi) in SO(3) using the zyx convention
    for 3-DOF model for surface vessels. Fossen (2.60).

        Parameters:
            psi (float): An heading angle in radians

        Returns:
            J_psi (ndarray): Rotation matrix in SO(3)
    '''
    
    cpsi = ca.cos(psi)
    spsi = ca.sin(psi)
    
    J_psi = np.array([
        [ cpsi, -spsi, 0 ],
        [ spsi,  cpsi, 0 ],
        [  0,     0,   1 ] ])

    return J_psi

def R_rot(psi):
    '''
    Returns the Euler angle rotation matrix R(psi) in SO(2) using the zyx convention
    for 2-DOF model for surface vessels.

        Parameters:
            psi (float): An heading angle in radians

        Returns:
            R_psi (ndarray): Rotation matrix in SO(2)
    '''
    
    cpsi = ca.cos(psi)
    spsi = ca.sin(psi)
    
    R_psi = np.array([
        [ cpsi, -spsi],
        [ spsi,  cpsi] ])

    return R_psi

def R(psi):
    '''
    Returns the Euler angle rotation matrix R(psi) in SO(2) using the zyx convention
    for 2-DOF model for surface vessels.

        Parameters:
            psi (float): An heading angle in radians

        Returns:
            R_psi (ndarray): Rotation matrix in SO(2)
    '''
    
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R_psi = np.array([
        [ cpsi, -spsi],
        [ spsi,  cpsi] ])

    return R_psi

def T(alpha):
    '''
    Returns the thrust configuration matrix dependent on the angles alpha. 
    Assuming two azimuth thrusters in the aft, with one tunnel thruster in
    the front with positions and angles given from the vessel model.
    Fossen ch. 11.2.1.

        Parameters:
            alpha (list): Azimuth thruster angles in radians

        Returns:
            T_thr (ndarray): Thruster configuration matrix
    '''
    
    lx1 = -35; ly1 = -7                                  # azimuth thruster 1 position (m)
    lx2 = -35; ly2 = 7                                 # azimuth thruster 2 position (m)
    lx3 = 35;                                           # bow tunnel thruster position (m)

    calpha1 = ca.cos(alpha[0])
    salpha1 = ca.sin(alpha[0])
    calpha2 = ca.cos(alpha[1])
    salpha2 = ca.sin(alpha[1])
    
    T_thr = ca.vertcat(
        ca.horzcat(calpha1, calpha2, 0),
        ca.horzcat(salpha1, salpha2, 1),
        ca.horzcat(lx1*salpha1-ly1*calpha1, lx2*salpha2-ly2*calpha2, lx3)
    )

    return T_thr
