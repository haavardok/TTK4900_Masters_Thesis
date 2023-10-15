#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py: Simple simulator for the SV Northern Clipper heavily inspired by Fossen's lecture notes.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd edition, John Wiley & Sons, Chichester, UK. 
URL: https://www.fossen.biz/wiley  
    
Author:     Håvard Olai Kopperstad
"""


import math
import numpy as np
from lib.gnc import *

def deg2rad(angleInDegrees): # Might need to check if input is float. See L1euler.m!
    """Convert angles from degrees to radians."""
    angleInRad = (math.pi/180) * angleInDegrees

    return angleInRad

def ssa(angle): # Might need to add support for vectors. See L1euler.m!
    """Smallest signed angle. Maps an angle in rad to the interval [-pi,pi)."""
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return angle


# User inputs
h = 0.1             # sampling time
N = 150             # number of samples

# Autopilot setpoints
u_d = 3             # desired surge veocity
psi_d = deg2rad(10) # desired yaw angle

# Model parameters
m = 1
Iz = 1
d_u = 1
d_r = 1

# Controller gains
kp_u = 1            # surge P-controller gain
kp_psi = 1          # yaw P-controller gain

# Initial states
eta = np.array([0, 0, 0, 0, 0, 0])          # eta = [x, y, z, phi, theta, psi]^T
nu = np.array([0.5, 0.1, 0, 0, 0, 0])       # nu = [u, v, w, p, q, r]^T

# Allocate empty table for simulation data
simdata = np.empty([0,12 + 2*2], float)

# MAIN LOOP
for i in range(0,N+1):

    t = i * h       # current simulation time step, t

    # Measurements
    psi = eta[5]
    u = nu[0]
    r = nu[5]

    # Control laws
    tau1 = d_u * u_d - kp_u * (u - u_d)     # surge P-controller
    tau6 = -kp_psi * ssa(psi - psi_d)       # yaw P-controller

    # Kinematics
    R = Rzyx(eta[3],eta[4],eta[5])
    T = Tzyx(eta[3],eta[4])
    J = np.array([[1, 0, 0, 0, 0, 0], # FIX THIS MATRIX: [R, zeros(3,3); zeros(3,3), T]
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    # Differential equations
    eta_dot = J * nu
    nu_dot = np.array([(1/m) * (tau1 - d_u * u), 0, 0, 0, 0, (1/Iz) * (tau6 - d_r * r)])

    # Store simulation data in table
    #simdata = np.vstack(simdata, [t, eta, nu])

# Store simulation time vector
#simTime = np.arange(start=0, stop=t+N, step=h)[:, None]


#print(simTime)