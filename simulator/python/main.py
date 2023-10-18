#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py: Simple simulator for the SV Northern Clipper heavily inspired by Fossen's lecture notes and code from the MSS toolbox.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd edition, John Wiley & Sons, Chichester, UK. 
URL: https://www.fossen.biz/wiley  
    
Author:     Håvard Olai Kopperstad
"""


import numpy as np
import matplotlib.pyplot as plt
from lib.gnc import *


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
simData = np.empty([N+1,1+np.size(eta)+np.size(nu)], float)


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
    J = np.array(np.concatenate((np.concatenate((R, np.zeros((3,3))), axis=1),
                                np.concatenate((np.zeros((3,3)), T), axis=1))))

    # Differential equations
    eta_dot = J @ nu
    nu_dot = np.array([(1/m) * (tau1 - d_u * u), 0, 0, 0, 0, (1/Iz) * (tau6 - d_r * r)])

    # Store simulation data in table
    simData[i,0] = t
    simData[i,1:] = np.array([np.concatenate((eta,nu))])

    # Euler's method (k+1)
    eta = eta + h * eta_dot
    nu = nu + h * nu_dot


# PLOT
t     = simData[:,0]

x     = simData[:,1]
y     = simData[:,2]
z     = simData[:,3]
phi   = rad2deg(ssa(simData[:,4]))
theta = rad2deg(ssa(simData[:,5]))
psi   = rad2deg(ssa(simData[:,6]))

u     = simData[:,7]
v     = simData[:,8]
w     = simData[:,9]
p     = rad2deg(ssa(simData[:,10]))
q     = rad2deg(ssa(simData[:,11]))
r     = rad2deg(ssa(simData[:,12]))

U = np.sqrt(np.multiply(u,u) + np.multiply(v,v))    # speed
beta_c = rad2deg(ssa(np.arctan2(v,u)))              # crab angle
chi = rad2deg(ssa(simData[:,6] + np.arctan2(v,u)))  # course angle, chi = psi + beta_c

# Position and Euler angle plots
legendTextSize = 6
titleTextSize = 8
axisTextSize = 6

plt.figure(1, figsize=(20/2.54, 10/2.54), dpi=150)

plt.subplot(3,2,1)
plt.plot(y,x)
plt.title("North-East positions (m)", fontsize=titleTextSize, fontweight="bold")
plt.ylabel("North (m)", fontsize=axisTextSize)
plt.xlabel("East (m)", fontsize=axisTextSize)
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(t, z)
plt.title("Down position (m)", fontsize=titleTextSize, fontweight="bold")
plt.xlabel("time (s)", fontsize=axisTextSize)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, phi, t, theta)
plt.title("Roll and pitch angles (deg)", fontsize=titleTextSize, fontweight="bold")
plt.xlabel("time (s)", fontsize=axisTextSize)
plt.legend(["Roll angle (deg)", "Pitch angle (deg)"], fontsize=legendTextSize).set_loc('upper right')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, psi, t, chi, t, beta_c)
plt.title("Heading and course angles (deg)", fontsize=titleTextSize, fontweight="bold")
plt.xlabel("time (s)", fontsize=axisTextSize)
plt.legend(["Yaw angle (deg)","Course angle (deg)","Crab angle (deg)"], fontsize=legendTextSize).set_loc('upper right')
plt.grid()

plt.figure(2, figsize=(20/2.54, 10/2.54), dpi=150)

plt.subplot(3, 1, 1)
plt.plot(t, U)
plt.title("Speed (m/s)", fontsize=titleTextSize, fontweight="bold")
plt.xlabel("time (s)", fontsize=axisTextSize)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, u, t, v, t, w)
plt.title("Linear velocities (m/s)", fontsize=titleTextSize, fontweight="bold")
plt.legend(["u (m/s)","v (m/s)","w (m/s)"], fontsize=legendTextSize).set_loc('upper right')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, p, t, q, t, r)
plt.title("Angular velocities (deg/s)", fontsize=titleTextSize, fontweight="bold")
plt.xlabel("time (s)", fontsize=axisTextSize)
plt.legend(["p (deg/s)", "q (deg/s)", "r (deg/s)"], fontsize=legendTextSize).set_loc('upper right')
plt.grid()

plt.subplots_adjust(hspace=1)
plt.show()
