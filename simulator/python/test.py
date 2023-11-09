#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py:
    Simulation of autonomous docking for the SV Northern Clipper using direct optimal control.

Description:
    This code is a replication of Andreas B. Martinsen's paper from 2019: "Autonomous docking using direct optimal control".
    The ship model used for the simulations is based on Martinsen 2019 and T. I. Fossen's model of the SV Northern Clipper.
    Theory applied in this simulator is based on Fossen's 2021 book.

References: 
    Martinsen, A., Lekkas, A. M. & Gros, S. (2019). Autonomous docking using direct optimal control. IFAC-PapersOnLine 52 (21), 97-102.
    https://www.sciencedirect.com/science/article/pii/S2405896319321755

    Fossen, T. I. (2021). Handbook of Marine Craft Hydrodynamics and Motion Control (2nd edition). John Wiley & Sons, Chichester, UK. 
    
Author:
    Håvard Olai Kopperstad
"""


import casadi as cd
import numpy as np
import math

# User inputs
T = 300                                             # time horizon (s)
N = 30                                              # no. time steps

# Initial states
eta_0 = [150, -325, -math.pi]                       # initial vessel pose
nu_0 = [0, 0, 0]                                    # initial vessel velocity
f_0 = [0, 0, 0]                                     # initial thruster force (kN)
alpha_0 = [0, 0, math.pi/2]                         # initial thruster angles (rad)

# Vessel parameters
L = 76.2                                            # vessel length (m)
g = 9.8                                             # gravitational acceleration (m/s^2)
m = 6000e3                                          # vessel mass (kg)
lx1 = -35; ly1 = 7                                  # azimuth thruster 1 position (m)
lx2 = -35; ly2 = -7                                 # azimuth thruster 2 position (m)
lx3 = 35;  ly3 = 0                                  # bow tunnel thruster position (m)
alpha3 = math.pi/2                                  # bow tunnel thruster angle (rad)
f1_min = -1/30 * m; f1_max = 1/30 * m               # azimuth thruster 1 force saturation (kN) -------------------------------CHECK THIS-------------------------------------------
f2_min = -1/30 * m; f2_max = 1/30 * m               # azimuth thruster 2 force saturation (kN) -------------------------------CHECK THIS-------------------------------------------
f3_min = -1/60 * m; f3_max = 1/60 * m               # bow tunnel thruster force saturation (kN) ------------------------------CHECK THIS-------------------------------------------
alpha1_min = -170; alpha1_max = 170                 # azimuth thruster 1 angle saturation (deg)
alpha2_min = -170; alpha2_max = 170                 # azimuth thruster 1 angle saturation (deg)
alpha_dot_max = 360/30                              # azimuth thruster max turnaround time (deg/s) ------------------------------CHECK THIS-------------------------------------------

N_mtrx = np.diag([1, 1, L])                         # diagonal normalization matrix (bis-system)
M_bis = np.array([                                  # non-dimensional mass matrix (bis-system)
    [1.1274, 0, 0],
    [0, 1.8902, -0.0744],
    [0, -0.0744, 0.1278]])

D_bis = np.array([                                  # non-dimensional dampening matrix (bis-system)
    [0.0358, 0, 0],
    [0, 0.1183, -0.0124],
    [0, -0.0041, 0.0308]])

M = m * N_mtrx @ M_bis @ N_mtrx                     # vessel mass matrix (kg)
D = m * math.sqrt(g/L) * N_mtrx @ D_bis @ N_mtrx    # vessel dampening matrix (kg)

# Weighting matrices
Q_eta = np.diag([1,1,1])                            # weighting matrix for position and Euler angle vector eta
Q_nu = np.diag([1,1,1])                             # weighting matrix for linear and angular velocity vector nu
R_f = np.diag([1,1,1])                              # weighting matrix for force vector f
W = np.diag([1,1,1])                                # thruster weighting matrix
epsilon = 1e-6                                      # small constant to avoid division by 0
rho = 1                                             # thruster weighting of maneuverability


# Define symbolic variables
eta = cd.MX.sym('eta', 3)
eta_d = cd.MX.sym('eta_d', 3)
nu = cd.MX.sym('nu', 3)
f = cd.MX.sym('f', 3)
alpha = cd.MX.sym('alpha', 2)

# # Define the cost function
# obj_det = cd.det(T(alpha)@cd.inv(W)@T(alpha).T)
# obj = (cd.mtimes((eta-eta_d).T, Q_eta, (eta-eta_d)) + 
#        cd.mtimes(nu.T, Q_nu, nu) + 
#        cd.mtimes(f.T, R_f, f) + 
#        rho / (epsilon + obj_det))

# # Define the dynamics
# eta_dot = cd.mtimes(J_psi, nu)
# nu_dot = cd.mtimes(cd.inv(M), cd.mtimes(R_psi, f) - cd.mtimes(D, nu))
# constraints = [
#     cd.mtimes(A_s, cd.mtimes(R_psi, x_i_b) + cd.vertcat(eta[0], eta[1])) <= 0,
#     f_min <= f,
#     f <= f_max,
#     alpha_min <= alpha,
#     alpha <= alpha_max,
#     cd.fabs(alpha_dot) <= alpha_dot_max
# ]

# # Formulate the optimization problem
# opti = cd.Opti()
# opti.minimize(cd.integral(obj, 0, T))
# opti.subject_to(cd.vertcat(eta_dot, nu_dot) == cd.vertcat(nu, nu_dot))
# opti.subject_to(constraints)

# # Choose solver
# opts = {'ipopt.print_level': 0, 'print_time': 0}
# solver_opts = {'ipopt': opts}
# solver = opti.solver('ipopt', solver_opts)

# # Solve the OCP
# sol = solver(x0=opti.x0())

print(eta)
