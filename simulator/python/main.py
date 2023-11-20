import math
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

#######################################################################
# Defining functions
#######################################################################

def J(psi):
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
    
    lx1 = -35; ly1 = 7                                  # azimuth thruster 1 position (m)
    lx2 = -35; ly2 = -7                                 # azimuth thruster 2 position (m)
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


#######################################################################
# Defining the ship parameters and OCP constraints
#######################################################################

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
alpha1_min = -170*math.pi/180; alpha1_max = 170*math.pi/180                 # azimuth thruster 1 angle saturation (deg)
alpha2_min = -170*math.pi/180; alpha2_max = 170*math.pi/180                 # azimuth thruster 1 angle saturation (deg)
alpha3 = math.pi/2                                  # bow tunnel thruster constant angle (rad)
alpha_dot_max = (360*math.pi/180)/30                              # azimuth thruster max turnaround time (deg/s) ------------------------------CHECK THIS-------------------------------------------

# Weighting matrices
Q_eta = np.diag([1e4,1e4,1e7])                            # weighting matrix for position and Euler angle vector eta ---------------------CHECK THESE THREE----------------------
Q_nu = np.diag([1,1e-2,1e-1])                             # weighting matrix for linear and angular velocity vector nu
R_f = np.diag([1e-7,1e-7,1e-7])                              # weighting matrix for force vector f
W = np.diag([1,1,1])                                # thruster weighting matrix
epsilon = 1e-3                                      # small constant to avoid division by 0
rho = 1                                             # thruster weighting of maneuverability

N_mtrx = np.diag([1, 1, L])                         # diagonal normalization matrix (bis-system)
M_bis = np.array([                                  # non-dimensional mass matrix (bis-system)
    [1.1274, 0, 0],
    [0, 1.8902, -0.0744],
    [0, -0.0744, 0.1278]])

D_bis = np.array([                                  # non-dimensional dampening matrix (bis-system)
    [0.0358, 0, 0],
    [0, 0.1183, -0.0124],
    [0, -0.0041, 0.0308]])

M_vessel = m * N_mtrx @ M_bis @ N_mtrx                     # vessel mass matrix (kg)
D_vessel = m * math.sqrt(g/L) * N_mtrx @ D_bis @ N_mtrx    # vessel dampening matrix (kg)

f_min = np.array([f1_min, f2_min, f3_min])
f_max = np.array([f1_max, f2_max, f3_max])

# alpha_min_values = np.array([alpha1_min, alpha2_min])
# alpha_max_values = np.array([alpha1_max, alpha2_max])
# alpha_min = ca.MX(alpha_min_values)
# alpha_max = ca.MX(alpha_max_values)

# alpha_dot_max = ca.MX((360*math.pi/180)/30)
# alpha_dot_magnitude = ca.norm_2(alpha_dot)


#######################################################################
# Direct collocation setup
#######################################################################

# Degree of interpolating polynomial
d = 3

# Get collocation points
tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

# Coefficients of the collocation equation
C = np.zeros((d+1, d+1))

# Coefficients of the continuity equation
D = np.zeros(d+1)

# Coefficients of the quadrature function
B = np.zeros(d+1)

# Construct polynomial basis
for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    L = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            L *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = L(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    Lder = np.polyder(L)
    for r in range(d+1):
        C[j, r] = Lder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    Lint = np.polyint(L)
    B[j] = Lint(1.0)

#######################################################################
# Building the NLP
#######################################################################

# Time horizon
time_horizon = 300

# Declare model variables
x   = ca.SX.sym('x')
y   = ca.SX.sym('y')
psi = ca.SX.sym('psi')
eta = ca.vertcat(x,y,psi)

x_d   = ca.SX.sym('x_d')
y_d   = ca.SX.sym('y_d')
psi_d = ca.SX.sym('psi_d')
eta_d = ca.vertcat(x_d,y_d,psi_d)

u  = ca.SX.sym('u')
v  = ca.SX.sym('v')
r  = ca.SX.sym('r')
nu = ca.vertcat(u,v,r)

f1    = ca.SX.sym('f1')
f2    = ca.SX.sym('f2')
f3    = ca.SX.sym('f3')
f_thr = ca.vertcat(f1,f2,f3)

alpha1 = ca.SX.sym('alpha1')
alpha2 = ca.SX.sym('alpha2')
alpha  = ca.vertcat(alpha1,alpha2)

X = ca.vertcat(eta,eta_d,nu)        # NLP state vector -------------MAYBE ETA_D INTO U INSTEAD??-------------
U = ca.vertcat(f_thr,alpha)         # NLP input vector

x_init = [150,-325,-math.pi, 0,0,0, 0,0,0]
u_init = [0,0,0, 0,0]

# Model equations
eta_dot = ca.mtimes(J(psi), nu)
nu_dot = ca.mtimes(ca.inv(M_vessel), ca.mtimes(T(alpha), f_thr) - ca.mtimes(D_vessel, nu))

# Objective term
obj_det = ca.det(ca.mtimes(T(alpha), ca.mtimes(ca.inv(W), T(alpha).T)))     # singular configuration cost
objective = (ca.mtimes((eta-eta_d).T, ca.mtimes(Q_eta, (eta-eta_d))) +      # continuous time objective
             ca.mtimes(nu.T, ca.mtimes(Q_nu, nu)) +
             ca.mtimes(f_thr.T, ca.mtimes(R_f, f_thr)) +
             rho / (epsilon + obj_det))

# Continuous time dynamics
f = ca.Function('f', [X, U], [eta_dot, nu_dot, objective], ['X', 'U'], ['eta_dot', 'nu_dot', 'objective'])

# Control discretization
N = 30                  # number of control intervals
h = time_horizon / N    # step time

# NLP formulation: parameters we wish to minimize the cost w.r.t.
w=[]                    # decision variables
w0 = []                 # init. cond. on every decision variable
lbw = []
ubw = []
J = 0                   # objective (x1,...xN,u1,...,uN)
g=[]                    # constraints
lbg = []
ubg = []

# For plotting x and u given w
x_plot = []
u_plot = []

# "Lift" initial conditions
Xk = ca.MX.sym('X0', X.size1())
w.append(Xk)
lbw.append(x_init)      # Equality constraint so bounded upper and lower --------------------------CHECK THIS---------------------------------
ubw.append(x_init)
w0.append(x_init)       # x(0)=150, y(0)=-325, psi(0)=-pi, x_d(0)=150, y_d(0)=-325, psi_d(0)=-pi, u(0)=0, v(0)=0, r(0)=0
x_plot.append(Xk)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k), U.size1())
    w.append(Uk)
    lbw.append([-1])
    ubw.append([1])
    w0.append([0])
    u_plot.append(Uk)

