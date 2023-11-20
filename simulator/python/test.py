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
Q_eta = np.diag([1e4,1e4,1e7])                            # weighting matrix for position and Euler angle vector eta
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
T_hor = 300

# Declare model variables
eta = ca.SX.sym('eta', 3)
eta_d = ca.vertcat(0,0,0)
nu = ca.SX.sym('nu', 3)
f_thr = ca.SX.sym('f_thr', 3)
alpha = ca.SX.sym('alpha', 2)
# alpha_dot = ca.SX.sym('alpha_dot', 2)

# Model equations
# eta_dot = ca.mtimes(J(eta[2]), nu)
eta_dot = ca.mtimes(J(eta[2]), eta)
nu_dot = ca.mtimes(ca.inv(M_vessel), ca.mtimes(T(alpha), f_thr) - ca.mtimes(D_vessel, nu))

# Objective term
# obj_det = ca.det(T(alpha)@ca.inv(W)@T(alpha).T)
# Obj_vessel = (ca.mtimes((eta-eta_d).T, ca.mtimes(Q_eta, (eta-eta_d))) +
#               ca.mtimes(nu.T, ca.mtimes(Q_nu, nu)) +
#               ca.mtimes(f_thr.T, ca.mtimes(R_f, f_thr)) +
#               rho / (epsilon + obj_det))
Obj_vessel = (ca.mtimes((eta-eta_d).T, ca.mtimes(Q_eta, (eta-eta_d))) +
              ca.mtimes(f_thr.T, ca.mtimes(R_f, f_thr)))

# Continuous time dynamics
# f_vessel = ca.Function('f_vessel', [eta, nu, f_thr, alpha], [eta_dot, nu_dot], ['eta', 'nu', 'f_thr', 'alpha'], ['eta_dot', 'nu_dot'])
f_vessel = ca.Function('f_vessel', [eta, f_thr], [eta_dot, Obj_vessel], ['eta', 'f_thr'], ['eta_dot', 'Obj_vessel'])

# Control discretization
N = 30 # number of control intervals
h = T_hor/N # time step

# Start with an empty NLP
# Params we wish to minimize the cost w.r.t.
w_vessel=[]        # decision variables
w0_vessel = []
lbw_vessel = []
ubw_vessel = []
J_vessel = 0       # objective
g_vessel=[]        # constraints
lbg_vessel = []
ubg_vessel = []

# For plotting eta, nu, f_thr and alpha given w
eta_plot = []
nu_plot = []
f_thr_plot = []
alpha_plot = []

# "Lift" initial conditions
eta_k = ca.MX.sym('eta0', 3)
w_vessel.append(eta_k)
lbw_vessel.append([150, -325, -math.pi])    # Equality constraint for the ship states
ubw_vessel.append([150, -325, -math.pi])
w0_vessel.append([150, -325, -math.pi])     # eta[0] = 150, eta[1] = -325, eta[2] = -pi
eta_plot.append(eta_k)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    f_thr_k = ca.MX.sym('f_thr_' + str(k), 3)
    w_vessel.append(f_thr_k)
    lbw_vessel.append(f_min)
    ubw_vessel.append(f_max)
    w0_vessel.append([0,0,0])   # "Initial guess" for the input at each opt. point
    f_thr_plot.append(f_thr_k)

    # State at collocation points
    eta_c = []
    for j in range(d):
        eta_kj = ca.MX.sym('eta_'+str(k)+'_'+str(j), 3)
        eta_c.append(eta_kj)
        w_vessel.append(eta_kj)     # Optimize at each collocation point
        lbw_vessel.append([-np.inf, -np.inf, -np.inf])      # Vessel states are unbounded (for now having no spatial constr.)
        ubw_vessel.append([np.inf, np.inf, np.inf])
        w0_vessel.append([0,0,0])       # "Initial guess" for the states at each opt. point

    # Loop over collocation points
    eta_k_end = D[0]*eta_k      # Link the states to the next opt. prob.
    for j in range(1,d+1):
        # Expression for the state derivative at the collocation point
        eta_p = C[0,j]*eta_k
        for r in range(d): eta_p = eta_p + C[r+1,j]*eta_c[r]

        # Append collocation equations
        fj_vessel, qj_vessel = f_vessel(eta_c[j-1],f_thr_k)
        g_vessel.append(h*fj_vessel - eta_p)        # See Gros 2022: Step length times x_dot - x_p
        lbg_vessel.append([0,0,0])      # -------ARE THESE CORRECT?--------
        ubg_vessel.append([0,0,0])

        # Add contribution to the end state
        eta_k_end = eta_k_end + D[j]*eta_c[j-1]

        # Add contribution to quadrature function
        # We wish to control the pendulum to its origin, so it makes sense to include X and U
        # from each collocation point in every control step
        J_vessel = J_vessel + B[j]*qj_vessel*h

    # New NLP variable for state at end of interval
    eta_k = ca.MX.sym('eta_' + str(k+1), 3)
    w_vessel.append(eta_k)
    lbw_vessel.append([-np.inf, -np.inf, -np.inf])
    ubw_vessel.append([np.inf, np.inf, np.inf])
    w0_vessel.append([0,0,0])       # ------------ETA DESIRED INSTEAD??------------
    eta_plot.append(eta_k)

    # Add equality constraint
    g_vessel.append(eta_k_end-eta_k)
    lbg_vessel.append([0,0,0])      # ------------WHAT SHOULD THESE BE?------------
    ubg_vessel.append([0,0,0])

# Concatenate vectors
w_vessel = ca.vertcat(*w_vessel)
g_vessel = ca.vertcat(*g_vessel)
eta_plot = ca.horzcat(*eta_plot)
f_thr_plot = ca.horzcat(*f_thr_plot)
w0_vessel = np.concatenate(w0_vessel)
lbw_vessel = np.concatenate(lbw_vessel)
ubw_vessel = np.concatenate(ubw_vessel)
lbg_vessel = np.concatenate(lbg_vessel)
ubg_vessel = np.concatenate(ubg_vessel)

# Create an NLP solver
nlp_vessel = {'f': J_vessel, 'x': w_vessel, 'g': g_vessel}
solver_vessel = ca.nlpsol('solver', 'ipopt', nlp_vessel)

# Function to get x and u trajectories from w
trajectories_vessel = ca.Function('trajectories_vessel', [w_vessel], [eta_plot, f_thr_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol_vessel = solver_vessel(x0=w0_vessel, lbx=lbw_vessel, ubx=ubw_vessel, lbg=lbg_vessel, ubg=ubg_vessel)
eta_opt, f_thr_opt = trajectories_vessel(sol_vessel['x'])
eta_opt = eta_opt.full() # to numpy array
f_thr_opt = f_thr_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, T_hor, N+1)
plt.figure(2)
plt.clf()       # clear current figure
plt.plot(tgrid, eta_opt[0], '--')
plt.plot(tgrid, eta_opt[1], '-')
plt.step(tgrid, np.append(np.nan, f_thr_opt[0]), '-.')
plt.xlabel('t')
plt.legend(['x_pos','y_pos','f_thr_1'])
plt.grid()
plt.show()
