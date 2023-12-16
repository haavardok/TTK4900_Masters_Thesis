import math
import numpy as np
import casadi as ca

from lib.helpers import*
from lib.plot import *


#######################################################################
# Defining the ship parameters
#######################################################################

# Vessel parameters
L = 76.2                                            # vessel length (m)
g = 9.8                                             # gravitational acceleration (m/s^2)
m = 6000e3                                          # vessel mass (kg)
lx1 = -35; ly1 = 7                                  # azimuth thruster 1 position (m)
lx2 = -35; ly2 = -7                                 # azimuth thruster 2 position (m)
lx3 = 35;  ly3 = 0                                  # bow tunnel thruster position (m)
alpha3 = ca.pi/2                                  # bow tunnel thruster constant angle (rad) in {b}
f1_min = -1/30 * m; f1_max = 1/30 * m               # azimuth thruster 1 force saturation (kN) -------------------------------CHECK THIS-------------------------------------------
f2_min = -1/30 * m; f2_max = 1/30 * m               # azimuth thruster 2 force saturation (kN) -------------------------------CHECK THIS-------------------------------------------
f3_min = -1/60 * m; f3_max = 1/60 * m               # bow tunnel thruster force saturation (kN) ------------------------------CHECK THIS-------------------------------------------
alpha1_min = -170*ca.pi/180; alpha1_max = 170*ca.pi/180                 # azimuth thruster 1 angle saturation (rad)
alpha2_min = -170*ca.pi/180; alpha2_max = 170*ca.pi/180                 # azimuth thruster 1 angle saturation (rad)
alpha1_dot_max = (360*ca.pi/180)/30               # azimuth thruster max turnaround time (rad/s) ------------------------------CHECK THIS-------------------------------------------
alpha2_dot_max = (360*ca.pi/180)/30
delta_alpha1_max = alpha1_dot_max * 10              # discrete time azimuth thruster turnaround time (cont. time * step size)
delta_alpha2_max = alpha2_dot_max * 10

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

S_v = np.array([                                    # vertices for the convex set Sv in {b} defining the vessel
    [-38.1, -9.4],
    [-38.1,  9.4],
    [ 21.8,  9.4],
    [ 38.1,  0.0],
    [ 21.8, -9.4] ]).T


#######################################################################
# Defining NLP weighting matrices and spatial constraints
#######################################################################

# Weighting matrices
Q_eta = np.diag([1e4,1e4,1e7])                      # weighting matrix for position and Euler angle vector eta
Q_nu = np.diag([0,1,1])                             # weighting matrix for linear and angular velocity vector nu
Q_s = np.diag([1,1,1])                        # weighting matrix for the slack variables
R_f = np.diag([1e-7,1e-7,1e-7])                     # weighting matrix for force vector f
R_alpha = np.diag([1e-7,1e-7])                      # weighting matrix for azimuth thruster turn rate
W = np.diag([1,1,1])                                # thruster weighting matrix
epsilon = 1e-3                                      # small constant to avoid division by 0
rho = 1                                             # thruster weighting of maneuverability

vessel_safety_margin = 1.1                          # safety margin M of set Sb w.r.t. Sv

S_b = S_v * vessel_safety_margin                    # vertices for the convex set Sb in {b} defining the vessel boundary (10 % dilution of Sv)

harbor_constraint = np.array([[0,0],                # vertices for the convex set in {n} defining the Trondheim harbor
                              [-8,22],
                              [10,75],
                              [750,250],
                              [750,40],
                              [0,0]])

A_s = np.array([                                    # spatial constraints for convex set (Hurtigruta terminal)
    [ 1.0000,  0.0000],
    [-0.2301,  0.9732],
    [ 0.0533, -0.9986],
    [-0.9469,  0.3216],
    [-0.9398, -0.3417]])

b_s = np.array([750, 70.6855, 0, 14.6499, 0])


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

u_d  = ca.SX.sym('u_d')
v_d  = ca.SX.sym('v_d')
r_d  = ca.SX.sym('r_d')
nu_d = ca.vertcat(u_d,v_d,r_d)

f1    = ca.SX.sym('f1')
f2    = ca.SX.sym('f2')
f3    = ca.SX.sym('f3')
f_thr = ca.vertcat(f1,f2,f3)

s1 = ca.SX.sym('s1')
s2 = ca.SX.sym('s2')
s3 = ca.SX.sym('s3')
s  = ca.vertcat(s1,s2,s3)

alpha1 = ca.SX.sym('alpha1')
alpha2 = ca.SX.sym('alpha2')
alpha  = ca.vertcat(alpha1,alpha2)

alpha1_0 = ca.SX.sym('alpha1_0')
alpha2_0 = ca.SX.sym('alpha2_0')
alpha_0  = ca.vertcat(alpha1_0,alpha2_0)            # previous input of azimuth angles

X   = ca.vertcat(eta,nu,s)                          # NLP state vector
U   = ca.vertcat(f_thr,alpha)                       # NLP input vector
X_d = ca.vertcat(eta_d,nu_d)                        # NLP desired state vector

x_init    = [600,120,-ca.pi, 0,0,0, 0,0,0]      # x(0)=500, y(0)=175, psi(0)=-pi/2, u(0)=0, v(0)=0, r(0)=0, s1(0)=0, s2(0)=0, s3(0)=0
x_desired = [200,21.5,0.05328285155969, 0,0,0]
u_init    = [0,0,0, 0,0]
u_min     = [f1_min,f2_min,f3_min, alpha1_min,alpha2_min]
u_max     = [f1_max,f2_max,f3_max, alpha1_max,alpha2_max]

# Plot initial pose of the problem
vessel = np.array([
            [-38.1, -9.4],
            [-38.1,  9.4],
            [ 21.8,  9.4],
            [ 38.1,  0.0],
            [ 21.8, -9.4],
            [-38.1, -9.4] ])
plot_endpoints_in_NE(harbor_constraint, vessel, x_init[:3], x_desired[:3], 100)
plt.show()

# Model equations
eta_dot = ca.mtimes(J_rot(psi), nu)
nu_dot = ca.mtimes(ca.inv(M_vessel), ca.mtimes(T(alpha), f_thr) + s - ca.mtimes(D_vessel, nu))

# Objective term
obj_det = ca.det(ca.mtimes(T(alpha), ca.mtimes(ca.inv(W), T(alpha).T)))     # singular configuration cost
# objective = (ca.mtimes((eta-eta_d).T, ca.mtimes(Q_eta, (eta-eta_d))) +      # continuous time objective
#              ca.mtimes((nu-nu_d).T, ca.mtimes(Q_nu, (nu-nu_d))) +
#              ca.mtimes(s.T, ca.mtimes(Q_s, s)) +
#              ca.mtimes(f_thr.T, ca.mtimes(R_f, f_thr)) +
#              ca.mtimes((alpha-alpha_0).T, ca.mtimes(R_alpha, (alpha-alpha_0))) +
#              rho / (epsilon + obj_det))

objective = (ca.mtimes((eta[:2]-eta_d[:2]).T, ca.mtimes(Q_eta[:2,:2], (eta[:2]-eta_d[:2]))) +      # continuous time objective
             ca.mtimes(ssa(eta[2]-eta_d[2]).T, ca.mtimes(Q_eta[2,2], ssa(eta[2]-eta_d[2]))) +
             ca.mtimes((nu-nu_d).T, ca.mtimes(Q_nu, (nu-nu_d))) +
             ca.mtimes(s.T, ca.mtimes(Q_s, s)) +
             ca.mtimes(f_thr.T, ca.mtimes(R_f, f_thr)) +
             ca.mtimes((alpha-alpha_0).T, ca.mtimes(R_alpha, (alpha-alpha_0))) +
             rho / (epsilon + obj_det))

# Continuous time dynamics
f = ca.Function('f', [X, U, X_d, alpha_0], [eta_dot, nu_dot, objective], ['X', 'U', 'X_d', 'alpha_0'], ['eta_dot', 'nu_dot', 'objective'])

# Spatial constraints
spatial_constraints = A_s @ ((R_rot(psi) @ S_b).T + ca.repmat(ca.horzcat(x, y), S_b.shape[1], 1)).T
f_harbor = ca.Function('f_harbor', [X], [spatial_constraints], ['X'], ['spatial_constraints'])

# Control discretization
N = 30                  # number of control intervals
h = time_horizon / N    # step time

# NLP formulation: parameters we wish to minimize the cost w.r.t.
w=[]                    # decision/optimization variables
w0 = []                 # decision variables, initial guess
lbw = []                # decision variables lower bound
ubw = []                # decision variables upper bound
J = 0                   # sum of objective variables (x0 + ... + xN + u1 + ... + uN)
g=[]                    # constraint variables
lbg = []                # constraint lower bounds
ubg = []                # constraint upper bounds

# For plotting x and u given w
x_plot = []
u_plot = []

# "Lift" initial conditions
Xk = ca.MX.sym('X0', X.size1())
w.append(Xk)
lbw.append(x_init)      # Equality constraint so bounded upper and lower
ubw.append(x_init)
w0.append(x_init)       # x(0)=150, y(0)=-325, psi(0)=-pi, u(0)=0, v(0)=0, r(0)=0
x_plot.append(Xk)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k), U.size1())
    w.append(Uk)
    lbw.append(u_min)
    ubw.append(u_max)
    w0.append(u_init)   # initial guess for the inputs
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), X.size1())
        Xc.append(Xkj)
        w.append(Xkj)                                                       # Optimize at each collocation point
        lbw.append([-ca.inf,-ca.inf,-ca.inf, -2*knot2ms,-2*knot2ms,-ca.inf, -ca.inf,-ca.inf,-ca.inf])      # Vessel states are bounded by inequality constraint
        ubw.append([ca.inf,ca.inf,ca.inf, 8*knot2ms,2*knot2ms,ca.inf, ca.inf,ca.inf,ca.inf])
        w0.append([0,0,0, 0,0,0, 0,0,0])                                           # initial guess for the states

    # Loop over collocation points
    Xk_end = D[0]*Xk                        # Link the states to the next opt. prob.
    for j in range(1,d+1):
        # Expression for the state derivative at the collocation point
        xp = C[0,j]*Xk
        for r in range(d): xp = xp + C[r+1,j]*Xc[r]

        # Append collocation equations
        if k==0:
            fj1, fj2, qj = f(Xc[j-1],Uk,x_desired,[0,0])
            fj = ca.vertcat(fj1,fj2,[0,0,0])
            g.append(h*fj - xp)
            lbg.append([0,0,0, 0,0,0, 0,0,0])
            ubg.append([0,0,0, 0,0,0, 0,0,0])
            g.append(Uk[3:])
            lbg.append([-delta_alpha1_max,-delta_alpha2_max])
            ubg.append([ delta_alpha1_max, delta_alpha2_max])
        else:
            index_Uk_previous = len(w)-9
            fj1, fj2, qj = f(Xc[j-1],Uk,x_desired,w[index_Uk_previous][3:])
            fj = ca.vertcat(fj1,fj2,[0,0,0])            # Need to concatenate eta_dot and nu_dot into xdot -------------------------THIS MAY BE INCORRECT-------------------------------
            g.append(h*fj - xp)                 # See Gros 2022: Step length times x_dot - x_p
            lbg.append([0,0,0, 0,0,0, 0,0,0])          # -------------------------ARE THESE CORRECT?-------------------------------
            ubg.append([0,0,0, 0,0,0, 0,0,0])
            g.append(Uk[3:]-w[index_Uk_previous][3:])
            lbg.append([-delta_alpha1_max,-delta_alpha2_max])
            ubg.append([ delta_alpha1_max, delta_alpha2_max])

        # Add contribution to the end state
        Xk_end = Xk_end + D[j]*Xc[j-1]

        # Add contribution to quadrature function
        J = J + B[j]*qj*h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym('X_' + str(k+1), X.size1())
    w.append(Xk)
    lbw.append([-ca.inf,-ca.inf,-ca.inf, -2*knot2ms,-2*knot2ms,-ca.inf, -ca.inf,-ca.inf,-ca.inf])
    ubw.append([ca.inf,ca.inf,ca.inf, 8*knot2ms,2*knot2ms,ca.inf, ca.inf,ca.inf,ca.inf])
    w0.append([0,0,0, 0,0,0, 0,0,0])               # -------------------------THIS MAY BE INCORRECT-------------------------------
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end-Xk)
    lbg.append([0,0,0, 0,0,0, 0,0,0])              # -------------------------THIS MAY BE INCORRECT-------------------------------
    ubg.append([0,0,0, 0,0,0, 0,0,0])

    # Add inequality constraints for the convex set
    spatial_constraint = f_harbor(Xk)
    for i in range(S_b.shape[1]):
        g.append(spatial_constraint[:,i])
        lbg.append([-ca.inf,-ca.inf,-ca.inf,-ca.inf,-ca.inf])
        ubg.append(b_s)

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)
w0 = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# Create an NLP solver
prob = {'f': J, 'x': w, 'g': g}
options = {'ipopt': {'max_iter':3000}, 'expand':True}
# p_opts = {'expand':True}                # Replace MX with SX expressions in problem formulation to cut eval time on nlp_hess_l and nlp_jac_g
solver = ca.nlpsol('solver', 'ipopt', prob, options)



# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
move_origin_of_plot = 100               # moving origin in North and East to better see plot
plotting_times = [10,15,21]
plot_trajectories(x_opt, x_desired, u_opt, time_horizon)
plot_NE_trajectory(x_opt, harbor_constraint, vessel, move_origin_of_plot, plotting_times)

# Show all plots
plt.show()
