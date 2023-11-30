import math
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

#######################################################################
# Defining functions
#######################################################################

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
f1_min = -1/30 * m*g; f1_max = 1/30 * m*g               # azimuth thruster 1 force saturation (kN) -------------------------------CHECK THIS-------------------------------------------
f2_min = -1/30 * m*g; f2_max = 1/30 * m*g               # azimuth thruster 2 force saturation (kN) -------------------------------CHECK THIS-------------------------------------------
f3_min = -1/60 * m*g; f3_max = 1/60 * m*g               # bow tunnel thruster force saturation (kN) ------------------------------CHECK THIS-------------------------------------------
alpha1_min = -170*math.pi/180; alpha1_max = 170*math.pi/180                 # azimuth thruster 1 angle saturation (rad)
alpha2_min = -170*math.pi/180; alpha2_max = 170*math.pi/180                 # azimuth thruster 1 angle saturation (rad)
alpha3 = math.pi/2                                  # bow tunnel thruster constant angle (rad)
alpha1_dot_max = (360*math.pi/180)/30               # azimuth thruster max turnaround time (rad/s) ------------------------------CHECK THIS-------------------------------------------
alpha2_dot_max = (360*math.pi/180)/30
vessel_safety_margin = 1.1                          # safety margin M of set Sb w.r.t. Sv

# Weighting matrices
Q_eta = np.diag([1e4,1e4,1e7])                            # weighting matrix for position and Euler angle vector eta ---------------------CHECK THESE THREE----------------------
Q_nu = np.diag([1,1e-2,1e-1])                             # weighting matrix for linear and angular velocity vector nu
Q_s = np.diag([1e3,1e3,1e3])                              # weighting matrix for the slack variables
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

S_v = np.array([                                    # vertices for the convex set Sv in {b} defining the vessel
    [-38.1, -9.4],
    [-38.1,  9.4],
    [ 21.8,  9.4],
    [ 38.1,  0.0],
    [ 21.8, -9.4] ])

S_b = S_v * vessel_safety_margin                    # vertices for the convex set Sb in {b} defining the vessel boundary (10 % dilution of Sv)

S_s = np.array([                                    # vertices for the convex set Ss in {n} defining the harbour area
    [-100, -100],
    [ -60,  150],
    [   0,  -65],
    [   0,  500],
    [  85,  150],
    [ 175,  500] ])

A_s = np.array([                                    # spatial constraints for convex set (Hurtigruta terminal)
    [ 0.0000,  1.0000],
    [-0.9856,  0.1690],
    [-0.9874,  0.1580],
    [ 0.9685, -0.2490],
    [ 0.9300, -0.3677],
    [ 0.3304, -0.9439]])

b_s = np.array([600, 2.8161, 0, 116.9109, 80.1280, 0])

A_s_no_eta_d = np.array([                            # spatial constraints for convex set (Hurtigruta terminal)
    [ 0.0000,  1.0000],
    [ 0.9685, -0.2490],
    [ 0.9300, -0.3677],
    [-0.9856,  0.1690],
    [-0.9874,  0.1580],
    [ 0.3304, -0.9439]])

b_s_no_eta_d = np.array([500, 44.9657, 23.8978, 84.4819, 82.9450, 61.3508])


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

s1 = ca.SX.sym('s1')
s2 = ca.SX.sym('s2')
s3 = ca.SX.sym('s3')
s  = ca.vertcat(s1,s2,s3)

alpha1 = ca.SX.sym('alpha1')
alpha2 = ca.SX.sym('alpha2')
alpha  = ca.vertcat(alpha1,alpha2)

alpha1_dot = ca.SX.sym('alpha1_dot')
alpha2_dot = ca.SX.sym('alpha2_dot')
alpha_dot  = ca.vertcat(alpha1_dot,alpha2_dot)

X = ca.vertcat(eta,nu,s)                  # NLP state vector
U = ca.vertcat(f_thr,alpha)             # NLP input vector

x_init = [50,300,-math.pi/2, 0,0,0, 0,0,0]     # x(0)=50, y(0)=300, psi(0)=-pi/2, u(0)=0, v(0)=0, r(0)=0
u_init = [0,0,0, 0,0]
u_min  = [f1_min,f2_min,f3_min, alpha1_min,alpha2_min]
u_max  = [f1_max,f2_max,f3_max, alpha1_max,alpha2_max]

# Model equations
eta_dot = ca.mtimes(J_rot(psi), nu)
nu_dot = ca.mtimes(ca.inv(M_vessel), ca.mtimes(T(alpha), f_thr) + s - ca.mtimes(D_vessel, nu))

# Objective term
obj_det = ca.det(ca.mtimes(T(alpha), ca.mtimes(ca.inv(W), T(alpha).T)))     # singular configuration cost
objective = (ca.mtimes((eta).T, ca.mtimes(Q_eta, (eta))) +                  # ----------------------------HOW TO HANDLE ETA_D---------------------
# objective = (ca.mtimes((eta-eta_d).T, ca.mtimes(Q_eta, (eta-eta_d))) +      # continuous time objective
             ca.mtimes(nu.T, ca.mtimes(Q_nu, nu)) +
             ca.mtimes(f_thr.T, ca.mtimes(R_f, f_thr)) +
             rho / (epsilon + obj_det) +
             ca.mtimes(s.T, ca.mtimes(Q_s, s)))

# Continuous time dynamics
f = ca.Function('f', [X, U], [eta_dot, nu_dot, objective], ['X', 'U'], ['eta_dot', 'nu_dot', 'objective'])

# Spatial constraints
# spatial_constraints = []
# for vertex in range(S_b.shape[0]):
#     constr = ca.mtimes(A_s_no_eta_d, ca.mtimes(R_rot(psi), S_b[vertex]) + X[0:2])
#     spatial_constraints.append(constr)

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
        lbw.append([-np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf])      # Vessel states are unbounded (for now having no spatial constr.)
        ubw.append([np.inf,np.inf,np.inf, np.inf,np.inf,np.inf, np.inf,np.inf,np.inf])
        w0.append([0,0,0, 0,0,0, 0,0,0])                                           # initial guess for the states

    # Loop over collocation points
    Xk_end = D[0]*Xk                        # Link the states to the next opt. prob.
    for j in range(1,d+1):
        # Expression for the state derivative at the collocation point
        xp = C[0,j]*Xk
        for r in range(d): xp = xp + C[r+1,j]*Xc[r]

        # Append collocation equations
        fj1, fj2, qj = f(Xc[j-1],Uk)
        fj = ca.vertcat(fj1,fj2,[0,0,0])            # Need to concatenate eta_dot and nu_dot into xdot -------------------------THIS MAY BE INCORRECT-------------------------------
        g.append(h*fj - xp)                 # See Gros 2022: Step length times x_dot - x_p
        lbg.append([0,0,0, 0,0,0, 0,0,0])          # -------------------------ARE THESE CORRECT?-------------------------------
        ubg.append([0,0,0, 0,0,0, 0,0,0])

        # Add contribution to the end state
        Xk_end = Xk_end + D[j]*Xc[j-1]

        # Add contribution to quadrature function
        J = J + B[j]*qj*h

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym('X_' + str(k+1), X.size1())
    w.append(Xk)
    lbw.append([-np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf])
    ubw.append([np.inf,np.inf,np.inf, np.inf,np.inf,np.inf, np.inf,np.inf,np.inf])
    w0.append([0,0,0, 0,0,0, 0,0,0])               # -------------------------THIS MAY BE INCORRECT-------------------------------
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end-Xk)
    lbg.append([0,0,0, 0,0,0, 0,0,0])              # -------------------------THIS MAY BE INCORRECT-------------------------------
    ubg.append([0,0,0, 0,0,0, 0,0,0])

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
options = {'ipopt': {'max_iter':3000}}
solver = ca.nlpsol('solver', 'ipopt', prob, options)

# Function to get x and u trajectories from w
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, time_horizon, N+1)
tgrid2 = np.linspace(0, time_horizon, N)

fig1, subplot1 = plt.subplots(3, sharex=True)
subplot1[0].plot(tgrid, x_opt[0], '-', label="$x$ [m]")
subplot1[0].legend(loc='upper right')
# subplot1[0].set_ylim([0,200])
subplot1[0].grid()
subplot1[1].plot(tgrid, x_opt[1], '-', label="$y$ [m]")
subplot1[1].legend(loc='upper right')
# subplot1[1].set_ylim([-350,0])
subplot1[1].grid()
subplot1[2].plot(tgrid, x_opt[2], '-', label="$\\psi$ [rad]")
subplot1[2].legend(loc='upper right')
# subplot1[2].set_ylim([-3.5,0])
subplot1[2].grid()
fig1.suptitle("Position and angle vector $\\boldsymbol{\\eta}$")
fig1.supxlabel("t")

fig2, subplot2 = plt.subplots(3, sharex=True)
subplot2[0].plot(tgrid, x_opt[3], '-', label="$u$ [m/s]")
subplot2[0].legend(loc='upper right')
subplot2[0].grid()
subplot2[1].plot(tgrid, x_opt[4], '-', label="$v$ [m/s]")
subplot2[1].legend(loc='upper right')
subplot2[1].grid()
subplot2[2].plot(tgrid, x_opt[5], '-',label="$r$ [rad/s]")
subplot2[2].legend(loc='upper right')
subplot2[2].grid()
fig2.suptitle("Linear and angular velocity vector $\\boldsymbol{\\nu}$")
fig2.supxlabel("t")

fig3, subplot3 = plt.subplots(3, sharex=True)
subplot3[0].step(tgrid2, u_opt[0], '-', label="$f_1$ [kN]")
subplot3[0].legend(loc='upper right')
subplot3[0].grid()
subplot3[1].step(tgrid2, u_opt[1], '-', label="$f_2$ [kN]")
subplot3[1].legend(loc='upper right')
subplot3[1].grid()
subplot3[2].step(tgrid2, u_opt[2], '-', label="$f_3$ [kN]")
subplot3[2].legend(loc='upper right')
subplot3[2].grid()
fig3.suptitle("Thruster forces $\\boldsymbol{f}$")
fig3.supxlabel("t")

fig4, subplot4 = plt.subplots(2, sharex=True)
subplot4[0].plot(tgrid2, u_opt[3], '-', label="$\\alpha_1$ [rad]")
subplot4[0].legend(loc='upper right')
subplot4[0].grid()
subplot4[1].plot(tgrid2, u_opt[4], '-', label="$\\alpha_2$ [rad]")
subplot4[1].legend(loc='upper right')
subplot4[1].grid()
fig4.suptitle("Thruster angles $\\boldsymbol{\\alpha}$")
fig4.supxlabel("t")

plt.figure(5)
plt.plot(tgrid, x_opt[6], '-')
plt.plot(tgrid, x_opt[7], '-')
plt.plot(tgrid, x_opt[8], '-')
plt.xlabel('t')
plt.legend(['s1','s2','s3'])
plt.grid()

plt.show()

# plt.plot(tgrid, x_opt[0], '--')
# plt.plot(tgrid, x_opt[1], '-')
# plt.plot(tgrid, x_opt[2], '-')
# plt.xlabel('t')
# plt.legend(['x','y','psi'])
# plt.grid()

# plt.figure(2)
# plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
# plt.step(tgrid, np.append(np.nan, u_opt[1]), '-.')
# plt.step(tgrid, np.append(np.nan, u_opt[2]), '-.')
# plt.xlabel('t')
# plt.legend(['f1','f2','f3'])
# plt.grid()
# plt.show()
