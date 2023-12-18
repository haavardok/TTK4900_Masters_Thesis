import numpy as np
import matplotlib.pyplot as plt
from lib.helpers import*


def plot_trajectories(x_optimal, x_desired, u_optimal, time_horizon):
    '''
    Plots the vessel pose error eta-eta_d, vessel velocities nu, thruster forces f, thruster angles alpha and
    the slack variables on the thruster forces on five different figures.

        Parameters:
            x_optimal (nparray): An array containing the states eta, nu and s
            x_optimal (nparray): An array containing the desired eta and nu values
            u_optimal (nparray): An array containing the inputs f and alpha

        Returns:
            None
    '''

    # Make time grids for the state and input plots
    tgrid = np.linspace(0, time_horizon, x_optimal.shape[1])
    tgrid2 = np.linspace(0, time_horizon, u_optimal.shape[1])

    # 3x1 plot of the vessel pose error eta-eta_d
    fig1, subplot1 = plt.subplots(3, sharex=True)
    subplot1[0].plot(tgrid, x_optimal[0]-x_desired[0], '-', label="$x-x_d$ [m]")
    subplot1[0].legend(loc='upper right')
    subplot1[0].grid()
    subplot1[1].plot(tgrid, x_optimal[1]-x_desired[1], '-', label="$y-y_d$ [m]")
    subplot1[1].legend(loc='upper right')
    subplot1[1].grid()
    subplot1[2].plot(tgrid, x_optimal[2]-x_desired[2], '-', label="$\\psi-\\psi_d$ [rad]")
    subplot1[2].legend(loc='lower right')
    subplot1[2].grid()
    fig1.suptitle("Vessel pose error $\\boldsymbol{\\eta}-\\boldsymbol{\\eta}_d$")
    fig1.supxlabel("t")

    # 3x1 plot of the vessel's linear and angular velocities
    fig2, subplot2 = plt.subplots(3, sharex=True)
    subplot2[0].plot(tgrid, x_optimal[3], '-', label="$u$ [m/s]")
    subplot2[0].legend(loc='upper right')
    subplot2[0].grid()
    subplot2[1].plot(tgrid, x_optimal[4], '-', label="$v$ [m/s]")
    subplot2[1].legend(loc='lower right')
    subplot2[1].grid()
    subplot2[2].plot(tgrid, x_optimal[5], '-',label="$r$ [rad/s]")
    subplot2[2].legend(loc='upper right')
    subplot2[2].grid()
    fig2.suptitle("Linear and angular velocity vector $\\boldsymbol{\\nu}$")
    fig2.supxlabel("t")

    # 3x1 plot of the thruster forces
    fig3, subplot3 = plt.subplots(3, sharex=True)
    subplot3[0].axhline(y=200, color='red', linestyle='-')
    subplot3[0].axhline(y=-200, color='red', linestyle='-')
    subplot3[0].step(tgrid2, u_optimal[0]/1000, '-', label="$f_1$ [kN]")
    subplot3[0].legend(loc='upper right')
    subplot3[0].grid()
    subplot3[1].axhline(y=200, color='red', linestyle='-')
    subplot3[1].axhline(y=-200, color='red', linestyle='-')
    subplot3[1].step(tgrid2, u_optimal[1]/1000, '-', label="$f_2$ [kN]")
    subplot3[1].legend(loc='upper right')
    subplot3[1].grid()
    subplot3[2].axhline(y=100, color='red', linestyle='-')
    subplot3[2].axhline(y=-100, color='red', linestyle='-')
    subplot3[2].step(tgrid2, u_optimal[2]/1000, '-', label="$f_3$ [kN]")
    subplot3[2].legend(loc='upper right')
    subplot3[2].grid()
    fig3.suptitle("Thruster forces $\\boldsymbol{f}$")
    fig3.supxlabel("t")

    # 2x1 plot of the azimuth thrusters' angles
    fig4, subplot4 = plt.subplots(2, sharex=True)
    subplot4[0].axhline(y=-260, color='red', linestyle='-')
    subplot4[0].axhline(y=80, color='red', linestyle='-')
    subplot4[0].step(tgrid2, u_optimal[3]*rad2deg, '-', label="$\\alpha_1$ [deg]")
    subplot4[0].legend(loc='lower right')
    subplot4[0].grid()
    subplot4[1].axhline(y=260, color='red', linestyle='-')
    subplot4[1].axhline(y=-80, color='red', linestyle='-')
    subplot4[1].step(tgrid2, u_optimal[4]*rad2deg, '-', label="$\\alpha_2$ [deg]")
    subplot4[1].legend(loc='upper right')
    subplot4[1].grid()
    fig4.suptitle("Thruster angles $\\boldsymbol{\\alpha}$")
    fig4.supxlabel("t")

    # Plot of the slack variables
    plt.figure(5)
    plt.plot(tgrid, x_optimal[6], '-')
    plt.plot(tgrid, x_optimal[7], '-')
    plt.plot(tgrid, x_optimal[8], '-')
    plt.legend(['s1','s2','s3'])
    plt.title('Slack variables for the thruster forces')
    plt.xlabel('t')
    plt.grid()


def plot_endpoints_in_NE(spatial_constraint, vessel_boundary, vessel_init_pose, vessel_desired_pose, origin_translation):

    plt.figure(figsize=(8, 8))  # width, height in inches

    # Set the limits for the plot
    # plt.xlim(0,450) # Trondheim terminal
    # plt.ylim(0,950)
    # plt.xlim(-800,200) # Hareid harbor
    # plt.ylim(-100,700)

    # Adjust vessel to given vessel pose
    vessel_boundary_init    = np.dot(vessel_boundary, R(vessel_init_pose[2]).T)
    safety_boundary_init    = vessel_boundary_init * 1.1     # 10 % dilution
    vessel_boundary_desired = np.dot(vessel_boundary, R(vessel_desired_pose[2]).T)
    safety_boundary_desired = vessel_boundary_desired * 1.1     # 10 % dilution

    # Plot the spatial constraints
    plt.plot(spatial_constraint[:,1]+origin_translation, spatial_constraint[:,0]+origin_translation, color='black')
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot the vessel with safety boundary in it's initial pose
    plt.plot(vessel_boundary_init[:,1] + vessel_init_pose[1] + origin_translation,
             vessel_boundary_init[:,0] + vessel_init_pose[0] + origin_translation, linestyle='-', color='green', label='Initial pose')
    plt.plot(safety_boundary_init[:,1] + vessel_init_pose[1] + origin_translation,
             safety_boundary_init[:,0] + vessel_init_pose[0]  +origin_translation, linestyle='--', color='green')
    
    # Plot the vessel with safety boundary in it's desired end point
    plt.plot(vessel_boundary_desired[:,1] + vessel_desired_pose[1] + origin_translation,
             vessel_boundary_desired[:,0] + vessel_desired_pose[0] + origin_translation, linestyle='-', color='blue', label='Desired pose')
    plt.plot(safety_boundary_desired[:,1] + vessel_desired_pose[1] + origin_translation,
             safety_boundary_desired[:,0] + vessel_desired_pose[0] + origin_translation, linestyle='--', color='blue')

    # Set the axis names
    plt.xlabel('East position [m]')
    plt.ylabel('North position [m]')
    plt.title("Vessel's initial and desired poses")
    plt.legend(loc='upper right')


def plot_NE_trajectory(x_optimal, spatial_constraint, vessel_boundary, origin_translation, plotting_times):

    plt.figure(figsize=(8, 8))  # width, height in inches

    # Set the limits for the plot
    # plt.xlim(0,450)
    # plt.ylim(0,950)

    # Plot the spatial constraints
    plt.plot(spatial_constraint[:,1]+origin_translation, spatial_constraint[:,0]+origin_translation, color='black', label='Convex set')
    plt.gca().set_aspect('equal', adjustable='box')

    # for i in range(len(x_optimal[0])):
    #     if i==0:
    #         # Plot the initial pose
    #         vessel_boundary_initial = np.dot(vessel_boundary, R(x_optimal[2,i]).T)
    #         safety_boundary_initial = vessel_boundary_initial * 1.1     # 10 % dilution
    #         plt.plot(vessel_boundary_initial[:,1] + x_optimal[1,i] + origin_translation,
    #                  vessel_boundary_initial[:,0] + x_optimal[0,i] + origin_translation, linestyle='-', color='green', label='Initial pose')
    #         plt.plot(safety_boundary_initial[:,1] + x_optimal[1,i] + origin_translation,
    #                  safety_boundary_initial[:,0] + x_optimal[0,i] + origin_translation, linestyle='--', color='green')
    #     elif i==len(x_optimal[0])-1:
    #         # Plot the final pose
    #         vessel_boundary_final = np.dot(vessel_boundary, R(x_optimal[2,i]).T)
    #         safety_boundary_final = vessel_boundary_final * 1.1     # 10 % dilution
    #         plt.plot(vessel_boundary_final[:,1] + x_optimal[1,i] + origin_translation,
    #                  vessel_boundary_final[:,0] + x_optimal[0,i] + origin_translation, linestyle='-', color='blue', label='Final pose')
    #         plt.plot(safety_boundary_final[:,1] + x_optimal[1,i] + origin_translation,
    #                  safety_boundary_final[:,0] + x_optimal[0,i] + origin_translation, linestyle='--', color='blue')
    #     else:
    #         # Plot poses on the trajectory
    #         vessel_boundary_underways = np.dot(vessel_boundary, R(x_optimal[2,i]).T)
    #         safety_boundary_underways = vessel_boundary_underways * 1.1     # 10 % dilution
    #         plt.plot(vessel_boundary_underways[:,1] + x_optimal[1,i] + origin_translation,
    #                  vessel_boundary_underways[:,0] + x_optimal[0,i] + origin_translation, linestyle='-', color='black')
    #         plt.plot(safety_boundary_underways[:,1] + x_optimal[1,i] + origin_translation,
    #                  safety_boundary_underways[:,0] + x_optimal[0,i] + origin_translation, linestyle='--', color='black')

    # Plot the initial pose
    vessel_boundary_initial = np.dot(vessel_boundary, R(x_optimal[2,0]).T)
    safety_boundary_initial = vessel_boundary_initial * 1.1     # 10 % dilution
    # plt.plot(vessel_boundary_initial[:,1] + x_optimal[1,0] + origin_translation,
    #          vessel_boundary_initial[:,0] + x_optimal[0,0] + origin_translation, linestyle='-', color='green', label='Initial pose')
    plt.plot(safety_boundary_initial[:,1] + x_optimal[1,0] + origin_translation,
             safety_boundary_initial[:,0] + x_optimal[0,0] + origin_translation, linestyle='--', color='green', label='Initial pose')
    
    # Plot the final pose
    vessel_boundary_final = np.dot(vessel_boundary, R(x_optimal[2,len(x_optimal[0])-1]).T)
    safety_boundary_final = vessel_boundary_final * 1.1     # 10 % dilution
    # plt.plot(vessel_boundary_final[:,1] + x_optimal[1,len(x_optimal[0])-1] + origin_translation,
    #          vessel_boundary_final[:,0] + x_optimal[0,len(x_optimal[0])-1] + origin_translation, linestyle='-', color='blue', label='Final pose')
    plt.plot(safety_boundary_final[:,1] + x_optimal[1,len(x_optimal[0])-1] + origin_translation,
             safety_boundary_final[:,0] + x_optimal[0,len(x_optimal[0])-1] + origin_translation, linestyle='--', color='blue', label='Final pose')

    for i in plotting_times:
        # Plot poses on the trajectory
        vessel_boundary_underways = np.dot(vessel_boundary, R(x_optimal[2,i]).T)
        safety_boundary_underways = vessel_boundary_underways * 1.1     # 10 % dilution
        # plt.plot(vessel_boundary_underways[:,1] + x_optimal[1,i] + origin_translation,
        #          vessel_boundary_underways[:,0] + x_optimal[0,i] + origin_translation, linestyle='-', color='black')
        plt.plot(safety_boundary_underways[:,1] + x_optimal[1,i] + origin_translation,
                 safety_boundary_underways[:,0] + x_optimal[0,i] + origin_translation, linestyle='--', color='black')

    # Plot the trajectory between the initial and the final pose
    plt.plot(x_optimal[1]+origin_translation, x_optimal[0]+origin_translation, '-')

    # Set names on axes
    plt.xlabel('East position [m]')
    plt.ylabel('North position [m]')
    plt.title("Vessel trajectory")
    plt.legend(fontsize='small',loc='upper right')
