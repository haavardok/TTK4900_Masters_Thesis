from lib.plot import*
from lib.helpers import*

harbor_constraint = np.array([[0,0],
                              [-8,20],
                              [10,75],
                              [900,250],
                              [900,40],
                              [0,0]])

S_v = np.array([                                    # vertices for the convex set Sv in {b} defining the vessel
    [-38.1, -9.4],
    [-38.1,  9.4],
    [ 21.8,  9.4],
    [ 38.1,  0.0],
    [ 21.8, -9.4],
    [-38.1, -9.4] ])

S_b = S_v * 1.1

plot_NE_trajectory(harbor_constraint, S_v, [600,125.75,math.pi], 0)

