"""
Example of the high-level interface in CasADi:
Solving a convex QP problem

minimize_(x,y) x**2 + y**2
s.t. x + y - 10 >= 0
"""

from casadi import *

# defining symbolic variables
x = SX.sym('x')
y = SX.sym('y')

# defining the QP problem
qp = {'x':vertcat(x,y),
      'f':x**2+y**2,
      'g':x+y-10}

# defining the solver object solver
solver = qpsol('solver', 'qpoases', qp)

# solving the QP problem without initial guess since the solution is unique
result = solver(lbg=0)
x_opt = result['x']

# printing the result
print('x_opt: ', x_opt)
