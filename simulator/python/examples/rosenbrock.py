"""
Example of solving the Rosenbrock problem with CasADi

minimize_(x,y,z) x**2 + 100*z**2
s.t. z + (1 - x)**2 - y = 0
"""

from casadi import *

# defining symbolic variables
x = SX.sym('x')
y = SX.sym('y')
z = SX.sym('z')

# defining the nonlinear problem
nlp = {'x': vertcat(x,y,z),
       'f': x**2+100*z**2,
       'g': z+(1-x)**2-y}

# defining the solver object solver
solver = nlpsol('solver', 'ipopt', nlp)

# solving the NLP problem using x0 as inital guess
result = solver(x0=[2.5,3.0,0.75], lbg=0, ubg=0)
x_opt = result['x']

# printing the result
print('x_opt: ', x_opt)
