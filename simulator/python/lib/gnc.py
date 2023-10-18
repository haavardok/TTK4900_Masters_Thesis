#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNC functions.

Reference: https://github.com/cybergalactic/PythonVehicleSimulator/blob/master/src/python_vehicle_simulator/lib/gnc.py
    
Author:     Håvard Olai Kopperstad (or maybe Fossen?)
"""

import numpy as np
import math

#------------------------------------------------------------------------------

def deg2rad(angleInDegrees): # Might need to check if input is float. See L1euler.m!
    """Convert angles from degrees to radians."""
    angleInRad = (math.pi/180) * angleInDegrees

    return angleInRad

#------------------------------------------------------------------------------

def rad2deg(angleInRad): # Might need to check if input is float. See L1euler.m!
    """Convert angles from radians to degrees."""
    angleInDegrees = (180/math.pi) * angleInRad

    return angleInDegrees

#------------------------------------------------------------------------------

def ssa(angle): # Might need to add support for vectors. See L1euler.m!
    """Smallest signed angle. Maps an angle in rad to the interval [-pi,pi)."""
    angle = (angle + math.pi) % (2 * math.pi) - math.pi

    return angle

#------------------------------------------------------------------------------

def Rzyx(phi,theta,psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R = np.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R

#------------------------------------------------------------------------------

def Tzyx(phi,theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)    

    try: 
        T = np.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])
        
    except ZeroDivisionError:  
        print ("Tzyx is singular for theta = +-90 degrees." )
        
    return T