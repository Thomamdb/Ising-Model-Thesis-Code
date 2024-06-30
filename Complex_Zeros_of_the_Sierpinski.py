import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import time


"""
Code to find the complex zeros of the partition function for the graphs that converge to the Sierpinski Gasket
"""
#Polynomials are worked with in python by using numpy arrays. This will be extensively used in this code.


def partition_zero(x_1, x_2, x_3, b_val):
    """
    This function finds for the first Sierpinski graph what Z^{x_1 x_2 x_3}_0 is.
    """
    Z_0 = ([0] *(x_1 + x_2 + x_3 + 1)) #This creates a list of size x_1 + x_2 + x_3 + 1 with each thing value 0.
    Z_0[0] = b_val**(abs(x_1 - x_2) + abs(x_2 - x_3) + abs(x_3 - x_1)) #Setting the coefficient of first term
    #The first value of our list is the coefficient to lambda^(x_1 + x_2 + x_3)
    Z_0 = np.array(Z_0)
    return Z_0

#Ordering of Z_list is Z^000 Z^100 Z^110 Z^111
def Z_next(x_1, x_2, x_3, Z_list):
    """
    This function outputs Z^{x_1 x_2 x_3}_{n+1} in terms of the Z_n given in the Z_list.
    """
    Z = np.array([0])
    for y_1 in [0, 1]:
        for y_2 in [0, 1]:
            for y_3 in [0, 1]:
                Z_first = Z_list[x_1 + y_2 + y_1]
                Z_second = Z_list[y_2 + x_2 + y_3]
                Z_third = Z_list[y_1 + y_3 + x_3]
                #print(f'The input {x_1}{x_2}{x_3} gives for {y_1}{y_2}{y_3} gives terms{ Z_first}, {Z_second}, {Z_third}')
                #This holds because Z^001 = Z^010 = Z^100 for all our graphs.
                lambda_power = ([0] * (y_1 + y_2 + y_3 + 1))
                lambda_power[0] = 1 #This makes the polynomial lambda^{y_1 + y_2 + y_3}
                Z_term = np.polymul(Z_first, np.polymul(Z_second, Z_third)) #Multiplying the three polynomials together.
                Z_term, remainder = np.polydiv(Z_term, lambda_power) #Returns the quotient and remainder
                Z_term = np.array(Z_term)
                Z = np.polyadd(Z, Z_term)
    return Z

def Z_iterator(N, Z_iteration_list, Counter):
    """This function is an iterative function that finds Z_N^000, Z_N^100, Z_N^110, Z_N^111 from Z_0."""
    Counter = Counter + 1 #Counter to count how many iterations we have done. We put in Counter is 1 as we start with graph with 1.
    # If statement means that we stop with the recursion at N. Else stops when we have reached Counter = N.
    if Counter < N + 1:
        #The following code finds all the Z_{Counter}^{x_1x_2x_3} values using the initial condition Z_iteration_list.
        Z_0 = Z_next(0, 0, 0, Z_iteration_list)
        Z_1 = Z_next(1, 0, 0, Z_iteration_list)
        Z_2 = Z_next(1, 1, 0, Z_iteration_list)
        Z_3 = Z_next(1, 1, 1, Z_iteration_list)
        Z_list = np.array([Z_0, Z_1, Z_2, Z_3])
        return Z_iterator(N, Z_list, Counter)
    else:
        return Z_iteration_list #This returns Z_N = [Z_N^000, Z_N^100, Z_N^110, Z_N^111]. Note that each Z_N^xyz is a np.array/polynomial.

def Zeros_finder(Partition_functions):
    """
    This function finds the roots of the partition function Z_N = Z_N^000 + 3 Z_N^100 + 3 Z_N^110 + Z_N^111.
    Input list is Z_N = [Z_N^000, Z_N^100, Z_N^110, Z_N^111].
    """
    Z = np.polyadd(np.polyadd(Partition_functions[0], 3 * Partition_functions[1]), np.polyadd(3 * Partition_functions[2], Partition_functions[3]))
    Roots = np.roots(Z) #This root function takes a long time for polynomials that have degree larger than 100. These are comments below and above
    x_vals = [ele.real for ele in Roots]
    y_vals = [ele.imag for ele in Roots]
    return Roots, x_vals, y_vals #Roots, real values and imaginary parts of the roots.

def zeros_plotter(Partition_functions, n_val, b_val):
    """
    This function just plots the roots of the input function Partition_functions.
    """
    Roots, x_vals, y_vals = Zeros_finder(Partition_functions)
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.plot(x_vals, y_vals, 'ro', markersize = 3)
    ax.set_ylabel('$\mathrm{Im}$', loc = 'top')
    ax.set_xlabel('$\mathrm{Re}$', loc = 'right')
    ax.set_aspect('equal')
    ax.spines.left.set_position('zero') #The following four lines makes the axis go through (0,0)
    ax.spines.right.set_color('none')
    ax.spines.bottom.set_position('zero')
    ax.spines.top.set_color('none')
    # text_box_content = f'n = {n_val} \nb = {b_val:.2f}' #This plots a text box with the values n_val and b_val
    # plt.text(0.75, 1, text_box_content, bbox=dict(facecolor='white', alpha=0.5))
    
    t = np.linspace(0, np.pi * 2, 400) #This code is to plot the unit circle.
    ax.plot(np.cos(t), np.sin(t), linewidth = 1)
    plt.show()
    return None

def Complex_Plane_Plot(b, n):
    """
    This function brings everything together and allows me to set the initial conditions b and n and then get a plot
    """
    Z_0 = partition_zero(0, 0, 0, b)
    Z_1 = partition_zero(1, 0, 0, b)
    Z_2 = partition_zero(1, 1, 0, b)
    Z_3 = partition_zero(1, 1, 1, b)
    Partition_0 = np.array([Z_0, Z_1, Z_2, Z_3])
    Partition_Function_N = Z_iterator(n, Partition_0, 1)
    #print(Partition_Function_N)
    #print('The Partition Function_N is', Partition_Function_N)
    zeros_plotter(Partition_Function_N, n, b)
    return None

b_val = float(input("What is the value of b you would like? \n"))


n_val = input("What is the value of n you would like? \n")
if n_val.isnumeric() == True:
    n_val = int(n_val)
else:
    print("The number you gave is not an integer.")
    n_val = int(input("What is the value of n you would like? \n"))
Complex_Plane_Plot(b_val, n_val)

def animator(b_min, b_max, intervals, N):
    """
    This function animates how the roots in the plane change when b changes from b_min to b_max
    """
    #The following code creates a figure with the unit circle on which we will plot the roots for different b values.
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.set_ylabel('$\mathrm{Im}$', loc = 'top')
    ax.set_xlabel('$\mathrm{Re}$', loc = 'right')
    ax.set_aspect('equal')
    ax.spines.left.set_position('zero')
    ax.spines.right.set_color('none')
    ax.spines.bottom.set_position('zero')
    ax.spines.top.set_color('none')
    t = np.linspace(0, np.pi * 2, 100)
    ax.plot(np.cos(t), np.sin(t), linewidth = 1)

    #We use an empty list as we will soon use the grabframe command.
    list_plot, = plt.plot([], [], 'ro', markersize = 3)
    metadata = dict(title = 'Complex_zeros_Gif', artist = 'Thomas de Boer') #The information about the animation that is created
    writer = PillowWriter(fps = 10, metadata = metadata)

    x_list = []
    y_list = []
    n = N
    with writer.saving(fig, 'Complex_zeros.gif', intervals): #with function makes a new file which we save our frames to.
        for b_val in np.linspace(b_min, b_max, intervals):
            #For every b value we plot the roots of the nth partition function.
            Z_zero = partition_zero(0, 0, 0, b_val)
            Z_1 = partition_zero(1, 0, 0, b_val)
            Z_2 = partition_zero(1, 1, 0, b_val)
            Z_3 = partition_zero(1, 1, 1, b_val)
            Partition_0 = np.array([Z_zero, Z_1, Z_2, Z_3])
            Partition_Function_N = Z_iterator(n, Partition_0, 1)
            text_box_content = f'n = {n} \nb = {b_val:.2f}' #Text box is hear as we want to b value to change with each frame
            plt.text(0.8, 1, text_box_content, bbox=dict(facecolor='white', alpha=0.5), ha = 'left')
            Roots, x_vals, y_vals = Zeros_finder(Partition_Function_N)
            
            list_plot.set_data(x_vals, y_vals) #This line refers back to the list_plot which was first empty and resets its
            #data values so that a new frame is created. Then we grab the frame which saves it to our file. Thus we get
            #a gif.
            writer.grab_frame()
            print(b_val)
    return None
animator(0.001, 0.8, 200, 5)
