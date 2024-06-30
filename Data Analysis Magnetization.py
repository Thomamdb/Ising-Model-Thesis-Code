
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import csv

def Figure_plotter_Magnetization(Temperatures, Data, Data_error, Label, Length):
    N = Length ** 2
    T_c = 2.26918531421
    T_c_num = 2.4
    max_y = max(Data)
    min_T = min(Temperatures)
    max_T = max(Temperatures)

    # Create figure and axis first
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # Create masks based on the critical temperature
    mask_below_or_equal = np.array(Temperatures) <= T_c_num
    mask_above = np.array(Temperatures) > T_c_num

    # Split the data based on the masks
    temperatures_below_or_equal = np.array(Temperatures)[mask_below_or_equal]
    data_below_or_equal = np.array(Data)[mask_below_or_equal]
    data_error_below_or_equal = np.array(Data_error)[mask_below_or_equal]

    temperatures_above = np.array(Temperatures)[mask_above]
    data_above = np.array(Data)[mask_above]
    data_error_above = np.array(Data_error)[mask_above]

    # Plot data with error bars
    ax.errorbar(temperatures_below_or_equal, data_below_or_equal, yerr=data_error_below_or_equal, fmt='bo', markersize=2)
    ax.errorbar(temperatures_above, data_above, yerr=data_error_above, fmt='o', markersize=2, color='#A9CCE3')

    # Set labels and title
    ax.set_xlabel('Temperature')
    ax.set_ylabel(Label)
    ax.set_title(Label + ' at different Temperatures')
    ax.axvline(x=T_c, color='red', linestyle='--')
    ax.text(T_c, -0.15, r'$T_c$', color='red', ha='center', va='bottom')

    # Set limits and customize plot
    ax.set_ylim(0, 1.1)
    ax.set_xlim(min([min_T - 0.2, 1]), max_T + 0.2)

    # Hide the top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.show()

Length = 32

temperatures =[]
magnetizations = []
errors = []
with open('Physics Coding/lis32and216sweeps.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    # Skip the header
    next(csv_reader)
    # Iterate through the rows and append to the respective lists
    for row in csv_reader:
        temperatures.append(float(row[0]))
        magnetizations.append(float(row[1]))
        errors.append(float(row[2]))
print(temperatures[1], magnetizations[1], errors[1])
Figure_plotter_Magnetization(np.array(temperatures), np.array(magnetizations), np.array(errors), r'$\langle s \rangle$', Length)

def magnetization_model(T, beta):
    return  (1 - (1 / (np.sinh(2 / T))** 2))**beta

# beta_list = []
# upper_bound_temps = np.linspace(1, 2.3, int((2.3 - 1)* 10))
# for upper_bound in upper_bound_temps:
#     y_fit_data = np.array(magnetizations)
#     t_fit_data = np.array(temperatures)
#     filtered_indices = t_fit_data <= upper_bound
#     y_fit_data = y_fit_data[filtered_indices]
#     t_fit_data = t_fit_data[filtered_indices]
#     # Create a mask for non-NaN values
#     mask = ~np.isnan(y_fit_data)
#     y_fit_data = y_fit_data[mask]
#     t_fit_data = t_fit_data[mask]
#     print(y_fit_data)
#     popt, pcov = curve_fit(magnetization_model, np.array(y_fit_data), np.array(t_fit_data))
#     beta = popt[0]

#     beta_estimated = popt[0]
#     beta_error = pcov[0]
#     beta_list.append(beta_estimated)
#     print(f'The value for beta is {beta} with error {beta_error}.')
#     # plt.plot(np.array(t_fit_data), magnetization_model(np.array(t_fit_data), beta_estimated), color='red', label='Fitted Curve')

#     # plt.xlabel('Temperature')
#     # plt.ylabel('Magnetization')
#     # plt.title('Curve Fitting Example')

#     # plt.legend()
#     # plt.show()
# beta_list = np.array(beta_list)
# print(beta_list)
# print('The upper bound is', upper_bound_temps)
# plt.plot(upper_bound, beta_list)
# plt.show()