
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import csv
import re

L = 32
Temperature_list = []
Correlations_list = []
Correlation_errors_list = []
Distances_list = []
csv_file_path = 'Physics Coding\Correlations15to34lis32.csv'



# Open and read the CSV file
float_re = re.compile(r'[-+]?\d*\.\d+e[-+]?\d+|\d+\.\d*|\.\d+|\d+')
def parse_list(s):
    return list(map(float, float_re.findall(s)))

# Open and read the CSV file
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    
    # Read each row and populate the lists
    for row in csvreader:
        Temperature_list.append(float(row['Temperature']))
        
        # Use parse_list to convert string representations back to lists of floats
        Correlations_list.append(np.array(parse_list(row['Correlations'])))
        Correlation_errors_list.append(np.array(parse_list(row['Correlation Errors'])))
        Distances_list.append(np.array(parse_list(row['Distances'])))
def correlation_model(x, c, a, b):
    return c + a * x + b * np.log(x)

def Correlation_at_temp_plotter(index, Correlation_list, Distances_list, error_list, Temperatures_list, Label):
    index = int(index)
    lCor_0 = np.log(Correlation_list[index])
    Dis = Distances_list[index]
    max_d = max(Dis)
    Cor_err = error_list[index]
    mask_dis = Dis <= L / 3
    mask_dis_n = Dis > L / 3
    lCor_n = lCor_0[mask_dis_n]
    Dis_n = Dis[mask_dis_n]
    lCor = lCor_0[mask_dis]
    Dis = Dis[mask_dis]

    # Cor_err = error_list[mask_dis]
    lCor_err = abs(Cor_err / Correlation_list[index])
    fig, ax = plt.subplots(figsize=(10, 10))
    mask_log = lCor >= - 6
    mask_log_d = lCor < -6
    lCor_d = lCor[mask_log_d]
    Dis_d = Dis[mask_log_d]
    lCor = lCor[mask_log]
    Dis = Dis[mask_log]
    ax.plot(Dis, lCor, 'bo', markersize = 5)
    ax.plot(Dis_n, lCor_n, 'o', markersize=5, color = '#A9CCE3')
    ax.plot(Dis_d, lCor_d, 'o', markersize=5, color = '#A9CCE3')
    ax.set_xlabel('Distance')
    ax.set_ylabel(Label)
    ax.set_title(Label + ' at different distances')

    # Set limits and customize plot
    # ax.set_ylim(0, )
    ax.set_xlim(0, max_d+1)
    print(Temperature_list[index])
    # Hide the top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()
Temp_length = len(Temperature_list)
print(Temp_length)
Correlation_at_temp_plotter(14, Correlations_list, Distances_list, Correlation_errors_list, Temperature_list, 'Correlations')

Correlation_lengths_list = []
Correlation_lengths_err_list = []
Correlation_temps_list = []
for index in range(len(Temperature_list)):
    T = Temperature_list[index]
    lCor_0 = np.log(Correlations_list[index])
    Dis = Distances_list[index]
    Cor_err = Correlation_errors_list[index]
    lCor_err = abs(Cor_err / Correlations_list[index])
    
    mask_dis = Dis <= L / 3
    lCor = lCor_0[mask_dis]
    Dis = Dis[mask_dis]
    lCor_err = lCor_err[mask_dis]
    
    mask_log = lCor >= -6
    lCor = lCor[mask_log]
    Dis = Dis[mask_log]
    lCor_err = lCor_err[mask_log]
    
    mask_zero = Dis > 0
    lCor = lCor[mask_zero]
    Dis = Dis[mask_zero]
    lCor_err = lCor_err[mask_zero]
    print('The length of lcor is ', len(lCor))
    if len(lCor) >= 5:
        print(lCor)
        print(Dis)
        popt, pcov = curve_fit(correlation_model, Dis, lCor, sigma = lCor_err)
        c_fit, a_fit, b_fit = popt
        c_err, a_err, b_err = np.sqrt(np.diag(pcov))
            
        Correlation_length = 1 / abs(a_fit)
        Correlation_length_error = a_err / (a_fit ** 2)
            
        print(f'The correlation length is equal to {Correlation_length} with error {Correlation_length_error}.')
            
        Correlation_temps_list.append(T)
        Correlation_lengths_list.append(Correlation_length)
        Correlation_lengths_err_list.append(Correlation_length_error)
Label = 'Correlation Length'
fig, ax = plt.subplots(figsize=(10, 10))
Correlation_lengths_list.pop(1)
Correlation_lengths_err_list.pop(1)
Correlation_temps_list.pop(1)
ax.errorbar(Correlation_temps_list, Correlation_lengths_list, yerr = Correlation_lengths_err_list, fmt = 'bo')
ax.set_xlabel('Temperature')
T_c = 2.26918531421
ax.set_ylabel(Label)
ax.set_title(Label + ' at different Temperatures')
ax.axvline(x=T_c, color='red', linestyle='--')
ax.text(T_c, -2, r'$T_c$', color='red', ha='center', va='bottom')
    # Set limits and customize plot
ax.set_ylim(0, max(Correlation_lengths_list) + max(Correlation_lengths_err_list))
# ax.set_xlim(1.2, 4)

    # Hide the top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()
plt.plot(Correlation_temps_list, Correlation_lengths_list, 'bo')
plt.show()





