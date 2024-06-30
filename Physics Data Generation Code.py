"""
Multiple Temperatures Monte Carlo Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import csv

def binning_analysis(samples):
    """Perform a binning analysis over samples and return 
    errors: an array of the error estimate at each binning level, 
    tau: the estimated integrated autocorrelation time, 
    converged: a flag indicating if the binning has converged, and 
    bins: the last bin values"""
    minbins = 2**6 # minimum number of bins     
    maxlevel = int(np.log2(len(samples)/minbins)) # number of binning steps
    maxsamples = minbins * 2**(maxlevel)   # the maximal number of samples considered 
    bins = np.array(samples[-maxsamples:]) 
    errors = np.zeros(maxlevel+1)
    for k in range(maxlevel):
        errors[k] = np.std(bins)/np.sqrt(len(bins)-1.)
        bins = np.array((bins[::2] + bins[1::2])/2.)
        
    errors[maxlevel] = np.std(bins)/np.sqrt(len(bins)-1.)    
    tau = 0.5*((errors[-1]/errors[0])**2 - 1.)
    relchange = (errors[1:] - errors[:-1]) / errors[1:]
    meanlastchanges = np.mean(relchange[-3:])    # get the average over last changes
    converged = 1
    if meanlastchanges > 0.05:
        print("warning: binning maybe not converged, meanlastchanges:", meanlastchanges)
        converged = 0
    return (errors, tau, converged, bins)


def plotlatt(latt):
    """ Plot the lattice """
    plt.matshow(1-latt.transpose(),cmap=plt.get_cmap('gray'))
    plt.show()


def getenergy_forloop(latt):
    """ Compute the energy of a configuration using for-loops"""      
    # Less efficient code based on for-loop
    L = latt.shape[0]
    energy = 0 
    for i in range(L):
       for j in range(L):
            sij = latt[i,j]
            val = latt[(i+1)%L, j] + latt[i,(j+1)%L]             
            energy += -val*sij
    return energy

def getcorrelations(latt):
    L = latt.shape[0]
    Correlations = []
    Spin_at_ij = []
    for r_i in range(int(L/2 + 1)):
        for r_j in range(r_i + 1):
            Correlations.append(latt[0, 0] * latt[r_i, r_j])
            Spin_at_ij.append(latt[r_i, r_j])
    # plotlatt(latt)
    return Correlations, Spin_at_ij
        
    
def getenergy(latt):
    """ Compute the energy of a configuration """
    # efficient code based on numpy-array operations
    energy = - np.sum(np.sum(latt * (np.roll(latt, 1, axis=0) + np.roll(latt, 1, axis=1))))      
    return energy
  
@jit()
def doMCsweep(latt, energy, mag, beta, correlations, spinatij):
    """ Do a MC sweep and return current energy and magnetization"""
    L = latt.shape[0]
    N = L*L

    # do one sweep (N MC steps)
    for k in range(N):
        i = np.random.randint(L)
        j = np.random.randint(L)
        sij = latt[i,j]
        val = latt[(i+1)%L, j] + latt[i,(j+1)%L] + latt[(i-1)%L,j] + latt[i,(j-1)%L]
        ediff = 2*val*sij
        random_number = np.random.random()
        prob = math.exp(-beta*ediff)
        for r_i in range(int(L/2 + 1)):
                for r_j in range(r_i + 1):
                    Correlation_index = sum(range(r_i + 1)) + r_j
                    correlations[Correlation_index] = latt[0, 0] * latt[r_i % L, r_j % L]
                    spinatij[Correlation_index] = latt[r_i % L, r_j % L]
        if (random_number < prob):
            latt[i,j] = - latt[i,j]
            energy += ediff
            mag -= 2*sij
    return (energy, mag, correlations, spinatij)

class Obsresult:
    """Class to evaluate the measurements"""
    def __init__(self, vals, name="obs"):
        self.mean = np.mean(vals)
        self.name = name
        # print(vals)
        (self.binerrors, self.tau, self.converged, self.binvals) = binning_analysis(vals)
        self.err = self.binerrors[-1]
        # print(name, ":", self.mean, " +/- ", self.err)
    def plotbinning(self):
        plt.figure()
        plt.plot(self.binerrors,'ko')
        plt.xlabel('binning step')
        plt.ylabel('estimated error')
    def relerrors(self):
        return (self.binerrors[1:] - self.binerrors[:-1]) / self.binerrors[-1]

@jit
def init_RNG(seed):
    """initialize random number generator with seed"""
    np.random.seed(seed)

def correlation_model(x, c, a, b):
    return c + a * x + b * np.log(x)

def runMCsim(L, T, Nsweeps, Ntherm, seed):    
    """run a Monte Carlo simulation"""
    N = L*L    
    beta = 1./T
    init_RNG(seed)
    
    # create lattice
    latt = np.random.rand(L,L)
    latt = np.sign(latt-0.5)
    
    # get initial values
    energy = getenergy(latt)
    mag = np.sum(latt)
    correlations, spinatij = getcorrelations(latt)
    # store measurements in:
    
    venergy = np.zeros((Nsweeps, 1))
    vmag = np.zeros((Nsweeps, 1))
    vcorrelations = np.zeros((Nsweeps, len(correlations)))
    vspinatij = np.zeros((Nsweeps, len(spinatij)))

    print(('Do 2D Ising simulation with L=%i, T=%f, Ntherm=%i, Nsweeps=%i, seed=%i' \
    % (L, T, Ntherm, Nsweeps, seed)))
    

    print(('do %i thermalization sweeps...' % Ntherm))
    for k in range(Ntherm):
        (energy, mag, correlations, spinatij) = doMCsweep(latt, energy, mag, beta, correlations, spinatij)
    # plotlatt(latt)
    print('done!')


    print(('perform %i MC sweeps and do measurements...' % Nsweeps))
    for k in range(Nsweeps):
        (energy, mag, correlations, spinatij) = doMCsweep(latt, energy, mag, beta, correlations, spinatij)
        # now store values
        venergy[k] = energy
        vmag[k] = mag
        vcorrelations[k] = correlations
        vspinatij[k] = spinatij
    print('done!')
    #plotlatt(latt)           # show lattice
    
    # create results
    print("--------------------")
    print("Evaluate observables")
    rese = Obsresult(venergy/N, "Energy per site")
    resmag = Obsresult(vmag/N, "Magnetization per site")
    resm = Obsresult(np.abs(vmag)/N, "m")
    resE2 = Obsresult(venergy*venergy, "Energy squared")
    vM2 = vmag*vmag
    resM2 = Obsresult(vM2, "M^2")
    resM4 = Obsresult(vM2*vM2, "M^4")
    Correlations_mean_list = []
    Spinzero = Obsresult(vspinatij[:, 0], "s at r_0")
    Spinat_largest_distance = Obsresult(vcorrelations[:, sum(range(int(L / 2) + 1))], f"Large Distance Correlation")
    for r_i in range(0, int(L /2) + 1):
        for r_j in range(r_i + 1):
            Correlation_index = sum(range(r_i + 1)) + r_j
            Spin_term = Obsresult(vspinatij[:, Correlation_index], f"Spin at ij ({r_i, r_j})")
            Term = Obsresult(vcorrelations[:, Correlation_index] - ((Spinzero.mean) * (Spin_term.mean)), f"Correlation ({r_i, r_j})")
            # Term = Obsresult(vcorrelations[:, Correlation_index] - (Spinzero.mean ** 2), f"Correlation ({r_i, r_j})")
            Correlations_mean_list.append([Term.mean, (Term.err + (abs(Spinzero.mean) * Spin_term.err + abs(Spin_term.mean) * Spinzero.err)), Term.name, np.sqrt((r_i ** 2 + r_j**2))])
    resCor = Obsresult(vcorrelations[:, 1], "NN Correlation per Site")
    # put all the results in a dictionary and return all results
    res = {"Es" : rese, "Ms" : resmag, "m" : resm, \
    "E2": resE2, "M2": resM2, "M4": resM4, "Cor": resCor, \
    "Coratij": Correlations_mean_list, "Large d Cor": Spinat_largest_distance}
    print("--------------------\n")
    first_values = [(sublist[0], sublist[-1], sublist[1]) for sublist in Correlations_mean_list]
    sorted_data = np.array(sorted(first_values, key=lambda x: float(x[1])))
    distances = sorted_data[:, 1]
    log_correlations = np.log(abs(sorted_data[:, 0])) #Please note here that I put the abs function.
    Lower_bound = 1
    Upper_bound_input = L
    # Upper_bound_input = float(input("Please enter an upper bound for the best fit: \n"))
    Upper_bound = min([Upper_bound_input, L/2 - 1])
    filtered_data = np.array(sorted_data[(sorted_data[:, 1] >= Lower_bound) & (sorted_data[:, 1] <= Upper_bound)])
    distances_filtered = filtered_data[:, 1].reshape(-1, 1)

    log_correlations_filtered = np.log(abs(filtered_data[:, 0]))
    log_correlations_errors_filtered = np.array(filtered_data[:, 2]) / np.abs(np.array(filtered_data[:, 0]))
    y_data = np.array(log_correlations_filtered).flatten()
    x_data = np.array(distances_filtered).flatten()
    y_err = np.array(log_correlations_errors_filtered).flatten()
    popt, pcov = curve_fit(correlation_model, x_data, y_data) #sigma = y_err used in fit
    c_fit, a_fit, b_fit = popt # y = c + ax + blog(x) is the model
    c_err, a_err, b_err = np.sqrt(np.diag(pcov))

    Correlation_length = 1 / abs(a_fit)
    Correlation_length_error = a_err / (a_fit ** 2)
    print(a_err)

    print(f'The correlation length is equal to {Correlation_length} with error {Correlation_length_error}.')
    plt.show()
    return res, Correlation_length, Correlation_length_error


if __name__ == "__main__":
    # This is a test run
    L = 32            # linear dimension of the lattice, lattice-size= N x N
    T = 2.2691           # temperature 2.26918531421
    Nsweeps = 2**17     # total number of Monte Carlo steps
    Ntherm = 2000      # Number of thermalization sweeps
    seed = 1
    
    res = runMCsim(L, T, Nsweeps, Ntherm, seed)
    Correlation_length_list = []
    Correlation_length_error_list = []
    Zero_field_magnetization = []
    Zero_field_err = []
    T_zero_field_list = []
    Temperature_list = []

    Correlations_list = []
    Correlation_errors_list = []
    Distances_list = []

    T_c = 2.26918531421
    Temperatures = np.linspace(1.5, 3.2, 35)
    for T in Temperatures:
        res, Correlation_length, Correlation_length_error = runMCsim(L, T, Nsweeps, Ntherm, seed)
        Correlation_length_list.append(Correlation_length)
        Correlation_length_error_list.append(Correlation_length_error)
        Temperature_list.append(T)
        Last_correlation = res["Large d Cor"] # Linear distance
        Correlations_mean_list = res["Coratij"]
        first_values = [(sublist[0], sublist[-1], sublist[1]) for sublist in Correlations_mean_list] # 0 is the mean. last is the distance. 1 is the error
        sorted_data = np.array(sorted(first_values, key=lambda x: float(x[1])))
        log_correlations = np.log(abs(sorted_data[:, 0]))
        Correlations_list.append(sorted_data[:, 0])
        Correlation_errors_list.append(sorted_data[:, 2])
        Distances_list.append(sorted_data[:, 1])
        Last_correlation_mean = Last_correlation.mean
        Upper_bound_fit = 0.5 * (L **2) * (1 / (L** 2))
        if np.where(Temperatures == T)[0] == 2:
            first_values = [(sublist[0], sublist[-1], sublist[1]) for sublist in Correlations_mean_list]
            sorted_data = np.array(sorted(first_values, key=lambda x: float(x[1])))
            distances = sorted_data[:, 1]
            log_correlations = np.log(abs(sorted_data[:, 0])) #Please note here that I put the abs function.
            plt.plot(distances, log_correlations, 'bo')
            plt.xlabel('Distance')
            plt.ylabel('Log of Correlation')
            plt.title('Correlation at different distances')
            plt.show()
        if Last_correlation_mean >= 0:
            Zero_field_magnetization.append(np.sqrt(Last_correlation_mean))
            T_zero_field_list.append(T)
            Zero_field_err.append(Last_correlation.err / (2 * np.sqrt(Last_correlation_mean)))
    with open('Correlationsforlis32.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
    
    # Write the header row
        header = ['Temperature', 'Correlations', 'Correlation Errors', 'Distances']
        csvwriter.writerow(header)
    T_zero_field_list = []
    Temperature_list = []
    T_c = 2.26918531421
    Temperatures = np.linspace(0.2, 3.4, 33)
    t_fit_data = []
    y_fit_data = []
    y_fit_errs = []
    for T in Temperatures:
        res = runMCsim(L, T, Nsweeps, Ntherm, seed)
        # Correlation_length_list.append(Correlation_length)
        # Correlation_length_error_list.append(Correlation_length_error)
        # Temperature_list.append(T)
        Last_correlation = res["Large d Cor"] # Linear distance
        Last_correlation_mean = Last_correlation.mean
        if Last_correlation_mean >= 0:
            Zero_field_magnetization.append(np.sqrt(Last_correlation_mean))
            T_zero_field_list.append(T)
            Zero_field_err.append(Last_correlation.err / (2 * np.sqrt(Last_correlation_mean)))
        if T <= T_c - Upper_bound_fit:
            t_fit_data.append(T)
            y_fit_data.append(np.sqrt(Last_correlation_mean))
            y_fit_errs.append(Last_correlation.err / (2 * np.sqrt(Last_correlation_mean)))
    with open('Magnetisationforlis32.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['List1', 'List2', 'List3'])
        for item1, item2, item3 in zip(T_zero_field_list, Zero_field_magnetization, Zero_field_err):
            writer.writerow([item1, item2, item3])
    # Write the data rows
        for i in range(len(Temperature_list)):
            row = [
                Temperature_list[i],
                Correlations_list[i],
                Correlation_errors_list[i],
                Distances_list[i]
            ]
            csvwriter.writerow(row)
