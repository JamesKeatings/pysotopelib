import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

def info_133Ba():
    energy = np.array([80.998, 276.398, 302.853, 356.017, 383.851])
    error_energy = np.array([0.005, 0.001, 0.001, 0.002, 0.003])
    intensity = np.array ([34.11, 7.147, 18.30, 61.94, 8.905])
    error_intensity = np.array([0.28, 0.030, 0.06, 0.14, 0.029])

    return energy, error_energy, intensity, error_intensity

def info_152Eu():
    energy = np.array([121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903, 867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970, 1299.152, 1408.022])    
    error_energy = np.array([0.0004, 0.0010, 0.0019, 0.003, 0.004, 0.006, 0.006, 0.004, 0.004, 0.14, 0.006, 0.013, 0.009, 0.004])
    intensity = np.array ([28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 1.731, 13.54, 1.412, 1.626, 20.85])
    error_intensity = np.array([0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.009, 0.06, 0.008, 0.011, 0.09])

    return energy, error_energy, intensity, error_intensity

def info_60Co():
    energy = np.array([1173.238, 1332.502])    
    error_energy = np.array([0.004, 0.005])
    intensity = np.array ([99.857, 99.983])
    error_intensity = np.array([0.022, 0.006])

    return energy, error_energy, intensity, error_intensity


def readascii(filename):
    # Read the file and load the data into a NumPy array
    data = np.loadtxt(filename)

    # Check if the data has one or two columns
    if data.ndim == 1:  # If the data has only one column
        y_values = data
        x_values = np.arange(len(y_values))  # Generate x-values starting from 0
    elif data.shape[1] == 2:  # If the data has two columns
        x_values = data[:, 0]
        y_values = data[:, 1]
    else:
        raise ValueError("The file must contain either one or two columns")

    # Determine the range based on min and max of x-values
    x_min = x_values.min()
    x_max = x_values.max()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    #plt.hist(x_values, bins=len(x_values), weights=y_values, color='blue', alpha=0.7, linewidth=3)
    plt.step(x_values, y_values, where='mid', color='blue', alpha=0.7, linewidth=1)  
    
    # Set the x-axis range manually
    plt.xlim(x_min, x_max)
    
    plt.xlabel('Energy [keV]')
    plt.ylabel(f'Counts [{x_values[1]} keV/ch]')

    # Show the plot
    plt.show()

    return data

# Gaussian with linear background function
def gaussian_with_background(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + c

# Alternative function
def efficiency_RadWare(EG, A, B, C, D, E, Scale):
    E1 = 100.0
    E2 = 1000.0
    G = 15
    x = np.log(EG / E1)
    y = np.log(EG / E2)   
    exponent = Scale * ((A + B*x)**(-G)+(C + D*y + E*y**2)**(-G))**(-1/G)
    
    return np.exp(exponent)


# Alternative function
def efficiency_Log(EG, A, B, C, D, E, Scale):
    E0 = 325.0
    s = np.log(EG / E0)
    exponent = A * s + B * s**2 + C * s**3 + D * s**4 + E * s**5
    
    return Scale * np.exp(exponent)


# Calibration function Radware
def calibrate_efficiency_Radware(data, scale):
    # Define the energy array within the function
    #energy = np.array([80.998, 121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903, 867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970, 1299.152, 1408.022])
    # Define the intensity and error_intensity arrays
    #intensity = np.array([42.64, 28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 1.731, 13.54, 1.412, 1.626, 20.85])
    #error_intensity = np.array([0.28, 0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.009, 0.06, 0.008, 0.011, 0.09])

    energy_152Eu, error_energy_152Eu, intensity_152Eu, error_intensity_152Eu = info_152Eu()
    energy_133Ba, error_energy_133Ba, intensity_133Ba, error_intensity_133Ba = info_133Ba()
    
    #intensity_133Ba = intensity_133Ba/1.5

    
    unsorted_energy = np.concatenate((energy_152Eu, energy_133Ba))
    unsorted_error_energy = np.concatenate((error_energy_152Eu, error_energy_133Ba))
    unsorted_intensity = np.concatenate((intensity_152Eu, intensity_133Ba))
    unsorted_error_intensity = np.concatenate((error_intensity_152Eu, error_intensity_133Ba))

    sorted_indices = np.argsort(unsorted_energy)

    energy = unsorted_energy[sorted_indices]
    error_energy = unsorted_error_energy[sorted_indices]
    intensity = unsorted_intensity[sorted_indices]
    error_intensity = unsorted_error_intensity[sorted_indices]
    

    x_values = data[:, 0]
    y_values = data[:, 1]

    results = []
    num_plots = len(energy)  # Number of subplots needed
    ncols = 4  # Number of columns for subplots
    nrows = int(np.ceil(num_plots / ncols))  # Calculate number of rows

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Define the fitting region around each energy peak
    for i, e in enumerate(energy):
        # Define a region of interest (ROI) around the energy
        roi_width=7
        roi_mask = (x_values > e - roi_width) & (x_values < e + roi_width)
        x_roi = x_values[roi_mask]
        y_roi = y_values[roi_mask]

        # Initial guess for the parameters A (amplitude), mu (mean), sigma (std dev), m, c
        initial_guess = [np.max(y_roi), e, 1.0, 0, np.min(y_roi)]

        try:
            # Perform the curve fitting using curve_fit
            popt, pcov = curve_fit(gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            # Calculate the area under the Gaussian peak
            area = A * sigma * np.sqrt(2 * np.pi)

            # Store the results along with the corresponding energy and intensity
            # results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': error_intensity[i]})
            results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': np.sqrt(area)})

            # Plot the fit for visual inspection
            ax = axes[i]  # Select the current subplot
            ax.step(x_roi, y_roi, label='Data', where='post', color='blue', alpha=0.7, linewidth=1)
            ax.plot(x_roi, gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
            ax.axvline(mu, color='g', linestyle='--', label=f'Centroid: {mu:.2f}')
            ax.set_title(f'Fit around {e:.2f} keV')
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel(f'Counts [{x_values[1]} keV/ch]')
            ax.legend()

        except RuntimeError:
            print(f"Fit could not be performed for peak around {e:.2f} keV")

    # If no valid fits were found, return an empty result
    if not results:
        return {}

    # Extract areas and normalize with respect to the most intense peak
    areas = np.array([result['area'] for result in results])
    max_area = np.max(areas)
    normalized_areas = areas / max_area * 100  # Multiply by 100
    

    # Prepare the data for fitting the efficiency function
    energies = np.array([result['energy'] for result in results])
    intensities = np.array([result['intensity'] for result in results])
    areas = np.array([result['area'] for result in results])
    errors = np.array([result['error_intensity'] for result in results])

    # Normalize areas by intensities
    normalized_areas = normalized_areas / intensities
    # errors =  (errors / area) / intensities
    errors =  (errors / area) * 300

    
    activityscale = scale
    for energy_value in energy_133Ba:
        # Find the indices where energies match the current value from energy_133Ba
        matching_indices = np.where(energies == energy_value)[0]
        
        # Multiply the corresponding values in normalized_areas by the constant
        normalized_areas[matching_indices] *= activityscale
    

    # Initial guess for the parameters A, B, C, D, E
    initial_guess2 = [0.8851239, 1.92979759, 0.39965769, -0.62558024, -0.04598688, 0.5] # for Radware efficiency
    #initial_guess2 = [2.529, -0.5, -0.07, 0.07, 0.034] # for Log efficiency
        
    # Perform the curve fitting
    popt2, pcov2 = curve_fit(efficiency_RadWare, energies, normalized_areas, p0=initial_guess2, sigma=errors, absolute_sigma=True)
    A, B, C, D, E, Scale = popt2
    print("Fit parameters:", popt2)

    # Generate a range of energies for plotting
    plot_energies = np.arange(energies.min(), energies.max())

    #popt2 = [ 0.77726656,  2.47711261,  0.39965775, -0.62558048, -0.04598727 ]

    # Calculate the fitted values
    fit_values = efficiency_RadWare(plot_energies, *popt2)

    # Calculate the confidence intervals
    # Initialize an array to store the variances at each energy point
    fit_var = np.zeros(len(plot_energies))

    # Propagate the uncertainties using the covariance matrix
    for i in range(len(popt2)):
        for j in range(len(popt2)):
            # Compute the derivative of the fit function with respect to each parameter
            deriv_i = (efficiency_RadWare(plot_energies, *(popt2 + np.eye(len(popt2))[i] * np.sqrt(np.diag(pcov2))[i])) -
                       efficiency_RadWare(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[i]
            deriv_j = (efficiency_RadWare(plot_energies, *(popt2 + np.eye(len(popt2))[j] * np.sqrt(np.diag(pcov2))[j])) -
                       efficiency_RadWare(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[j]
            
            # Add to the variance
            fit_var += deriv_i * deriv_j * pcov2[i, j]

    # The standard deviation (1-sigma) of the fit is the square root of the variance
    fit_std = np.sqrt(fit_var)

    # Define the upper and lower bounds
    upper_bound = fit_values + fit_std
    lower_bound = fit_values - fit_std

    # Open a file for writing
    #file_path = "fit_values.txt"
    #with open(file_path, 'w') as file:
        #for energy, fit_val, fit_err in zip(plot_energies, fit_values, fit_std):
            #file.write(f"Energy: {energy:.0f} keV, Fit: {fit_val:.6f} +/- {fit_err:.6f}\n")

    x_152Eu = []
    points_152Eu = []
    errors_152Eu = []

    for energy_value in energy_152Eu:
        # Find the indices where energies match the current value from energy_133Ba
        matching_indices = np.where(energies == energy_value)[0]
    
        # Flatten or index to get scalar values
        if matching_indices.size > 0:  # Ensure matching indices are found
            x_152Eu.append(energies[matching_indices][0])
            points_152Eu.append(normalized_areas[matching_indices][0])
            errors_152Eu.append(errors[matching_indices][0])

    x_133Ba = []
    points_133Ba = []
    errors_133Ba = []

    for energy_value in energy_133Ba:
        # Find the indices where energies match the current value from energy_133Ba
        matching_indices = np.where(energies == energy_value)[0]
    
        # Flatten or index to get scalar values
        if matching_indices.size > 0:  # Ensure matching indices are found
            x_133Ba.append(energies[matching_indices][0])
            points_133Ba.append(normalized_areas[matching_indices][0])
            errors_133Ba.append(errors[matching_indices][0])
        

    #print("152Eu x:", x_152Eu)
    #print("152Eu y:", points_152Eu)
    #print("133Ba x:", x_133Ba)
    #print("133Ba y:", points_133Ba)    
    
    
    # Plotting the results
    plt.figure()
    plt.errorbar(x_152Eu, points_152Eu, yerr=errors_152Eu, fmt="x", label='152Eu Data')
    plt.errorbar(x_133Ba, points_133Ba, yerr=errors_133Ba, fmt="o", label='133Ba Data') 
    plt.plot(plot_energies, fit_values, 'r-', label='Fit')
    plt.fill_between(plot_energies, lower_bound, upper_bound, color='red', alpha=0.15, label='Confidence Interval (1σ)')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Efficiency [Rel.]')
    plt.legend()
    plt.show()
    
# Calibration function Log
def calibrate_efficiency_Log(data, scale):
    # Define the energy array within the function
    #energy = np.array([80.998, 121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903, 867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970, 1299.152, 1408.022])
    # Define the intensity and error_intensity arrays
    #intensity = np.array([34.11, 28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 1.731, 13.54, 1.412, 1.626, 20.85])
    #error_intensity = np.array([0.28, 0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.009, 0.06, 0.008, 0.011, 0.09])

    energy_152Eu, error_energy_152Eu, intensity_152Eu, error_intensity_152Eu = info_152Eu()
    energy_133Ba, error_energy_133Ba, intensity_133Ba, error_intensity_133Ba = info_133Ba()
    
    #intensity_133Ba = intensity_133Ba/1.5

    
    unsorted_energy = np.concatenate((energy_152Eu, energy_133Ba))
    unsorted_error_energy = np.concatenate((error_energy_152Eu, error_energy_133Ba))
    unsorted_intensity = np.concatenate((intensity_152Eu, intensity_133Ba))
    unsorted_error_intensity = np.concatenate((error_intensity_152Eu, error_intensity_133Ba))

    sorted_indices = np.argsort(unsorted_energy)

    energy = unsorted_energy[sorted_indices]
    error_energy = unsorted_error_energy[sorted_indices]
    intensity = unsorted_intensity[sorted_indices]
    error_intensity = unsorted_error_intensity[sorted_indices]
    

    x_values = data[:, 0]
    y_values = data[:, 1]

    results = []
    num_plots = len(energy)  # Number of subplots needed
    ncols = 4  # Number of columns for subplots
    nrows = int(np.ceil(num_plots / ncols))  # Calculate number of rows

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Define the fitting region around each energy peak
    for i, e in enumerate(energy):
        # Define a region of interest (ROI) around the energy
        roi_width=7
        roi_mask = (x_values > e - roi_width) & (x_values < e + roi_width)
        x_roi = x_values[roi_mask]
        y_roi = y_values[roi_mask]

        # Initial guess for the parameters A (amplitude), mu (mean), sigma (std dev), m, c
        initial_guess = [np.max(y_roi), e, 1.0, 0, np.min(y_roi)]

        try:
            # Perform the curve fitting using curve_fit
            popt, pcov = curve_fit(gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            # Calculate the area under the Gaussian peak
            area = A * sigma * np.sqrt(2 * np.pi)

            # Store the results along with the corresponding energy and intensity
            # results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': error_intensity[i]})
            results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': np.sqrt(area)})

            # Plot the fit for visual inspection
            ax = axes[i]  # Select the current subplot
            ax.step(x_roi, y_roi, label='Data', where='post', color='blue', alpha=0.7, linewidth=1)
            ax.plot(x_roi, gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
            ax.axvline(mu, color='g', linestyle='--', label=f'Centroid: {mu:.2f}')
            ax.set_title(f'Fit around {e:.2f} keV')
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel(f'Counts [{x_values[1]} keV/ch]')
            ax.legend()

        except RuntimeError:
            print(f"Fit could not be performed for peak around {e:.2f} keV")

    # If no valid fits were found, return an empty result
    if not results:
        return {}

    # Extract areas and normalize with respect to the most intense peak
    areas = np.array([result['area'] for result in results])
    max_area = np.max(areas)
    normalized_areas = areas / max_area * 100  # Multiply by 100
    

    # Prepare the data for fitting the efficiency function
    energies = np.array([result['energy'] for result in results])
    intensities = np.array([result['intensity'] for result in results])
    areas = np.array([result['area'] for result in results])
    errors = np.array([result['error_intensity'] for result in results])

    # Normalize areas by intensities
    normalized_areas = normalized_areas / intensities
    # errors =  (errors / area) / intensities
    errors =  (errors / area) * 300

    
    activityscale = scale
    for energy_value in energy_133Ba:
        # Find the indices where energies match the current value from energy_133Ba
        matching_indices = np.where(energies == energy_value)[0]
        
        # Multiply the corresponding values in normalized_areas by the constant
        normalized_areas[matching_indices] *= activityscale
    

    # Initial guess for the parameters A, B, C, D, E
    #initial_guess2 = [2.66, 1.9, 0.39965769, -0.62558024, -0.04598688] # for Radware efficiency
    initial_guess2 = [2.529, -0.5, -0.07, 0.07, 0.034, 1] # for Log efficiency
        
    # Perform the curve fitting
    popt2, pcov2 = curve_fit(efficiency_Log, energies, normalized_areas, p0=initial_guess2, sigma=errors, absolute_sigma=True)
    A, B, C, D, E, Scale = popt2
    print("Fit parameters:", popt2)

    # Generate a range of energies for plotting
    plot_energies = np.arange(energies.min(), energies.max())

    # Calculate the fitted values
    fit_values = efficiency_Log(plot_energies, *popt2)

    # Calculate the confidence intervals
    # Initialize an array to store the variances at each energy point
    fit_var = np.zeros(len(plot_energies))

    # Propagate the uncertainties using the covariance matrix
    for i in range(len(popt2)):
        for j in range(len(popt2)):
            # Compute the derivative of the fit function with respect to each parameter
            deriv_i = (efficiency_Log(plot_energies, *(popt2 + np.eye(len(popt2))[i] * np.sqrt(np.diag(pcov2))[i])) -
                       efficiency_Log(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[i]
            deriv_j = (efficiency_Log(plot_energies, *(popt2 + np.eye(len(popt2))[j] * np.sqrt(np.diag(pcov2))[j])) -
                       efficiency_Log(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[j]
            
            # Add to the variance
            fit_var += deriv_i * deriv_j * pcov2[i, j]

    # The standard deviation (1-sigma) of the fit is the square root of the variance
    fit_std = np.sqrt(fit_var)

    # Define the upper and lower bounds
    upper_bound = fit_values + fit_std
    lower_bound = fit_values - fit_std

    # Open a file for writing
    #file_path = "fit_values.txt"
    #with open(file_path, 'w') as file:
        #for energy, fit_val, fit_err in zip(plot_energies, fit_values, fit_std):
            #file.write(f"Energy: {energy:.0f} keV, Fit: {fit_val:.6f} +/- {fit_err:.6f}\n")


    x_152Eu = []
    points_152Eu = []
    errors_152Eu = []

    for energy_value in energy_152Eu:
        # Find the indices where energies match the current value from energy_133Ba
        matching_indices = np.where(energies == energy_value)[0]
    
        # Flatten or index to get scalar values
        if matching_indices.size > 0:  # Ensure matching indices are found
            x_152Eu.append(energies[matching_indices][0])
            points_152Eu.append(normalized_areas[matching_indices][0])
            errors_152Eu.append(errors[matching_indices][0])

    x_133Ba = []
    points_133Ba = []
    errors_133Ba = []

    for energy_value in energy_133Ba:
        # Find the indices where energies match the current value from energy_133Ba
        matching_indices = np.where(energies == energy_value)[0]
    
        # Flatten or index to get scalar values
        if matching_indices.size > 0:  # Ensure matching indices are found
            x_133Ba.append(energies[matching_indices][0])
            points_133Ba.append(normalized_areas[matching_indices][0])
            errors_133Ba.append(errors[matching_indices][0])
        

    #print("152Eu x:", x_152Eu)
    #print("152Eu y:", points_152Eu)
    #print("133Ba x:", x_133Ba)
    #print("133Ba y:", points_133Ba)    
    
    
    # Plotting the results
    plt.figure()
    plt.errorbar(x_152Eu, points_152Eu, yerr=errors_152Eu, fmt="x", label='152Eu Data')
    plt.errorbar(x_133Ba, points_133Ba, yerr=errors_133Ba, fmt="o", label='133Ba Data') 
    plt.plot(plot_energies, fit_values, 'r-', label='Fit')
    plt.fill_between(plot_energies, lower_bound, upper_bound, color='red', alpha=0.15, label='Confidence Interval (1σ)')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Efficiency [Rel.]')
    plt.legend()
    plt.show()
