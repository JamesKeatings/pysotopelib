
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

def _info_133Ba():
    energy = np.array([80.998, 276.398, 302.853, 356.017, 383.851])
    error_energy = np.array([0.005, 0.001, 0.001, 0.002, 0.003])
    intensity = np.array ([34.11, 7.147, 18.30, 61.94, 8.905])
    error_intensity = np.array([0.28, 0.030, 0.06, 0.14, 0.029])

    return energy, error_energy, intensity, error_intensity


def _info_152Eu():
    #energy = np.array([121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903, 867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970, 1299.152, 1408.022])    
    #error_energy = np.array([0.0004, 0.0010, 0.0019, 0.003, 0.004, 0.006, 0.006, 0.004, 0.004, 0.14, 0.006, 0.013, 0.009, 0.004])
    #intensity = np.array ([28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 1.731, 13.54, 1.412, 1.626, 20.85])
    #error_intensity = np.array([0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.009, 0.06, 0.008, 0.011, 0.09])

    energy = np.array([121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903, 867.390, 964.055, 1085.842, 1112.087, 1212.970, 1299.152, 1408.022])    
    error_energy = np.array([0.0004, 0.0010, 0.0019, 0.003, 0.004, 0.006, 0.006, 0.004, 0.004, 0.006, 0.013, 0.009, 0.004])
    intensity = np.array ([28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 13.54, 1.412, 1.626, 20.85])
    error_intensity = np.array([0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.06, 0.008, 0.011, 0.09])

    return energy, error_energy, intensity, error_intensity


def _info_60Co():
    energy = np.array([1173.238, 1332.502])    
    error_energy = np.array([0.004, 0.005])
    intensity = np.array ([99.857, 99.983])
    error_intensity = np.array([0.022, 0.006])

    return energy, error_energy, intensity, error_intensity


def readascii(filename, showplot=False):
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

    if showplot:
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
def _gaussian_with_background(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + c
 
# Quadratic function  
def quadratic(x, a, b, c):
    """Quadratic function for curve fitting."""
    return a * x**2 + b * x + c    


# Alternative function
def _efficiency_Radware(EG, A, B, C, D, E, Scale):
    E1 = 100.0
    E2 = 1000.0
    G = 15
    x = np.log(EG / E1)
    y = np.log(EG / E2)   
    exponent = Scale * ((A + B*x)**(-G)+(C + D*y + E*y**2)**(-G))**(-1/G)
    
    return np.exp(exponent)


# Alternative function
def _efficiency_Log(EG, A, B, C, D, E, Scale):
    E0 = 325.0
    s = np.log(EG / E0)
    exponent = A * s + B * s**2 + C * s**3 + D * s**4 + E * s**5
    
    return Scale * np.exp(exponent)


# Calibration function Radware
def _calibrate_efficiency_Radware(data, scale=1):
    # Define the energy array within the function
    #energy = np.array([80.998, 121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903, 867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970, 1299.152, 1408.022])
    # Define the intensity and error_intensity arrays
    #intensity = np.array([34.11, 28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 1.731, 13.54, 1.412, 1.626, 20.85])
    #error_intensity = np.array([0.28, 0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.009, 0.06, 0.008, 0.011, 0.09])

    energy_152Eu, error_energy_152Eu, intensity_152Eu, error_intensity_152Eu = _info_152Eu()
    energy_133Ba, error_energy_133Ba, intensity_133Ba, error_intensity_133Ba = _info_133Ba()
    
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
        #roi_width=7
        roi_width=y_values.max()*0.005
        roi_mask = (x_values > e - roi_width) & (x_values < e + roi_width)
        x_roi = x_values[roi_mask]
        y_roi = y_values[roi_mask]

        # Initial guess for the parameters A (amplitude), mu (mean), sigma (std dev), m, c
        initial_guess = [np.max(y_roi), e, 1.0, 0, np.min(y_roi)]

        try:
            # Perform the curve fitting using curve_fit
            popt, pcov = curve_fit(_gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            # Calculate the area under the Gaussian peak
            area = A * sigma * np.sqrt(2 * np.pi)

            # Store the results along with the corresponding energy and intensity
            # results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': error_intensity[i]})
            results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': np.sqrt(area)})

            # Plot the fit for visual inspection
            ax = axes[i]  # Select the current subplot
            ax.step(x_roi, y_roi, label='Data', where='post', color='blue', alpha=0.7, linewidth=1)
            ax.plot(x_roi, _gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
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
        
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.7)

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
    errors =  (errors / area) * 100

    
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
    popt2, pcov2 = curve_fit(_efficiency_Radware, energies, normalized_areas, p0=initial_guess2, sigma=errors, absolute_sigma=True)
    A, B, C, D, E, Scale = popt2
    print("Fit parameters:", popt2)

    # Generate a range of energies for plotting
    plot_energies = np.arange(energies.min(), energies.max())

    # Calculate the fitted values
    fit_values = _efficiency_Radware(plot_energies, *popt2)

    # Calculate the confidence intervals
    # Initialize an array to store the variances at each energy point
    fit_var = np.zeros(len(plot_energies))

    # Propagate the uncertainties using the covariance matrix
    for i in range(len(popt2)):
        for j in range(len(popt2)):
            # Compute the derivative of the fit function with respect to each parameter
            deriv_i = (_efficiency_Radware(plot_energies, *(popt2 + np.eye(len(popt2))[i] * np.sqrt(np.diag(pcov2))[i])) -
                       _efficiency_Radware(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[i]
            deriv_j = (_efficiency_Radware(plot_energies, *(popt2 + np.eye(len(popt2))[j] * np.sqrt(np.diag(pcov2))[j])) -
                       _efficiency_Radware(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[j]
            
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
    
    """
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
    """
    # Create a new figure for the results and differences
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plotting the results
    plt.subplot(2, 1, 1)  # Top subplot
    plt.errorbar(x_152Eu, points_152Eu, yerr=errors_152Eu, fmt="x", label='152Eu Data')
    plt.errorbar(x_133Ba, points_133Ba, yerr=errors_133Ba, fmt="o", label='133Ba Data') 
    plt.plot(plot_energies, fit_values, 'r-', label='Fit')
    plt.fill_between(plot_energies, lower_bound, upper_bound, color='red', alpha=0.15, label='Confidence Interval (1σ)')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Efficiency [Rel.]')
    plt.legend()
   

    # Calculate the differences between calibrated points and reference points
    points_152Eu = np.array(points_152Eu)
    x_152Eu = np.array(x_152Eu)
    points_133Ba = np.array(points_133Ba)
    x_133Ba = np.array(x_133Ba)
    
    # Calculate the fitted values
    fit_values_152Eu = _efficiency_Log(x_152Eu, *popt2)
    fit_values_133Ba = _efficiency_Log(x_133Ba, *popt2)
    
    # Calculate the differences between calibrated points and reference points
    differences_152Eu = points_152Eu - fit_values_152Eu
    differences_133Ba = points_133Ba - fit_values_133Ba
    
    # Plot differences as a function of energy
    plt.subplot(2, 1, 2)  # Bottom subplot
    plt.errorbar(x_152Eu, differences_152Eu, yerr=errors_152Eu, fmt="x", label='152Eu Data')
    plt.errorbar(x_133Ba, differences_133Ba, yerr=errors_133Ba, fmt="o", label='133Ba Data') 
    plt.axhline(0, color='black', linestyle='--', label='Zero Difference')
    plt.fill_between(plot_energies, -fit_std, fit_std, color='red', alpha=0.15)
    #plt.plot(plot_energies, fit_std, 'r-')
    #plt.plot(plot_energies, -fit_std, 'r-')
    
    plt.xlabel('Energy [keV]')
    plt.ylabel('Difference [Rel.]')
    plt.xlim(0, x_152Eu[-1]+100)
    plt.show()
    
    
# Calibration function Log
def _calibrate_efficiency_Log(data, scale=1):
    # Define the energy array within the function
    #energy = np.array([80.998, 121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903, 867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970, 1299.152, 1408.022])
    # Define the intensity and error_intensity arrays
    #intensity = np.array([34.11, 28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 1.731, 13.54, 1.412, 1.626, 20.85])
    #error_intensity = np.array([0.28, 0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.009, 0.06, 0.008, 0.011, 0.09])

    energy_152Eu, error_energy_152Eu, intensity_152Eu, error_intensity_152Eu = _info_152Eu()
    energy_133Ba, error_energy_133Ba, intensity_133Ba, error_intensity_133Ba = _info_133Ba()
    
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
            popt, pcov = curve_fit(_gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            # Calculate the area under the Gaussian peak
            area = A * sigma * np.sqrt(2 * np.pi)

            # Store the results along with the corresponding energy and intensity
            # results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': error_intensity[i]})
            results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': np.sqrt(area)})

            # Plot the fit for visual inspection
            ax = axes[i]  # Select the current subplot
            ax.step(x_roi, y_roi, label='Data', where='post', color='blue', alpha=0.7, linewidth=1)
            ax.plot(x_roi, _gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
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

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.7)

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
    errors =  (errors / area) * 100

    
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
    popt2, pcov2 = curve_fit(_efficiency_Log, energies, normalized_areas, p0=initial_guess2, sigma=errors, absolute_sigma=True)
    A, B, C, D, E, Scale = popt2
    print("Fit parameters:", popt2)

    # Generate a range of energies for plotting
    plot_energies = np.arange(energies.min(), energies.max())

    # Calculate the fitted values
    fit_values = _efficiency_Log(plot_energies, *popt2)

    # Calculate the confidence intervals
    # Initialize an array to store the variances at each energy point
    fit_var = np.zeros(len(plot_energies))

    # Propagate the uncertainties using the covariance matrix
    for i in range(len(popt2)):
        for j in range(len(popt2)):
            # Compute the derivative of the fit function with respect to each parameter
            deriv_i = (_efficiency_Log(plot_energies, *(popt2 + np.eye(len(popt2))[i] * np.sqrt(np.diag(pcov2))[i])) -
                       _efficiency_Log(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[i]
            deriv_j = (_efficiency_Log(plot_energies, *(popt2 + np.eye(len(popt2))[j] * np.sqrt(np.diag(pcov2))[j])) -
                       _efficiency_Log(plot_energies, *popt2)) / np.sqrt(np.diag(pcov2))[j]
            
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
    
    # Create a new figure for the results and differences
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plotting the results
    plt.subplot(2, 1, 1)  # Top subplot
    plt.errorbar(x_152Eu, points_152Eu, yerr=errors_152Eu, fmt="x", label='152Eu Data')
    plt.errorbar(x_133Ba, points_133Ba, yerr=errors_133Ba, fmt="o", label='133Ba Data') 
    plt.plot(plot_energies, fit_values, 'r-', label='Fit')
    plt.fill_between(plot_energies, lower_bound, upper_bound, color='red', alpha=0.15, label='Confidence Interval (1σ)')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Efficiency [Rel.]')
    plt.legend()
   

    # Calculate the differences between calibrated points and reference points
    points_152Eu = np.array(points_152Eu)
    x_152Eu = np.array(x_152Eu)
    points_133Ba = np.array(points_133Ba)
    x_133Ba = np.array(x_133Ba)
    
    # Calculate the fitted values
    fit_values_152Eu = _efficiency_Log(x_152Eu, *popt2)
    fit_values_133Ba = _efficiency_Log(x_133Ba, *popt2)
    
    # Calculate the differences between calibrated points and reference points
    differences_152Eu = points_152Eu - fit_values_152Eu
    differences_133Ba = points_133Ba - fit_values_133Ba
    
    # Plot differences as a function of energy
    plt.subplot(2, 1, 2)  # Bottom subplot
    plt.errorbar(x_152Eu, differences_152Eu, yerr=errors_152Eu, fmt="x", label='152Eu Data')
    plt.errorbar(x_133Ba, differences_133Ba, yerr=errors_133Ba, fmt="o", label='133Ba Data') 
    plt.axhline(0, color='black', linestyle='--', label='Zero Difference')
    plt.fill_between(plot_energies, -fit_std, fit_std, color='red', alpha=0.15)
    #plt.plot(plot_energies, fit_std, 'r-')
    #plt.plot(plot_energies, -fit_std, 'r-')
    
    plt.xlabel('Energy [keV]')
    plt.ylabel('Difference [Rel.]')
    plt.xlim(0, x_152Eu[-1]+100)
    plt.show()
    
    
def calibrate_efficiency(data, scale=1):
    _calibrate_efficiency_Log(data, scale)

    
def calibrate_energy(data, height=None, prominence=5000, distance=10, tolerance=30, drawhists=True):
    # Get energy information from isotopes
    energy_152Eu, error_energy_152Eu, intensity_152Eu, error_intensity_152Eu = _info_152Eu()
    energy_133Ba, error_energy_133Ba, intensity_133Ba, error_intensity_133Ba = _info_133Ba()

    # Combine and sort the energy data
    unsorted_energy = np.concatenate((energy_152Eu, energy_133Ba))
    unsorted_error_energy = np.concatenate((error_energy_152Eu, error_energy_133Ba))
    unsorted_intensity = np.concatenate((intensity_152Eu, intensity_133Ba))
    unsorted_error_intensity = np.concatenate((error_intensity_152Eu, error_intensity_133Ba))

    sorted_indices = np.argsort(unsorted_energy)
    energy = unsorted_energy[sorted_indices]
    error_energy = unsorted_error_energy[sorted_indices]
    intensity = unsorted_intensity[sorted_indices]
    error_intensity = unsorted_error_intensity[sorted_indices]

    if data.ndim == 1:
        x_values = np.arange(len(data))  # Generate x-values starting from 0
        y_values = data
    elif data.shape[1] == 2:
        x_values = data[:, 0]
        y_values = data[:, 1]
    else:
        raise ValueError("Data must contain either one or two columns")

    # Find peaks in the data
    smoothed_y = savgol_filter(y_values, window_length=5, polyorder=2)
    peaks, _ = find_peaks(smoothed_y, height=height, prominence=prominence, distance=distance)
    
    #print(x_values[peaks])

    # Map the peak positions to energy values and filter out those below the minimum energy in the dataset
    min_energy = np.min(energy)  # Get the lowest energy from the sorted list
    peak_energies = x_values[peaks]  # Assume that `x_values` represent energy bins or calibrated channel positions
    
    #print(peak_energies)
    
    # Filtering out peaks lower than the minimum energy
    valid_peaks = peak_energies[peak_energies >= min_energy]
    
    #print(valid_peaks)

    # Print out the number of valid peaks
    num_peaks = len(valid_peaks)
    #print(f"Number of peaks identified (above minimum energy threshold): {num_peaks}")

    if valid_peaks.any():
        ratio_first = energy[0] / valid_peaks[0]
        ratio_last = energy[-1] / valid_peaks[-1]
        ratio = (ratio_first + ratio_last) / 2
    else:
        ratio = None  # No valid peaks found



    # Step 2: Guess the peak positions based on the ratio
    guessed_positions = energy / ratio

    # Step 3: Filter out energy peaks that don't have corresponding detected peaks
    valid_peak_positions = []
    valid_energy = []

    for i, guess in enumerate(guessed_positions):
        # Find the closest detected peak to the guessed position
        closest_peak_idx = np.argmin(np.abs(peak_energies - guess))
        closest_peak_position = peak_energies[closest_peak_idx]

        # If the detected peak is within the tolerance, keep it
        if np.abs(closest_peak_position - guess) <= tolerance:
            valid_peak_positions.append(closest_peak_position)
            valid_energy.append(energy[i])

    valid_peak_positions = np.array(valid_peak_positions)
    valid_energy = np.array(valid_energy)
    
    #print(valid_peak_positions)

    # Create a figure for centroids fits
    num_plots = len(valid_energy)  # Number of subplots needed
    ncols = 4  # Number of columns for subplots
    nrows = int(np.ceil(num_plots / ncols))  # Calculate number of rows

    # Create subplots for each valid peak fit
    if drawhists==True:
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

    mean_values = []

    for i, e in enumerate(valid_peak_positions):
        roi_width = tolerance / 2
        roi_mask = (x_values > e - roi_width) & (x_values < e + roi_width)
        x_roi = x_values[roi_mask]
        y_roi = y_values[roi_mask]

        initial_guess = [np.max(y_roi), e, 1.0, 0, np.min(y_roi)]

        try:
            popt, _ = curve_fit(_gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            mean_values.append(mu)  # Store the mean value
            if drawhists==True:
                ax = axes[i]
                ax.step(x_roi, y_roi, label='Data', where='post', color='blue', alpha=0.7, linewidth=1)
                ax.plot(x_roi, _gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
                ax.axvline(mu, color='g', linestyle='--', label=f'Centroid: {mu:.2f}')
                ax.set_title(f'Fit around {e:.2f} keV')
                ax.set_xlabel('Channel')
                ax.set_ylabel(f'Counts [{x_values[1]} keV/ch]')
                ax.legend()

        except RuntimeError:
            print(f"Fit could not be performed for peak around {e:.2f} keV")

    # Convert mean values list to a NumPy array
    mean_values_array = np.array(mean_values)


    # Step 4: Perform quadratic fit between valid_peak_positions and valid_energy
    initial_guesses2 = [0.001, ratio, -1]
    popt, pcov = curve_fit(quadratic, mean_values_array, valid_energy, p0=initial_guesses2)
    perr = np.sqrt(np.diag(pcov))  # Standard deviation of the parameters
    print(f"Quadratic fit coefficients: a={popt[0]:.6f}, b={popt[1]:.6f}, c={popt[2]:.6f}")


    if drawhists==True:
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.7)
        # Create a new figure for the quadratic fit and differences
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

        # Plot the quadratic fit of valid_peak_positions vs valid_energy
        plt.subplot(2, 1, 1)  # Top subplot
        plt.scatter(mean_values_array, valid_energy, label='Data Points', color='blue')
        x_fit = np.linspace(np.min(valid_peak_positions), np.max(valid_peak_positions), 100)
        y_fit = quadratic(x_fit, *popt)

        # Calculate error bounds for the fit
        y_fit_upper = quadratic(x_fit, popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2])
        y_fit_lower = quadratic(x_fit, popt[0] - perr[0], popt[1] - perr[1], popt[2] - perr[2])

        plt.plot(x_fit, y_fit, 'r-', label='Fit')
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.3, label='Confidence Interval (1σ)')
        plt.title('Quadratic Fit of Valid Peak Positions vs. Energy')
        plt.ylabel('Peak Positions [Channels]')
        plt.xlim(0, mean_values_array[-1]+100)
        plt.gca().set_xticks([])
        plt.legend()

        # Calculate the calibrated points using the quadratic fit
        calibrated_points = quadratic(valid_peak_positions, *popt)

        # Calculate the differences between calibrated points and reference points
        differences = calibrated_points - valid_energy

        # Plot differences as a function of energy
        plt.subplot(2, 1, 2)  # Bottom subplot
        plt.scatter(mean_values_array, differences, label='Differences', color='orange')
        plt.axhline(0, color='black', linestyle='--', label='Zero Difference')

        plt.fill_between(x_fit, -0.3, 0.3, color='red', alpha=0.3)
        #plt.axhline(0.3, color='red', linestyle='-')
        #plt.axhline(-0.3, color='red', linestyle='-')    
        plt.xlabel('Channel')
        plt.ylabel('Difference [keV]')
        plt.ylim(-1, 1)  # Adjust y limits to make the subplot shorter
        plt.xlim(0, mean_values_array[-1]+100)
        plt.legend()
        plt.show()
    
    return popt


def recalibrate_energy(data, height=None, prominence=500000, distance=10, tolerance=5, drawhists=True):
    
    if data.ndim == 1:
        x_values = np.arange(len(data))  # Generate x-values starting from 0
        y_values = data
    elif data.shape[1] == 2:
        x_values = data[:, 0]
        y_values = data[:, 1]
    else:
        raise ValueError("Data must contain either one or two columns")

    # Find peaks in the data
    smoothed_y = savgol_filter(y_values, window_length=5, polyorder=2)
    peaks, _ = find_peaks(smoothed_y, height=height, prominence=prominence, distance=distance)
    #print("Smoothed\n")

    # Print the peak positions (energies)
    peak_positions = x_values[peaks]
    #print(f"Peaks found at positions: {peak_positions}")

    # Getting energy and related data from 152Eu and 133Ba isotopes
    energy_152Eu, error_energy_152Eu, intensity_152Eu, error_intensity_152Eu = _info_152Eu()
    energy_133Ba, error_energy_133Ba, intensity_133Ba, error_intensity_133Ba = _info_133Ba()

    # Combine energy data from both sources
    unsorted_energy = np.concatenate((energy_152Eu, energy_133Ba))
    unsorted_error_energy = np.concatenate((error_energy_152Eu, error_energy_133Ba))
    unsorted_intensity = np.concatenate((intensity_152Eu, intensity_133Ba))
    unsorted_error_intensity = np.concatenate((error_intensity_152Eu, error_intensity_152Eu))

    # Sort the energy data
    sorted_indices = np.argsort(unsorted_energy)
    energy = unsorted_energy[sorted_indices]
    error_energy = unsorted_error_energy[sorted_indices]
    intensity = unsorted_intensity[sorted_indices]
    error_intensity = unsorted_error_intensity[sorted_indices]

    # Step 1: Estimate the initial conversion ratio using the last peak
    ratio = energy[-1] / peak_positions[-1]

    # Step 2: Guess the peak positions based on the ratio
    guessed_positions = energy / ratio

    # Step 3: Filter out energy peaks that don't have corresponding detected peaks
    valid_peak_positions = []
    valid_energy = []

    for i, guess in enumerate(guessed_positions):
        # Find the closest detected peak to the guessed position
        closest_peak_idx = np.argmin(np.abs(peak_positions - guess))
        closest_peak_position = peak_positions[closest_peak_idx]

        # If the detected peak is within the tolerance, keep it
        if np.abs(closest_peak_position - guess) <= tolerance:
            valid_peak_positions.append(closest_peak_position)
            valid_energy.append(energy[i])

    valid_peak_positions = np.array(valid_peak_positions)
    valid_energy = np.array(valid_energy)

    # Create a figure for centroids fits
    num_plots = len(valid_energy)  # Number of subplots needed
    ncols = 4  # Number of columns for subplots
    nrows = int(np.ceil(num_plots / ncols))  # Calculate number of rows

    # Create subplots for each valid peak fit
    if drawhists==True:
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Define the fitting region around each energy peak
    for i, e in enumerate(valid_energy):
        # Define a region of interest (ROI) around the energy
        roi_width = 7
        roi_mask = (x_values > e - roi_width) & (x_values < e + roi_width)
        x_roi = x_values[roi_mask]
        y_roi = y_values[roi_mask]

        # Initial guess for the parameters A (amplitude), mu (mean), sigma (std dev), m, c
        initial_guess = [np.max(y_roi), e, 1.0, 0, np.min(y_roi)]

        try:
            # Perform the curve fitting using curve_fit
            popt, pcov = curve_fit(_gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            # Plot the fit for visual inspection
            if drawhists==True:
                ax = axes[i]  # Select the current subplot
                ax.step(x_roi, y_roi, label='Data', where='post', color='blue', alpha=0.7, linewidth=1)
                ax.plot(x_roi, _gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
                ax.axvline(mu, color='g', linestyle='--', label=f'Centroid: {mu:.2f}')
                ax.set_title(f'Fit around {e:.2f} keV')
                ax.set_xlabel('Energy [keV]')
                ax.set_ylabel(f'Counts [{x_values[1]} keV/ch]')
                ax.legend()

        except RuntimeError:
            print(f"Fit could not be performed for peak around {e:.2f} keV")


    # Perform curve fitting with initial parameter guesses
    popt, pcov = curve_fit(quadratic, valid_peak_positions, valid_energy)
    perr = np.sqrt(np.diag(pcov))  # Standard deviation of the parameters
    print(f"Quadratic fit coefficients: a={popt[0]:.6f}, b={popt[1]:.6f}, c={popt[2]:.6f}")

    if drawhists==True:
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.7)
        # Create a new figure for the quadratic fit and differences
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

        # Plot the quadratic fit of valid_peak_positions vs valid_energy
        plt.subplot(2, 1, 1)  # Top subplot
        plt.scatter(valid_energy, valid_peak_positions, label='Data Points', color='blue')
        x_fit = np.linspace(np.min(valid_peak_positions), np.max(valid_peak_positions), 100)
        y_fit = quadratic(x_fit, *popt)

        # Calculate error bounds for the fit
        y_fit_upper = quadratic(x_fit, popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2])
        y_fit_lower = quadratic(x_fit, popt[0] - perr[0], popt[1] - perr[1], popt[2] - perr[2])

        plt.plot(x_fit, y_fit, 'r-', label='Fit')
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.3, label='Confidence Interval (1σ)')
        plt.title('Quadratic Fit of Valid Peak Positions vs. Energy')
        plt.ylabel('Peak Positions [Channels]')
        plt.xlim(0, valid_energy[-1]+100)
        plt.gca().set_xticks([])
        plt.legend()

        # Calculate the calibrated points using the quadratic fit
        calibrated_points = quadratic(valid_peak_positions, *popt)

        # Calculate the differences between calibrated points and reference points
        differences = calibrated_points - valid_energy

        # Plot differences as a function of energy
        plt.subplot(2, 1, 2)  # Bottom subplot
        plt.scatter(valid_energy, differences, label='Differences', color='orange')
        plt.axhline(0, color='black', linestyle='--', label='Zero Difference')

        plt.fill_between(x_fit, -0.3, 0.3, color='red', alpha=0.3)
        #plt.axhline(0.3, color='red', linestyle='-')
        #plt.axhline(-0.3, color='red', linestyle='-')    
        plt.xlabel('Energy [keV]')
        plt.ylabel('Difference [keV]')
        plt.ylim(-1, 1)  # Adjust y limits to make the subplot shorter
        plt.xlim(0, valid_energy[-1]+100)
        plt.legend()
        plt.show()
    
    return popt
        
    
def calibrate_energy_final(data, height=None, prominence=5000, distance=10, tolerance=30):
    popt = calibrate_energy(data, height, prominence, distance, tolerance, False)
    popt2 = recalibrate_energy(data, height, prominence, distance, tolerance, True)
    popt_final = [popt[0]*(1+popt2[0]), popt[1]*(1+popt2[1]), popt[2]*(1+popt2[2])]
    return popt_final
