import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

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
    plt.step(x_values, y_values, where='post', color='blue', alpha=0.7, linewidth=1)  
    
    # Set the x-axis range manually
    plt.xlim(x_min, x_max)
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')

    # Show the plot
    plt.show()

    return data


# Gaussian with linear background function
def gaussian_with_background(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + c

def efficiency_function(EG, A, B, D, E, F):
    E1 = 100.0  # Given E1
    E2 = 1000.0 # Given E2
    s = np.log(EG / E1)
    t = np.log(EG / E2)
    return np.exp(( (A + B * s)**(-15) + (D + E * t + F * t**(-15)) )**(-1/15))
    
def efficiency_function2(EG, A, B, C, D, E):
    E0 = 325.0
    return np.exp(A*np.log(EG/E0) + B*np.log(EG/E0)**2 + C*np.log(EG/E0)**3 + D*np.log(EG/E0)**4 + E*np.log(EG/E0)**5)

def cost_function(params, energies, normalized_areas, errors):
    A, B, D, E, F = params
    # Prevent division by zero in error array by replacing zero errors with a small value
    errors = np.clip(errors, 1e-10, None)
    model_values = efficiency_function2(energies, A, B, D, E, F)
    return np.sum(((normalized_areas - model_values) / errors) ** 2)


def calibrate_efficiency(data):
    # Define the energy array within the function
    energy = np.array([80.998, 
        121.7824, 244.6989, 344.2811, 411.126, 443.965, 778.903,
        867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970,
        1299.152, 1408.022
    ])

    # Define the intensity and error_intensity arrays
    intensity = np.array([34.11, 28.37, 7.53, 26.57, 2.238, 3.125, 12.97, 4.214, 14.63, 10.13, 1.731, 13.54, 1.412, 1.626, 20.85])
    error_intensity = np.array([0.28, 0.13, 0.04, 0.11, 0.010, 0.014, 0.06, 0.025, 0.06, 0.05, 0.009, 0.06, 0.008, 0.011, 0.09])

    x_values = data[:, 0]
    y_values = data[:, 1]

    results = []

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
            # Perform the curve fitting
            popt, pcov = curve_fit(gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            # Calculate the area under the Gaussian peak
            area = A * sigma * np.sqrt(2 * np.pi)

            # Store the results along with the corresponding energy and intensity
            results.append({'energy': e, 'centroid': mu, 'area': area, 'intensity': intensity[i], 'error_intensity': error_intensity[i]})

            # Optional: Plot the fit for visual inspection
            plt.figure()
            plt.step(x_roi, y_roi, 'b-', label='Data', where='post', color='blue', alpha=0.7, linewidth=1)
            plt.plot(x_roi, gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
            plt.axvline(mu, color='g', linestyle='--', label=f'Centroid: {mu:.2f}')
            plt.title(f'Fit around {e:.2f} keV')
            plt.xlabel('Energy (keV)')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()

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
    errors = np.array([result['error_intensity'] for result in results])

    # Normalize areas by intensities
    normalized_areas = normalized_areas / intensities
    errors = errors / intensities

    # Add the point (0, 0) to the data before fitting
    energies = np.insert(energies, 0, 0)
    normalized_areas = np.insert(normalized_areas, 0, 0)
    errors = np.insert(errors, 0, 0.1)  # Use a small nonzero error for (0,0)

    # Initial guess for the parameters A, B, D, E, F
    initial_guess = [2.66, 1.9, 2.08, -0.71, -0.04]

    # Perform the minimization
    result = minimize(cost_function, initial_guess, args=(energies, normalized_areas, errors))

    # Extract the optimized parameters
    A_opt, B_opt, D_opt, E_opt, F_opt = result.x

    # Display the optimized parameters
    print(f"Optimized parameters: A={A_opt}, B={B_opt}, D={D_opt}, E={E_opt}, F={F_opt}")

    # Plot the fitted efficiency function against the data
    plt.figure()
    plt.errorbar(energies, normalized_areas, yerr=errors, fmt='o', label='Data')
    plt.plot(energies, efficiency_function2(energies, A_opt, B_opt, D_opt, E_opt, F_opt), 'r-', label='Fit')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Normalized Efficiency')
    plt.title('Efficiency Calibration')
    plt.legend()
    plt.show()

    return result
