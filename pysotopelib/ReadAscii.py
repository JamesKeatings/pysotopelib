import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def readascii(filename):
    """
    Reads an ASCII file, checks if it has one or two columns, and returns the data
    as a NumPy array. If the file has one column, it uses the values as y-values
    and generates x-values incrementally starting from 0. If the file has two columns,
    it uses the first column as x-values and the second as y-values. It then plots
    a histogram.

    Parameters:
    filename (str): The name of the file to read.

    Returns:
    np.ndarray: A NumPy array with the contents of the columns.
    """
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
    plt.hist(x_values, bins=len(x_values), weights=y_values, color='blue', alpha=0.7)
    
    # Set the x-axis range manually
    plt.xlim(x_min, x_max)
    
    plt.title('Histogram Using First Column as X and Second Column as Y')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')

    # Show the plot
    plt.show()

    return data

# Gaussian with linear background function
def gaussian_with_background(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + c

def calibrate_efficiency(data):
    """
    Fits Gaussian peaks plus a background to multiple areas of the data
    and extracts the peak centroid and area for each region.

    Parameters:
    data (np.ndarray): The data object containing x and y values in two columns.
    energy (np.ndarray): Array of energies around which to fit the Gaussian peaks.

    Returns:
    list of dict: Each dict contains the 'centroid' and 'area' for each fitted peak.
    """
    x_values = data[:, 0]
    y_values = data[:, 1]

    energy = np.array([121.7824, 244.6989, 344.2811, 411.126, 443.965, 788.903, 867.390, 964.055, 1085.842, 1089.767, 1112.087, 1212.970, 1299.152, 1408.022])
    
    results = []

    # Define the fitting region around each energy peak
    for e in energy:
        # Define a region of interest (ROI) around the energy
        # You may want to adjust the width of the ROI based on your data
        roi_mask = (x_values > e - 10) & (x_values < e + 10)
        x_roi = x_values[roi_mask]
        y_roi = y_values[roi_mask]

        # Initial guess for the parameters A (amplitude), mu (mean), sigma (std dev), m, c
        initial_guess = [np.max(y_roi), e, 1.0, 0, np.min(y_roi)]

        try:
            # Perform the curve fitting
            popt, _ = curve_fit(gaussian_with_background, x_roi, y_roi, p0=initial_guess)
            A, mu, sigma, m, c = popt

            # Calculate the area under the Gaussian peak
            area = A * sigma * np.sqrt(2 * np.pi)

            # Store the results
            results.append({'centroid': mu, 'area': area})

            # Optional: Plot the fit for visual inspection
            plt.figure()
            plt.plot(x_roi, y_roi, 'b-', label='Data')
            plt.plot(x_roi, gaussian_with_background(x_roi, *popt), 'r--', label='Fit')
            plt.axvline(mu, color='g', linestyle='--', label=f'Centroid: {mu:.2f}')
            plt.title(f'Fit around {e:.2f} keV')
            plt.xlabel('Energy (keV)')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()

        except RuntimeError as e:
            print(f"Fit could not be performed for peak around {e} keV: {e}")

    return results


data = readascii("Jurogam_AddBack_Energy_Total_(0.5keV_bin).dat")
calibrate_efficiency(data)
