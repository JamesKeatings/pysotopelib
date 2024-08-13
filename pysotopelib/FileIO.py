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
    
def writeascii(data, filename)
