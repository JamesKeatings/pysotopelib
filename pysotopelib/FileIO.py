import numpy as np
import matplotlib.pyplot as plt
import re

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

def readrootmarco(filename):
    # Check if the file has a .C extension
    if not filename.endswith('.C'):
        raise ValueError("The file must have a .C extension")

    data = []

    # Compile a regular expression to match the desired pattern
    pattern = re.compile(r'->SetBinContent\((\d+),\s*(\d+)\);')

    # Open the file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            # Search for the pattern in each line
            match = pattern.search(line)
            if match:
                # Extract the two numbers and convert them to integers
                data.append([match.group(1), match.group(2)])
                #number1 = int(match.group(1))
                #number2 = int(match.group(2))
                # Append the numbers as a list to the data array
                #data.append([number1, number2])
                
    # Determine the range based on min and max of x-values
    x_min = data[:, 0].min()
    x_max = data[:, 0].max()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    #plt.hist(x_values, bins=len(x_values), weights=y_values, color='blue', alpha=0.7, linewidth=3)
    plt.step(data[:, 0], data[:, 1], where='mid', color='blue', alpha=0.7, linewidth=1)  
    
    # Set the x-axis range manually
    plt.xlim(x_min, x_max)
    
    plt.xlabel('Energy [keV]')
    plt.ylabel(f'Counts [{data[1][0]} keV/ch]')

    # Show the plot
    plt.show()
    return data

def writeascii(data, filename):
    # Open the file for writing
    with open(filename, 'w') as file:
        # Check if data is 1D
        if data.ndim == 1:
            for item in data:
                file.write(f"{item}\n")
        # Check if data is 2D
        elif data.ndim == 2:
            for row in data:
                file.write(f"{row[0]} {row[1]}\n")
        else:
            raise ValueError("Data must be either 1D or 2D.")
            
def writerootmacro(data, filename):
    if data.ndim != 1 or data.ndeim != 2:
        raise ValueError("Data must be either 1D or 2D.")
    else:
        with open(filename, 'w') as file:
            file.write("// File automatically generated by pysotopelib\n\n")
            file.write("void unnamed()\n")
            file.write("{\n")
            file.write("\tTCanvas *c1 = new TCanvas(\"c1\",\"c1\");\n\n")
            if data.ndim == 1:
                file.write(f"\tTH1D {file} = new TH1D(\"{file}\",\"{file}\",{len(data)},{np.min(data)},{len(data)});\n")
                i = np.min(data)
                for item in data:
                    file.write(f"\t{file}->SetBinContent({i},{item});\n")
                    i=i+1
            # Check if data is 2D
            elif data.ndim == 2:
                file.write(f"\tTH1D {file} = new TH1D(\"{file}\",\"{file}\",{len(data)},{np.min(data)},{np.max(data)});\n")
                for row in data:
                    file.write(f"\t{file}->SetBinContent({row[0]},{row[1]});\n")
            file.write("\n}")
