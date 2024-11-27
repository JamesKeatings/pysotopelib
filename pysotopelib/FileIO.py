import numpy as np
import matplotlib.pyplot as plt
import re

# READ FROM ASCII
def readascii(filename):
    # READ FILE INTO NUMPY ARRAY
    data = np.loadtxt(filename)

    # CHECK IF 1 COLUMN OR 2
    if data.ndim == 1:
        y_values = data
        x_values = np.arange(len(y_values))
    elif data.shape[1] == 2:
        x_values = data[:, 0]
        y_values = data[:, 1]
    else:
        raise ValueError("The file must contain either one or two columns")

    # RANGE FROM MIN AND MAX VALUES
    x_min = x_values.min()
    x_max = x_values.max()

    # PLOT HISTOGRAM
    plt.figure(figsize=(8, 6))
    plt.step(x_values, y_values, where='mid', color='blue', alpha=0.7, linewidth=1)  
    
    # SET X RANGE
    plt.xlim(x_min, x_max)
    
    plt.xlabel('Energy [keV]')
    plt.ylabel(f'Counts [{x_values[1]} keV/ch]')

    # SHOW PLOT
    plt.show()

    return data

# READ FROM ROOT C FILE
def readrootmarco(filename):
    # CHECK EXTENSION
    if not filename.endswith('.C'):
        raise ValueError("The file must have a .C extension")

    data = []

    # REGEX TO SEARCH
    pattern = re.compile(r'->SetBinContent\((\d+),\s*(\d+)\);')

    # OPEN FILE AND SEARCH
    with open(filename, 'r') as file:
        for line in file:
            # SEARCH REGEX
            match = pattern.search(line)
            if match:
                # EXTRACT VALUES
                data.append([match.group(1), match.group(2)])
                
    # RANGE FROM MIN AND MAX VALUES
    x_min = data[:, 0].min()
    x_max = data[:, 0].max()

    # PLOT HISTOGRAM
    plt.figure(figsize=(8, 6))
    plt.step(data[:, 0], data[:, 1], where='mid', color='blue', alpha=0.7, linewidth=1)  
    
    # SET X RANGE
    plt.xlim(x_min, x_max)
    
    plt.xlabel('Energy [keV]')
    plt.ylabel(f'Counts [{data[1][0]} keV/ch]')

    # SHOW PLOT
    plt.show()
    return data


# WRITE TO ASCII
def writeascii(data, filename):
    # OPEN OUTFILE
    with open(filename, 'w') as file:
        # CHECK DIMENSIONS OF ARRAY
        if data.ndim == 1:
            for item in data:
                file.write(f"{item}\n")
        elif data.ndim == 2:
            for row in data:
                file.write(f"{row[0]} {row[1]}\n")
        else:
            raise ValueError("Data must be either 1D or 2D.")


# WRITE TO ROOT MACRO
def writerootmacro(data, filename):
    # CHECK DIMENSIONS OF ARRAY
    if data.ndim != 1 or data.ndeim != 2:
        raise ValueError("Data must be either 1D or 2D.")
    else:
        with open(filename, 'w') as file:
            # DISCLAIMER
            file.write("// File automatically generated by pysotopelib\n\n")
            
            # FUNCTION DEFINITION
            file.write("void unnamed()\n")
            file.write("{\n")

            # CANVAS DEFINITION
            file.write("\tTCanvas *c1 = new TCanvas(\"c1\",\"c1\");\n\n")

            #CHECK DIMESIONS OF ARRAY
            if data.ndim == 1:
                file.write(f"\tTH1D {file} = new TH1D(\"{file}\",\"{file}\",{len(data)},{np.min(data)},{len(data)});\n")
                i = np.min(data)
                for item in data:
                    file.write(f"\t{file}->SetBinContent({i},{item});\n")
                    i=i+1
            elif data.ndim == 2:
                file.write(f"\tTH1D {file} = new TH1D(\"{file}\",\"{file}\",{len(data)},{np.min(data)},{np.max(data)});\n")
                for row in data:
                    file.write(f"\t{file}->SetBinContent({row[0]},{row[1]});\n")
            file.write("\n}")
