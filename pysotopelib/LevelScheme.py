import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

def read_level_data(filename):
    level_labels = {}
    transitions = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Read the first line as isotope label and remove it from the lines list
        isotope_label = lines[0].strip()
        lines = lines[1:]  # Remove the first line from further processing
        
        # Parse level labels
        level_section = True
        for line in lines:
            line = line.strip()
            
            # Check if the transition section starts
            if line.startswith("# Transitions"):
                level_section = False
                continue
            
            # Skip comments or empty lines
            if line.startswith("#") or not line:
                continue
            
            if level_section:
                # Match any band line as "BandName: label energy, ..."
                band_match = re.match(r'^([^:]+): (.+)$', line)
                if band_match:
                    band_name, levels_str = band_match.groups()
                    levels = []
                    for level_str in levels_str.split(','):
                        label, energy = level_str.strip().rsplit(' ', 1)
                        levels.append((label, int(energy)))
                    level_labels[band_name.strip()] = levels
            else:
                # Parse transitions
                # Example: "368 202 Band1 Band2 2"
                parts = line.split()
                if len(parts) == 5:
                    start, end = int(parts[0]), int(parts[1])
                    band_start, band_end = parts[2], parts[3]
                    width = int(parts[4])
                    transitions.append(((start, end), (band_start, band_end), width))
                    
    return isotope_label, level_labels, transitions

def plot_level_scheme(filename="Example.txt"):
    size_font = 12
    size_level = 0.3
    size_label_buffer = 0.02

    # Use DejaVu Serif as an alternative to Computer Modern
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",  # This will use Computer Modern for mathtext
    })

    isotope_label, level_labels, transitions = read_level_data(filename)

    # Extract energy levels from level_labels
    energy_levels = {band: [level[1] for level in levels] for band, levels in level_labels.items()}

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 12))

    # Set up x-ticks and labels on top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(energy_levels)))
    ax.set_xticklabels(energy_levels.keys(), fontsize=size_font)

    # Set up y-axis label
    ax.set_ylabel("Energy [keV]", fontsize=size_font)

    # Set x-axis title at the bottom
    ax.set_xlabel(isotope_label, fontsize=40, labelpad=20)  # Set labelpad to position it at the bottom

    # Plot energy levels and add labels
    for i, (band, levels) in enumerate(energy_levels.items()):
        num_levels = len(levels)
        for j in range(num_levels):
            level = levels[j]
            ax.hlines(level, i - size_level, i + size_level, color='black', linewidth=2)
        
            # Determine label positions
            if j < num_levels - 1:  # Check if there is a next level
                next_level = levels[j + 1]
                if abs(level - next_level) < 100:  # Adjust for close levels
                    label_pos = 'below'
                else:
                    label_pos = 'above'
            else:
                # For the last level in the list, default to 'above'
                label_pos = 'above'
        
            # Left label (e.g., "1$^{+}$" or "2$^{-}$") - move closer to the middle
            left_label, right_label = level_labels[band][j]
            ax.text(i - size_level + size_label_buffer, level - 30 if label_pos == 'below' else level + 10,
                left_label, fontsize=size_font, ha='left')
        
            # Right label (energy) - move closer to the middle
            ax.text(i + size_level - size_label_buffer, level - 34 if label_pos == 'below' else level + 10,
                f"{right_label}", fontsize=size_font, ha='right')

    # Plot transitions with labels
    for (start, end), (band_start, band_end), width in transitions:
        x_start = list(energy_levels.keys()).index(band_start)
        x_end = list(energy_levels.keys()).index(band_end)
    
        # Normalize width to a maximum of 30
        arrow_width = (width / 100) * 30
        delta = start - end
    
        if band_start == band_end:
            # Intra-band transition (draw from center)
            arrow = ax.annotate('', xy=(x_start, end), xytext=(x_start, start),
                            arrowprops=dict(facecolor='black', shrink=0.01, width=arrow_width, headwidth=3*arrow_width))
        else:
            # Inter-band transition
            if x_start < x_end:
                # Draw from right of start to left of end
                arrow = ax.annotate('', xy=(x_end - size_level, end), xytext=(x_start + size_level, start),
                                arrowprops=dict(facecolor='black', shrink=0.01, width=arrow_width, headwidth=10*arrow_width))
            else:
                # Draw from left of start to right of end
                arrow = ax.annotate('', xy=(x_end + size_level, end), xytext=(x_start - size_level, start),
                                arrowprops=dict(facecolor='black', shrink=0.01, width=arrow_width, headwidth=10*arrow_width))
    
        # Add a label for the transition difference
        mid_x = (x_start + x_end) / 2
        mid_y = (start + end) / 2
        ax.text(mid_x, mid_y, f"{delta}", fontsize=size_font, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    # Calculate the maximum energy level and set y-axis limits
    max_energy = max(max(levels) for levels in energy_levels.values())
    ax.set_ylim(-1, max_energy + 100)  # Set upper limit to the highest energy level + 100 keV

    # Hide the box around the graph
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Hide x-axis ticks and labels at the bottom
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Add a solid line for the y-axis
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1)  # Adjust the thickness if needed

    # Hide y-axis right spine and ticks, but keep y-axis left spine and ticks
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='both', right=False, labelsize=size_font)

    # Save the plot as an SVG file
    #plt.savefig('levelscheme.svg', format='svg')

    # Show the plot
    plt.show()

def example_level_scheme(filename="Example.txt"):

    with open(filename, 'w') as f:
        print(f"Opened file, writing example to {filename}")
        f.write("$^{224}$Ra\n")
        f.write("# Levels\n")
        f.write("Band1: 2$^{+}$ 966\n")
        f.write("Band2: 0$^{+}$ 0, 2$^{+}$ 84, 4$^{+}$ 251, 6$^{+}$ 479, 8$^{+}$ 755, 10$^{+}$ 1067\n")
        f.write("Band3: 1$^{-}$ 216, 3$^{-}$ 290, 5$^{-}$ 433, 7$^{-}$ 641, 9$^{-}$ 906\n")
        f.write("# Transitions\n")
        f.write("966 0 Band1 Band2 2\n")
        f.write("966 84 Band1 Band2 2\n")
        f.write("84 0 Band2 Band2 2\n")
        f.write("251 84 Band2 Band2 2\n")
        f.write("479 251 Band2 Band2 2\n")
        f.write("755 479 Band2 Band2 2\n")
        f.write("1067 755 Band2 Band2 2\n")
        f.write("290 216 Band3 Band3 2\n")
        f.write("433 290 Band3 Band3 2\n")
        f.write("641 433 Band3 Band3 2\n")
        f.write("906 641 Band3 Band3 2\n")
        f.write("216 0 Band3 Band2 2\n")
        f.write("216 84 Band3 Band2 2\n")
        f.write("290 84 Band3 Band2 2\n")
        f.write("433 251 Band3 Band2 2\n")
        f.write("641 479 Band3 Band2 2\n")
        f.write("755 641 Band2 Band3 2\n")
        f.write("906 755 Band3 Band2 2\n")

