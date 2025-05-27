import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

# READ DATA FROM FILE
def _read_level_data(filename):
    level_labels = {}
    transitions = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # READ ISOTOPE LABEL AND SKIP LINE
        isotope_label = lines[0].strip()
        lines = lines[1:]
        
        # READ LEVEL DATA
        level_section = True
        for line in lines:
            line = line.strip()
            
            # CHECK FOR TRANSITIONS FLAG
            if line.startswith("# Transitions"):
                level_section = False
                continue
            
            # SKIP COMMENTS OR EMPTY LINES
            if line.startswith("#") or not line:
                continue
            
            if level_section:
                # MATCH BAND NAMES FOR TRANSITIONS
                band_match = re.match(r'^([^:]+): (.+)$', line)
                if band_match:
                    band_name, levels_str = band_match.groups()
                    levels = []
                    for level_str in levels_str.split(','):
                        label, energy = level_str.strip().rsplit(' ', 1)
                        levels.append((label, int(energy)))
                    level_labels[band_name.strip()] = levels
            else:
                # PARSE TRANSITIONS
                parts = line.split()
                if len(parts) == 5:
                    start, end = int(parts[0]), int(parts[1])
                    band_start, band_end = parts[2], parts[3]
                    width = int(parts[4])
                    transitions.append(((start, end), (band_start, band_end), width))
                    
    return isotope_label, level_labels, transitions


# PLOT LEVEL SCHEME
def plot_level_scheme(filename="Example.txt"):
    size_font = 12
    size_level = 0.3
    size_label_buffer = 0.02

    # SETUP FONT
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
    })

    isotope_label, level_labels, transitions = _read_level_data(filename)

    # EXTRACT ENERGY LEVELS
    energy_levels = {band: [level[1] for level in levels] for band, levels in level_labels.items()}

    # CREATE FIGURE
    fig, ax = plt.subplots(figsize=(8, 12))

    # SET UP X AXIS WITH LAVELS
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(energy_levels)))
    ax.set_xticklabels(energy_levels.keys(), fontsize=size_font)

    # Y AXIS TITLE
    ax.set_ylabel("Energy [keV]", fontsize=size_font)

    # SET LABEL TO BOTTOM
    ax.set_xlabel(isotope_label, fontsize=40, labelpad=20)

    # PLOT ENERGY LEVELS
    for i, (band, levels) in enumerate(energy_levels.items()):
        num_levels = len(levels)
        for j in range(num_levels):
            level = levels[j]
            ax.hlines(level, i - size_level, i + size_level, color='black', linewidth=2)
        
            # DETERMINE LABEL POSITIONS
            if j < num_levels - 1:  # CHECK IF LAST LEVEL
                next_level = levels[j + 1]
                if abs(level - next_level) < 100:  # SET BELOW FOR WITHIN 100 KEV
                    label_pos = 'below'
                else:
                    label_pos = 'above'
            else:
                # DEFAULT POSITION FOR LABELS ABOVE
                label_pos = 'above'
        
            # LEFT LABEL POSITION
            left_label, right_label = level_labels[band][j]
            ax.text(i - size_level + size_label_buffer, level - 30 if label_pos == 'below' else level + 10,
                left_label, fontsize=size_font, ha='left')
        
            # RIGHT LABEL POSITION
            ax.text(i + size_level - size_label_buffer, level - 34 if label_pos == 'below' else level + 10,
                f"{right_label}", fontsize=size_font, ha='right')

    # PLOT TRANSITIONS
    for (start, end), (band_start, band_end), width in transitions:
        x_start = list(energy_levels.keys()).index(band_start)
        x_end = list(energy_levels.keys()).index(band_end)
    
        # NORMALISE WIDTH
        arrow_width = (width / 100) * 30
        delta = start - end
    
        if band_start == band_end:
            # VERTICAL TRANSITION
            arrow = ax.annotate('', xy=(x_start, end), xytext=(x_start, start),
                            arrowprops=dict(facecolor='black', shrink=0.01, width=arrow_width, headwidth=3*arrow_width))
        else:
            # DIAGONAL TRANSITIONS BASED ON X POSITION OF INITIA AND FINAL BAND
            if x_start < x_end:
                arrow = ax.annotate('', xy=(x_end - size_level, end), xytext=(x_start + size_level, start),
                                arrowprops=dict(facecolor='black', shrink=0.01, width=arrow_width, headwidth=10*arrow_width))
            else:
                arrow = ax.annotate('', xy=(x_end + size_level, end), xytext=(x_start - size_level, start),
                                arrowprops=dict(facecolor='black', shrink=0.01, width=arrow_width, headwidth=10*arrow_width))
    
        # ADD TRANSITION LABEL WITH BACKGROUND
        mid_x = (x_start + x_end) / 2
        mid_y = (start + end) / 2
        ax.text(mid_x, mid_y, f"{delta}", fontsize=size_font, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    # SET Y AXIS LIMITS
    max_energy = max(max(levels) for levels in energy_levels.values())
    ax.set_ylim(-1, max_energy + 100)

    # HIDE BOX
    for spine in ax.spines.values():
        spine.set_visible(False)

    # HIDE X AXIS
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # ADD SOLID LINE FOR Y AXIS
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1)

    # HIDE TICKS
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='both', right=False, labelsize=size_font)

    # SAVE AS SVG
    #plt.savefig('levelscheme.svg', format='svg')

    # SHOW THE LEVEL SCHEME
    plt.show()

def _extract_numerator(label):
    """Extract the numerator from a spin-parity label like '(9/2$^{-}$)'."""
    match = re.match(r'\(?(\d+)/\d+', label)
    return int(match.group(1)) if match else None

def _build_band_energy_dict(levels):
    """Build dictionary mapping spin numerators to energies."""
    d = {}
    for label, energy in levels:
        num = _extract_numerator(label)
        if num is not None:
            d[num] = energy
    return d

def plot_interleave_e(filename, mainband, altband):
    # Read data from file
    isotope_label, level_labels, transitions = _read_level_data(filename)
    
    band2_levels = level_labels.get(mainband, [])
    band1_levels = level_labels.get(altband, [])
    band1_dict = _build_band_energy_dict(band1_levels)
    
    if len(band2_levels) < 2 or not band1_dict:
        print("Not enough data in Band2 or Band1")
        return

    x_vals = []
    y_deltas = []

    for i in range(len(band2_levels) - 1):
        label1, e1 = band2_levels[i]
        label2, e2 = band2_levels[i + 1]

        # Midpoint energy
        mid_energy = (e1 + e2) / 2
        
        # Midpoint spin numerator
        num1 = _extract_numerator(label1)
        num2 = _extract_numerator(label2)
        if num1 is None or num2 is None:
            continue
        mid_spin_num = (num1 + num2) / 2

        # Round to nearest integer to match Band1 spins
        mid_spin_rounded = round(mid_spin_num)
        
        if mid_spin_rounded in band1_dict:
            band1_energy = band1_dict[mid_spin_rounded]
            delta = band1_energy - mid_energy
            x_vals.append(mid_spin_num)
            y_deltas.append(delta)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_deltas, marker='o')
    plt.xlabel("2J (h)")
    plt.ylabel("ΔE (keV)")
    plt.title(f"{isotope_label}: ΔE({mainband}, {altband}) against 2J")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# PRODUCE EXAMPLE SCHEME FILE
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

