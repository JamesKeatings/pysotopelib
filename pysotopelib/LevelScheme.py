import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import os

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

def add_band(filename, bandnumber):
    isotope_line = ""
    lines = []

    # CHECK FILE EXISTS
    if not os.path.exists(filename):
        # ISOTOPE INFORMATION FOR LABEL
        mass = input("Enter atomic mass number: ")
        element = input("Enter element symbol: ")
        isotope_line = f"$^{{{mass}}}${element}\n"
        lines = [isotope_line, "# Levels\n"]
        with open(filename, 'w') as f:
            f.writelines(lines)
        print(f"Created file '{filename}' with isotope header and level section.")
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()

    band_label = f"Band{bandnumber}:"

    # CHECK BAND EXISTS
    if any(line.startswith(band_label) for line in lines):
        print(f"'{band_label}' already exists in the file.")
        return

    # CHECK FOR TRANSITIONS FLAG
    insert_index = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == "# Transitions":
            insert_index = i
            break

    # ADD BAND
    lines.insert(insert_index, f"{band_label} \n")

    # WRITE TO FILE
    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f"Added '{band_label}' to file.")
        

def add_level(filename, bandnumber, spin, parity, energy):
    # CHECK FILE
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return

    band_label = f"Band{bandnumber}:"
    level_entry = f"({spin}$^{{{'+' if parity else '-'}}}$) {energy}"

    with open(filename, 'r') as f:
        lines = f.readlines()

    # FIND BAND
    found = False
    for i, line in enumerate(lines):
        if line.startswith(band_label):
            found = True
            line = line.strip()
            existing_levels = line[len(band_label):].split(',')

            # CHECK LEVEL EXISTS
            for level in existing_levels:
                level = level.strip()
                if level and level[-1].isdigit():
                    try:
                        existing_energy = int(level.split()[-1])
                        if existing_energy == energy:
                            print(f"Level with energy {energy} already exists in '{band_label}'.")
                            return
                    except ValueError:
                        continue  # SKIP

            # ADD LEVEL
            if line.endswith(':'):
                lines[i] = f"{band_label} {level_entry}\n"
            else:
                lines[i] = f"{line}, {level_entry}\n"
            break

    if not found:
        print(f"Error: '{band_label}' not found in the file.")
        return

    # WRITE FILE
    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f"Added level to '{band_label}': {level_entry}")    
    

def add_transition(filename, start_band, final_band, start_energy, final_energy, intensity):
    # CHECK FILE
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return

    # CHECK VALID ENERGY
    if start_energy <= final_energy:
        print("Error: Starting energy must be higher than final energy.")
        return

    # FORMAT LINE
    transition_line = f"{start_energy} {final_energy} Band{start_band} Band{final_band} {intensity}\n"

    with open(filename, 'r') as f:
        lines = f.readlines()

    # CHECK LEVELS EXIST
    def band_has_energy(band, energy):
        label = f"Band{band}:"
        for line in lines:
            if line.startswith(label):
                level_entries = line[len(label):].split(',')
                for entry in level_entries:
                    entry = entry.strip()
                    if entry.endswith(str(energy)):
                        return True
        return False

    if not band_has_energy(start_band, start_energy):
        print(f"Error: Band{start_band} does not contain a level with energy {start_energy}.")
        return

    if not band_has_energy(final_band, final_energy):
        print(f"Error: Band{final_band} does not contain a level with energy {final_energy}.")
        return

    # FIND TRANSITIONS FLAG
    transition_index = None
    for i, line in enumerate(lines):
        if line.strip() == "# Transitions":
            transition_index = i
            break

    # ADD TRANSITIONS FLAG IF DOES NOT EXIST
    if transition_index is None:
        lines.append("# Transitions\n")
        lines.append(transition_line)
    else:
        # ADD LINE
        insert_index = transition_index + 1
        while insert_index < len(lines) and lines[insert_index].strip() and not lines[insert_index].startswith("#"):
            insert_index += 1
        lines.insert(insert_index, transition_line)

    # WRITE FILE
    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f"Added transition: {transition_line.strip()}")


def delete_band(filename, bandnumber):
    # CHECK FILE
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return

    band_label = f"Band{bandnumber}:"
    band_token = f"Band{bandnumber}"

    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_transitions = False
    transition_header_found = False

    for line in lines:
        stripped = line.strip()

        # REMOVE BAND DEFINITION LINE
        if stripped.startswith(band_label):
            continue

        # REMOVE TRANSITIONS
        if stripped == "# Transitions":
            in_transitions = True
            transition_header_found = True
            new_lines.append(line)
            continue

        if in_transitions:
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
            parts = stripped.split()
            if len(parts) == 5 and (parts[2] == band_token or parts[3] == band_token):
                # SKIP TRANSITION USING DELETED BAND
                continue
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # WRITE FILE
    with open(filename, 'w') as f:
        f.writelines(new_lines)

    print(f"Deleted '{band_label}' and any transitions involving it.")


def delete_level(filename, bandnumber, spin, parity, energy):
    # CHECK FILE
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return

    band_label = f"Band{bandnumber}:"
    band_token = f"Band{bandnumber}"
    energy = int(round(energy))

    # FORMAT LINE
    level_to_remove = f"({spin}$^{{{'+' if parity else '-'}}}$) {energy}"

    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    band_found = False
    level_removed = False
    in_transitions = False

    # FIND BAND LINE
    for line in lines:
        stripped = line.strip()

        # MODIFY BAND DEFINITION LINE
        if stripped.startswith(band_label):
            band_found = True
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                levels = parts[1].split(",")
                cleaned_levels = [lvl.strip() for lvl in levels if lvl.strip() != level_to_remove]
                if len(cleaned_levels) < len(levels):
                    level_removed = True
                new_line = f"{band_label} {', '.join(cleaned_levels)}\n" if cleaned_levels else f"{band_label}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
            continue

        # REMOVE TRANSITIONS
        if stripped == "# Transitions":
            in_transitions = True
            new_lines.append(line)
            continue

        if in_transitions:
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
            parts = stripped.split()
            if len(parts) == 5:
                e_start = int(parts[0])
                e_final = int(parts[1])
                b_start = parts[2]
                b_final = parts[3]

                if (b_start == band_token and e_start == energy) or (b_final == band_token and e_final == energy):
                    # SKIP TRANSITION USING DELETED BAND
                    continue

            new_lines.append(line)
        else:
            new_lines.append(line)

    if not band_found:
        print(f"Band{bandnumber} not found.")
        return

    if not level_removed:
        print(f"Level '{level_to_remove}' not found in Band{bandnumber}.")
        return

    # WRITE FILE
    with open(filename, 'w') as f:
        f.writelines(new_lines)

    print(f"Removed level '{level_to_remove}' and any transitions involving it.")


def delete_transition(filename, start_band, final_band, start_energy, final_energy):
    # CHECK FILE
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return

    start_band_token = f"Band{start_band}"
    final_band_token = f"Band{final_band}"
    start_energy = int(round(start_energy))
    final_energy = int(round(final_energy))

    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_transitions = False
    removed = False

    for line in lines:
        stripped = line.strip()

        if stripped == "# Transitions":
            in_transitions = True
            new_lines.append(line)
            continue

        if in_transitions:
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue

            parts = stripped.split()
            if len(parts) == 5:
                e_start, e_final = int(parts[0]), int(parts[1])
                b_start, b_final = parts[2], parts[3]

                if (e_start == start_energy and
                    e_final == final_energy and
                    b_start == start_band_token and
                    b_final == final_band_token):
                    removed = True
                    continue  # SKIP MATCHING TRANSITION

            new_lines.append(line)
        else:
            new_lines.append(line)
            
    # WRITE FILE
    if removed:
        with open(filename, 'w') as f:
            f.writelines(new_lines)
        print(f"Deleted transition: {start_energy} {final_energy} {start_band_token} {final_band_token}")
    else:
        print("Transition not found. Nothing deleted.")

# TEST LEVEL SCHEME EDITS
def example_level_scheme_edit(filename="Example_edit.txt"):
	add_band(filename,1)
	add_band(filename,2)
	add_level(filename, 1, "15/2", True, 368)
	add_level(filename, 1, "19/2", False, 619)
	add_level(filename, 1, "21/2", True, 368)  # Will not be added (duplicate energy)
	add_level(filename, 2, "15/2", True, 368)
	add_level(filename, 2, "19/2", False, 619)
	add_transition(filename, 1, 1, 619, 368, 10)
	add_band(filename,3)
	delete_transition(filename, 1, 1, 619, 368)
	add_transition(filename, 1, 1, 619, 368, 10)
	delete_level(filename, 2, "19/2", False, 619)
	add_transition(filename, 1, 1, 619, 368, 10)
	delete_band(filename,2)
	

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

