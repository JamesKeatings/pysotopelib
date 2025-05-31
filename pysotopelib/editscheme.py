import os

def add_band(filename, bandnumber):
    isotope_line = ""
    lines = []

    # Check if file exists
    if not os.path.exists(filename):
        # Prompt for isotope information
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

    # Check if band already exists
    if any(line.startswith(band_label) for line in lines):
        print(f"'{band_label}' already exists in the file.")
        return

    # Find where to insert the band line (before '# Transitions' if it exists)
    insert_index = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == "# Transitions":
            insert_index = i
            break

    # Insert the new band line
    lines.insert(insert_index, f"{band_label} \n")

    # Write back to file
    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f"Added '{band_label}' to file.")
        

def add_level(filename, bandnumber, spin, parity, energy):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return

    band_label = f"Band{bandnumber}:"
    level_entry = f"({spin}$^{{{'+' if parity else '-'}}}$) {energy}"

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the band line
    found = False
    for i, line in enumerate(lines):
        if line.startswith(band_label):
            found = True
            line = line.strip()
            existing_levels = line[len(band_label):].split(',')

            # Check for duplicate energy
            for level in existing_levels:
                level = level.strip()
                if level and level[-1].isdigit():
                    try:
                        existing_energy = int(level.split()[-1])
                        if existing_energy == energy:
                            print(f"Level with energy {energy} already exists in '{band_label}'.")
                            return
                    except ValueError:
                        continue  # Skip malformed entries

            # Add the level
            if line.endswith(':'):
                lines[i] = f"{band_label} {level_entry}\n"
            else:
                lines[i] = f"{line}, {level_entry}\n"
            break

    if not found:
        print(f"Error: '{band_label}' not found in the file.")
        return

    # Write changes back
    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f"Added level to '{band_label}': {level_entry}")    
    

def add_transition(filename, start_band, final_band, start_energy, final_energy, intensity):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return

    if start_energy <= final_energy:
        print("Error: Starting energy must be higher than final energy.")
        return

    transition_line = f"{start_energy} {final_energy} Band{start_band} Band{final_band} {intensity}\n"

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Check for start and final levels in their respective bands
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

    # Find the line index of '# Transitions'
    transition_index = None
    for i, line in enumerate(lines):
        if line.strip() == "# Transitions":
            transition_index = i
            break

    # Add '# Transitions' if missing
    if transition_index is None:
        lines.append("# Transitions\n")
        lines.append(transition_line)
    else:
        # Insert the transition line after the last transition or the header
        insert_index = transition_index + 1
        while insert_index < len(lines) and lines[insert_index].strip() and not lines[insert_index].startswith("#"):
            insert_index += 1
        lines.insert(insert_index, transition_line)

    # Write the updated file
    with open(filename, 'w') as f:
        f.writelines(lines)

    print(f"Added transition: {transition_line.strip()}")


add_band("Th222.txt",1)
add_band("Th222.txt",2)
add_level("Th222.txt", 1, "15/2", True, 368)
add_level("Th222.txt", 1, "19/2", False, 619)
add_level("Th222.txt", 1, "21/2", True, 368)  # Will not be added (duplicate energy)
add_level("Th222.txt", 2, "15/2", True, 368)
add_level("Th222.txt", 2, "19/2", False, 619)
add_transition("Th222.txt", 1, 1, 619, 368, 10)
add_band("Th222.txt",3)
