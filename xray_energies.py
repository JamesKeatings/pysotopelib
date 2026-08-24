#!/usr/bin/env python3
"""
xray_energies.py

Look up characteristic x-ray emission-line energies for a given element,
transcribed from the LBNL X-Ray Data Booklet, Table 1-2 ("Photon energies,
in electron volts, of principal K-, L-, and M-shell emission lines"),
covering elements 3 <= Z <= 95.
Source: https://xdb.lbl.gov/Section1/Table_1-2.pdf

Only the strongest line of each type is tabulated: Ka1, Ka2, Kb1, La1, La2,
Lb1, Lb2, Lg1, Ma1. Not every element has every line -- fainter elements
(e.g. Li-F) only show Ka1/Ka2, and heavier elements only develop L- and
M-lines once the corresponding shells are populated. Missing lines are
simply not printed.

Usage:
    python3 xray_energies.py Eu
    python3 xray_energies.py 63        (lookup by atomic number also works)
    python3 xray_energies.py --list
"""

import sys
import argparse

# LBNL X-Ray Data Booklet, Table 1-2: photon energies (eV) of principal
# K-, L-, and M-shell x-ray emission lines, for elements 3 <= Z <= 95.
# Source: https://xdb.lbl.gov/Section1/Table_1-2.pdf
# Columns: Ka1, Ka2, Kb1, La1, La2, Lb1, Lb2, Lg1, Ma1 (eV). Missing lines are None.
XRAY_LINE_DATA = {
    "Li": {"Z": 3, "lines": [54.3, None, None, None, None, None, None, None, None]},
    "Be": {"Z": 4, "lines": [108.5, None, None, None, None, None, None, None, None]},
    "B": {"Z": 5, "lines": [183.3, None, None, None, None, None, None, None, None]},
    "C": {"Z": 6, "lines": [277, None, None, None, None, None, None, None, None]},
    "N": {"Z": 7, "lines": [392.4, None, None, None, None, None, None, None, None]},
    "O": {"Z": 8, "lines": [524.9, None, None, None, None, None, None, None, None]},
    "F": {"Z": 9, "lines": [676.8, None, None, None, None, None, None, None, None]},
    "Ne": {"Z": 10, "lines": [848.6, 848.6, None, None, None, None, None, None, None]},
    "Na": {"Z": 11, "lines": [1040.98, 1040.98, 1071.1, None, None, None, None, None, None]},
    "Mg": {"Z": 12, "lines": [1253.60, 1253.60, 1302.2, None, None, None, None, None, None]},
    "Al": {"Z": 13, "lines": [1486.70, 1486.27, 1557.45, None, None, None, None, None, None]},
    "Si": {"Z": 14, "lines": [1739.98, 1739.38, 1835.94, None, None, None, None, None, None]},
    "P": {"Z": 15, "lines": [2013.7, 2012.7, 2139.1, None, None, None, None, None, None]},
    "S": {"Z": 16, "lines": [2307.84, 2306.64, 2464.04, None, None, None, None, None, None]},
    "Cl": {"Z": 17, "lines": [2622.39, 2620.78, 2815.6, None, None, None, None, None, None]},
    "Ar": {"Z": 18, "lines": [2957.70, 2955.63, 3190.5, None, None, None, None, None, None]},
    "K": {"Z": 19, "lines": [3313.8, 3311.1, 3589.6, None, None, None, None, None, None]},
    "Ca": {"Z": 20, "lines": [3691.68, 3688.09, 4012.7, 341.3, 341.3, 344.9, None, None, None]},
    "Sc": {"Z": 21, "lines": [4090.6, 4086.1, 4460.5, 395.4, 395.4, 399.6, None, None, None]},
    "Ti": {"Z": 22, "lines": [4510.84, 4504.86, 4931.81, 452.2, 452.2, 458.4, None, None, None]},
    "V": {"Z": 23, "lines": [4952.20, 4944.64, 5427.29, 511.3, 511.3, 519.2, None, None, None]},
    "Cr": {"Z": 24, "lines": [5414.72, 5405.509, 5946.71, 572.8, 572.8, 582.8, None, None, None]},
    "Mn": {"Z": 25, "lines": [5898.75, 5887.65, 6490.45, 637.4, 637.4, 648.8, None, None, None]},
    "Fe": {"Z": 26, "lines": [6403.84, 6390.84, 7057.98, 705.0, 705.0, 718.5, None, None, None]},
    "Co": {"Z": 27, "lines": [6930.32, 6915.30, 7649.43, 776.2, 776.2, 791.4, None, None, None]},
    "Ni": {"Z": 28, "lines": [7478.15, 7460.89, 8264.66, 851.5, 851.5, 868.8, None, None, None]},
    "Cu": {"Z": 29, "lines": [8047.78, 8027.83, 8905.29, 929.7, 929.7, 949.8, None, None, None]},
    "Zn": {"Z": 30, "lines": [8638.86, 8615.78, 9572.0, 1011.7, 1011.7, 1034.7, None, None, None]},
    "Ga": {"Z": 31, "lines": [9251.74, 9224.82, 10264.2, 1097.92, 1097.92, 1124.8, None, None, None]},
    "Ge": {"Z": 32, "lines": [9886.42, 9855.32, 10982.1, 1188.00, 1188.00, 1218.5, None, None, None]},
    "As": {"Z": 33, "lines": [10543.72, 10507.99, 11726.2, 1282.0, 1282.0, 1317.0, None, None, None]},
    "Se": {"Z": 34, "lines": [11222.4, 11181.4, 12495.9, 1379.10, 1379.10, 1419.23, None, None, None]},
    "Br": {"Z": 35, "lines": [11924.2, 11877.6, 13291.4, 1480.43, 1480.43, 1525.90, None, None, None]},
    "Kr": {"Z": 36, "lines": [12649, 12598, 14112, 1586.0, 1586.0, 1636.6, None, None, None]},
    "Rb": {"Z": 37, "lines": [13395.3, 13335.8, 14961.3, 1694.13, 1692.56, 1752.17, None, None, None]},
    "Sr": {"Z": 38, "lines": [14165, 14097.9, 15835.7, 1806.56, 1804.74, 1871.72, None, None, None]},
    "Y": {"Z": 39, "lines": [14958.4, 14882.9, 16737.8, 1922.56, 1920.47, 1995.84, None, None, None]},
    "Zr": {"Z": 40, "lines": [15775.1, 15690.9, 17667.8, 2042.36, 2039.9, 2124.4, 2219.4, 2302.7, None]},
    "Nb": {"Z": 41, "lines": [16615.1, 16521.0, 18622.5, 2165.89, 2163.0, 2257.4, 2367.0, 2461.8, None]},
    "Mo": {"Z": 42, "lines": [17479.34, 17374.3, 19608.3, 2293.16, 2289.85, 2394.81, 2518.3, 2623.5, None]},
    "Tc": {"Z": 43, "lines": [18367.1, 18250.8, 20619, 2424, 2420, 2538, 2674, 2792, None]},
    "Ru": {"Z": 44, "lines": [19279.2, 19150.4, 21656.8, 2558.55, 2554.31, 2683.23, 2836.0, 2964.5, None]},
    "Rh": {"Z": 45, "lines": [20216.1, 20073.7, 22723.6, 2696.74, 2692.05, 2834.41, 3001.3, 3143.8, None]},
    "Pd": {"Z": 46, "lines": [21177.1, 21020.1, 23818.7, 2838.61, 2833.29, 2990.22, 3171.79, 3328.7, None]},
    "Ag": {"Z": 47, "lines": [22162.92, 21990.3, 24942.4, 2984.31, 2978.21, 3150.94, 3347.81, 3519.59, None]},
    "Cd": {"Z": 48, "lines": [23173.6, 22984.1, 26095.5, 3133.73, 3126.91, 3316.57, 3528.12, 3716.86, None]},
    "In": {"Z": 49, "lines": [24209.7, 24002.0, 27275.9, 3286.94, 3279.29, 3487.21, 3713.81, 3920.81, None]},
    "Sn": {"Z": 50, "lines": [25271.3, 25044.0, 28486.0, 3443.98, 3435.42, 3662.80, 3904.86, 4131.12, None]},
    "Sb": {"Z": 51, "lines": [26359.1, 26110.8, 29725.6, 3604.72, 3595.32, 3843.57, 4100.78, 4347.79, None]},
    "Te": {"Z": 52, "lines": [27472.3, 27201.7, 30995.7, 3769.33, 3758.8, 4029.58, 4301.7, 4570.9, None]},
    "I": {"Z": 53, "lines": [28612.0, 28317.2, 32294.7, 3937.65, 3926.04, 4220.72, 4507.5, 4800.9, None]},
    "Xe": {"Z": 54, "lines": [29779, 29458, 33624, 4109.9, None, None, None, None, None]},
    "Cs": {"Z": 55, "lines": [30972.8, 30625.1, 34986.9, 4286.5, 4272.2, 4619.8, 4935.9, 5280.4, None]},
    "Ba": {"Z": 56, "lines": [32193.6, 31817.1, 36378.2, 4466.26, 4450.90, 4827.53, 5156.5, 5531.1, None]},
    "La": {"Z": 57, "lines": [33441.8, 33034.1, 37801.0, 4650.97, 4634.23, 5042.1, 5383.5, 5788.5, 833]},
    "Ce": {"Z": 58, "lines": [34719.7, 34278.9, 39257.3, 4840.2, 4823.0, 5262.2, 5613.4, 6052, 883]},
    "Pr": {"Z": 59, "lines": [36026.3, 35550.2, 40748.2, 5033.7, 5013.5, 5488.9, 5850, 6322.1, 929]},
    "Nd": {"Z": 60, "lines": [37361.0, 36847.4, 42271.3, 5230.4, 5207.7, 5721.6, 6089.4, 6602.1, 978]},
    "Pm": {"Z": 61, "lines": [38724.7, 38171.2, 43826, 5432.5, 5407.8, 5961, 6339, 6892, None]},
    "Sm": {"Z": 62, "lines": [40118.1, 39522.4, 45413, 5636.1, 5609.0, 6205.1, 6586, 7178, 1081]},
    "Eu": {"Z": 63, "lines": [41542.2, 40901.9, 47037.9, 5845.7, 5816.6, 6456.4, 6843.2, 7480.3, 1131]},
    "Gd": {"Z": 64, "lines": [42996.2, 42308.9, 48697, 6057.2, 6025.0, 6713.2, 7102.8, 7785.8, 1185]},
    "Tb": {"Z": 65, "lines": [44481.6, 43744.1, 50382, 6272.8, 6238.0, 6978, 7366.7, 8102, 1240]},
    "Dy": {"Z": 66, "lines": [45998.4, 45207.8, 52119, 6495.2, 6457.7, 7247.7, 7635.7, 8418.8, 1293]},
    "Ho": {"Z": 67, "lines": [47546.7, 46699.7, 53877, 6719.8, 6679.5, 7525.3, 7911, 8747, 1348]},
    "Er": {"Z": 68, "lines": [49127.7, 48221.1, 55681, 6948.7, 6905.0, 7810.9, 8189.0, 9089, 1406]},
    "Tm": {"Z": 69, "lines": [50741.6, 49772.6, 57517, 7179.9, 7133.1, 8101, 8468, 9426, 1462]},
    "Yb": {"Z": 70, "lines": [52388.9, 51354.0, 59370, 7415.6, 7367.3, 8401.8, 8758.8, 9780.1, 1521.4]},
    "Lu": {"Z": 71, "lines": [54069.8, 52965.0, 61283, 7655.5, 7604.9, 8709.0, 9048.9, 10143.4, 1581.3]},
    "Hf": {"Z": 72, "lines": [55790.2, 54611.4, 63234, 7899.0, 7844.6, 9022.7, 9347.3, 10515.8, 1644.6]},
    "Ta": {"Z": 73, "lines": [57532, 56277, 65223, 8146.1, 8087.9, 9343.1, 9651.8, 10895.2, 1710]},
    "W": {"Z": 74, "lines": [59318.24, 57981.7, 67244.3, 8397.6, 8335.2, 9672.35, 9961.5, 11285.9, 1775.4]},
    "Re": {"Z": 75, "lines": [61140.3, 59717.9, 69310, 8652.5, 8586.2, 10010.0, 10275.2, 11685.4, 1842.5]},
    "Os": {"Z": 76, "lines": [63000.5, 61486.7, 71413, 8911.7, 8841.0, 10355.3, 10598.5, 12095.3, 1910.2]},
    "Ir": {"Z": 77, "lines": [64895.6, 63286.7, 73560.8, 9175.1, 9099.5, 10708.3, 10920.3, 12512.6, 1979.9]},
    "Pt": {"Z": 78, "lines": [66832, 65112, 75748, 9442.3, 9361.8, 11070.7, 11250.5, 12942.0, 2050.5]},
    "Au": {"Z": 79, "lines": [68803.7, 66989.5, 77984, 9713.3, 9628.0, 11442.3, 11584.7, 13381.7, 2122.9]},
    "Hg": {"Z": 80, "lines": [70819, 68895, 80253, 9988.8, 9897.6, 11822.6, 11924.1, 13830.1, 2195.3]},
    "Tl": {"Z": 81, "lines": [72871.5, 70831.9, 82576, 10268.5, 10172.8, 12213.3, 12271.5, 14291.5, 2270.6]},
    "Pb": {"Z": 82, "lines": [74969.4, 72804.2, 84936, 10551.5, 10449.5, 12613.7, 12622.6, 14764.4, 2345.5]},
    "Bi": {"Z": 83, "lines": [77107.9, 74814.8, 87343, 10838.8, 10730.91, 13023.5, 12979.9, 15247.7, 2422.6]},
    "Po": {"Z": 84, "lines": [79290, 76862, 89800, 11130.8, 11015.8, 13447, 13340.4, 15744, None]},
    "At": {"Z": 85, "lines": [81520, 78950, 92300, 11426.8, 11304.8, 13876, None, 16251, None]},
    "Rn": {"Z": 86, "lines": [83780, 81070, 94870, 11727.0, 11597.9, 14316, None, 16770, None]},
    "Fr": {"Z": 87, "lines": [86100, 83230, 97470, 12031.3, 11895.0, 14770, 14450, 17303, None]},
    "Ra": {"Z": 88, "lines": [88470, 85430, 100130, 12339.7, 12196.2, 15235.8, 14841.4, 17849, None]},
    "Ac": {"Z": 89, "lines": [90884, 87670, 102850, 12652.0, 12500.8, 15713, None, 18408, None]},
    "Th": {"Z": 90, "lines": [93350, 89953, 105609, 12968.7, 12809.6, 16202.2, 15623.7, 18982.5, 2996.1]},
    "Pa": {"Z": 91, "lines": [95868, 92287, 108427, 13290.7, 13122.2, 16702, 16024, 19568, 3082.3]},
    "U": {"Z": 92, "lines": [98439, 94665, 111300, 13614.7, 13438.8, 17220.0, 16428.3, 20167.1, 3170.8]},
    "Np": {"Z": 93, "lines": [None, None, None, 13944.1, 13759.7, 17750.2, 16840.0, 20784.8, None]},
    "Pu": {"Z": 94, "lines": [None, None, None, 14278.6, 14084.2, 18293.7, 17255.3, 21417.3, None]},
    "Am": {"Z": 95, "lines": [None, None, None, 14617.2, 14411.9, 18852.0, 17676.5, 22065.2, None]},
}

XRAY_LINE_COLUMNS = ["Ka1", "Ka2", "Kb1", "La1", "La2", "Lb1", "Lb2", "Lg1", "Ma1"]

def format_energy(value):
    """1234.5 eV -> '1,234.5' (thousands separator, matching the booklet's
    own formatting), keeping however many decimal places the source gave."""
    if value is None:
        return ""
    return f"{value:,.10g}".rstrip()


def pad_columns(header, rows, sep="\t\t"):
    """Left-justify every column to its widest entry, then join with sep."""
    n_cols = len(header)
    widths = [len(h) for h in header]
    for row in rows:
        for i in range(n_cols):
            widths[i] = max(widths[i], len(row[i]))
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(header))
    row_lines = [
        sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    ]
    return header_line, row_lines


def resolve_element(query):
    """Accept a symbol ('Eu', 'eu') or an atomic number ('63') and return
    the matching symbol, or None if not found."""
    query = query.strip()
    for sym, entry in XRAY_LINE_DATA.items():
        if sym.lower() == query.lower():
            return sym
    if query.isdigit():
        z = int(query)
        for sym, entry in XRAY_LINE_DATA.items():
            if entry["Z"] == z:
                return sym
    return None


def print_report(query, file=sys.stdout):
    sym = resolve_element(query)
    if sym is None:
        print(f"No data found for element '{query}'.", file=file)
        print("Use --list to see all available elements.", file=file)
        return 1

    entry = XRAY_LINE_DATA[sym]
    z = entry["Z"]
    lines = entry["lines"]

    print(f"=== X-ray emission lines: {sym} (Z={z}) ===", file=file)
    header = ["Line", "Energy(eV)"]
    rows = []
    for col_name, value in zip(XRAY_LINE_COLUMNS, lines):
        if value is None:
            continue
        rows.append([col_name, format_energy(value)])
    if not rows:
        print("(no emission lines tabulated for this element)", file=file)
        return 0

    header_line, row_lines = pad_columns(header, rows)
    print(header_line, file=file)
    for line in row_lines:
        print(line, file=file)
    return 0


def list_elements():
    entries = sorted(XRAY_LINE_DATA.items(), key=lambda kv: kv[1]["Z"])
    print("Available elements (ordered by Z):")
    for sym, entry in entries:
        n_lines = sum(1 for v in entry["lines"] if v is not None)
        print(f"  {entry['Z']:>3}  {sym:<3}\t({n_lines} lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Look up LBNL X-Ray Data Booklet (Table 1-2) characteristic "
        "x-ray emission-line energies for an element."
    )
    parser.add_argument(
        "element", nargs="?", help="Element symbol (e.g. Eu) or atomic number (e.g. 63)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available elements and exit"
    )
    args = parser.parse_args()

    if args.list or not args.element:
        list_elements()
        return 0

    return print_report(args.element)


if __name__ == "__main__":
    sys.exit(main())
