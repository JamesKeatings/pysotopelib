#!/usr/bin/env python3
"""
nuclide_xray_gamma.py

Look up X-ray and gamma-ray reference energies/emission probabilities for
calibration nuclides, transcribed from:

    IAEA-TECDOC-619, "X-Ray and Gamma-Ray Standards for Detector
    Calibration", Table 2 (X-ray standards) and Table 3 (gamma-ray
    standards).

Usage:
    python3 nuclide_xray_gamma.py 152Eu
    python3 nuclide_xray_gamma.py 60Co
    python3 nuclide_xray_gamma.py --list

Values in parentheses in the original tables denote the uncertainty in the
last digit(s), e.g. "1274.542(7)" means 1274.542 +/- 0.007. This is parsed
automatically for the gamma-ray table. X-ray energies are sometimes given
as a range (e.g. "4.95-5.43" for a Kx blend) rather than a single value
with an uncertainty, so those are kept as-is.
"""

import sys
import argparse
import re

# ---------------------------------------------------------------------------
# Data transcribed from Table 2 (X-ray standards): energies and emission
# probabilities. Keyed by nuclide, e.g. "51Cr" for Cr-51.
# Each entry: (Transition, Energy keV [str, may be a range], Probability [str])
# ---------------------------------------------------------------------------
XRAY_DATA = {
    "51Cr": [
        ("VKa", "4.95", "0.201(3)"),
        ("VKb", "5.43", "0.027(1)"),
        ("VKx", "4.95-5.43", "0.228(3)"),
    ],
    "54Mn": [
        ("CrKa", "5.41", "0.226(7)"),
        ("CrKb", "5.95", "0.030(1)"),
        ("CrKx", "5.41-5.95", "0.256(8)"),
    ],
    "55Fe": [
        ("MnKa", "5.89", "0.249(9)"),
        ("MnKb", "6.49", "0.034(1)"),
        ("MnKx", "5.89-6.49", "0.283(10)"),
    ],
    "57Co": [
        ("FeKa", "6.40", "0.510(7)"),
        ("FeKb", "7.06", "0.069(1)"),
        ("FeKx", "6.40-7.06", "0.579(8)"),
    ],
    "58Co": [
        ("FeKa", "6.40", "0.235(3)"),
        ("FeKb", "7.06", "0.032(1)"),
        ("FeKx", "6.40-7.06", "0.267(3)"),
    ],
    "65Zn": [
        ("CuKa", "8.03-8.05", "0.341(6)"),
        ("CuKb", "8.91", "0.046(1)"),
        ("CuKx", "8.03-8.91", "0.387(6)"),
    ],
    "75Se": [
        ("AsKa", "10.51-10.54", "0.493(11)"),
        ("AsKb", "11.72-11.95", "0.075(2)"),
        ("AsKx", "10.51-11.95", "0.568(13)"),
    ],
    "85Sr": [
        ("RbKa", "13.34-13.40", "0.500(3)"),
        ("RbKb", "14.96-15.29", "0.087(2)"),
        ("RbKx", "13.34-15.29", "0.587(4)"),
    ],
    "88Y": [
        ("SrKa", "14.10-14.17", "0.522(6)"),
        ("SrKb", "15.83-16.19", "0.094(2)"),
        ("SrKx", "14.10-16.19", "0.616(7)"),
    ],
    "93mNb": [
        ("NbKa", "16.52-16.62", "0.0925(30)"),
        ("NbKb", "18.62-19.07", "0.0179(7)"),
        ("NbKx", "16.52-19.07", "0.1104(35)"),
    ],
    "109Cd": [
        ("AgKa", "21.99-22.16", "0.821(9)"),
        ("AgKb", "24.93-25.60", "0.173(3)"),
        ("AgKx", "21.99-25.60", "0.994(10)"),
    ],
    "111In": [
        ("CdKa", "22.98-23.17", "0.684(5)"),
        ("CdKb", "26.09-26.80", "0.146(3)"),
        ("CdKx", "22.98-26.80", "0.830(5)"),
    ],
    "113Sn": [
        ("InKa", "24.00-24.21", "0.796(6)"),
        ("InKb", "27.27-28.02", "0.172(3)"),
        ("InKx", "24.00-28.02", "0.968(6)"),
    ],
    "125I": [
        ("TeKa", "27.20-27.47", "1.135(21)"),
        ("TeKb", "30.98-31.88", "0.255(6)"),
        ("TeKx", "27.20-31.88", "1.390(25)"),
    ],
    "137Cs": [
        ("BaKa", "31.82-32.19", "0.0566(16)"),
        ("BaKb", "36.36-37.45", "0.0134(5)"),
        ("BaKx", "31.82-37.45", "0.0700(20)"),
    ],
    "133Ba": [
        ("CsKa", "30.63-30.97", "0.980(14)"),
        ("CsKb", "34.97-36.01", "0.230(5)"),
        ("CsKx", "30.63-36.01", "1.210(16)"),
    ],
    "139Ce": [
        ("LaKa", "33.03-33.44", "0.643(18)"),
        ("LaKb", "37.78-38.93", "0.154(5)"),
        ("LaKx", "33.03-38.93", "0.797(22)"),
    ],
    "152Eu": [
        ("SmKa", "39.52-40.12", "0.591(12)"),
        ("GdKa", "42.31-43.00", "0.00648(22)"),
        ("SmKb", "45.38-46.82", "0.149(3)"),
        ("GdKb", "48.65-50.21", "0.00176(18)"),
        ("SmKx", "39.52-46.82", "0.740(12)"),
        ("GdKx", "42.31-50.21", "0.00824(28)"),
        ("(Sm+Gd)Kx", "39.52-50.21", "0.748(12)"),
    ],
    "154Eu": [
        ("GdKa", "42.31-43.00", "0.205(6)"),
        ("GdKb", "48.65-50.21", "0.051(2)"),
        ("GdKx", "42.31-50.21", "0.256(6)"),
    ],
    "198Au": [
        ("HgKa", "68.89-70.82", "0.0219(8)"),
        ("HgKb", "80.12-82.78", "0.0061(3)"),
        ("HgKx", "68.89-82.78", "0.0280(10)"),
    ],
    "203Hg": [
        ("TlLx", "8.95-14.40", "0.060(12)"),
        ("TlKa2", "70.83", "0.038(2)"),
        ("TlKa1", "72.87", "0.064(2)"),
        ("TlKb'1", "82.43", "0.022(1)"),
        ("TlKb'2", "85.19", "0.0063(3)"),
        ("TlKx", "70.83-85.19", "0.130(4)"),
    ],
    "207Bi": [
        ("PbLx", "9.19-14.91", "0.325(13)"),
        ("PbKa2", "72.80", "0.226(12)"),
        ("PbKa1", "74.97", "0.382(20)"),
        ("PbKb'1", "84.79", "0.130(10)"),
        ("PbKb'2", "87.63", "0.039(3)"),
        ("PbKx", "72.80-87.63", "0.777(26)"),
    ],
    "241Am": [
        ("NpLl", "11.871", "0.0085(3)"),
        ("NpLa", "13.927", "0.132(4)"),
        ("NpLbeta_eta", "17.611", "0.194(6)"),
        ("NpLgamma", "20.997", "0.049(2)"),
    ],
}

# ---------------------------------------------------------------------------
# Data transcribed from Table 3 (gamma-ray standards): energies and emission
# probabilities, with literature reference tag. Values and their
# parenthetical uncertainties are stored as separate fields (the uncertainty
# is in the last digit(s) of the value, e.g. value "1274.542" with error
# digits "7" means 1274.542 +/- 0.007).
# Each entry: (Energy keV [str], Energy error digits [str],
#              Probability [str], Probability error digits [str],
#              Reference [str], is_daughter [bool])
# ---------------------------------------------------------------------------
GAMMA_DATA = {
    "22Na": [
        ("1274.542", "7", "0.99935", "15", "[4]", False),
    ],
    "24Na": [
        ("1368.633", "6", "0.999936", "15", "[4]", False),
        ("2754.030", "14", "0.99855", "5", "[4]", False),
    ],
    "46Sc": [
        ("889.277", "3", "0.999844", "16", "[5]", False),
        ("1120.545", "4", "0.999874", "11", "[5]", False),
    ],
    "51Cr": [
        ("320.0842", "9", "0.0986", "5", "[6]", False),
    ],
    "54Mn": [
        ("834.843", "6", "0.999758", "24", "[5]", False),
    ],
    "56Co": [
        ("846.764", "6", "0.99933", "7", "[5]", False),
        ("1037.844", "4", "0.1413", "5", "[5]", False),
        ("1175.099", "8", "0.02239", "11", "[5]", False),
        ("1238.287", "6", "0.6607", "19", "[5]", False),
        ("1360.206", "6", "0.04256", "15", "[5]", False),
        ("1771.350", "15", "0.1549", "5", "[5]", False),
        ("2015.179", "11", "0.03029", "13", "[5]", False),
        ("2034.759", "11", "0.07771", "27", "[5]", False),
        ("2598.460", "10", "0.1696", "6", "[5]", False),
        ("3201.954", "14", "0.0313", "9", "[5]", False),
        ("3253.417", "14", "0.0762", "24", "[5]", False),
        ("3272.998", "14", "0.0178", "6", "[5]", False),
        ("3451.154", "13", "0.0093", "4", "[5]", False),
        ("3548.27", "10", "0.00178", "9", "[5]", False),
    ],
    "57Co": [
        ("14.4127", "4", "0.0916", "15", "[7]", False),
        ("122.0614", "3", "0.8560", "17", "[7]", False),
        ("136.4743", "5", "0.1068", "8", "[7]", False),
    ],
    "58Co": [
        ("810.775", "9", "0.9945", "1", "[7]", False),
    ],
    "60Co": [
        ("1173.238", "4", "0.99857", "22", "[4]", False),
        ("1332.502", "5", "0.99983", "6", "[4]", False),
    ],
    "65Zn": [
        ("1115.546", "4", "0.5060", "24", "[6]", False),
    ],
    "75Se": [
        ("96.7344", "10", "0.0341", "4", "[6]", False),
        ("121.1171", "14", "0.171", "1", "[6]", False),
        ("136.0008", "6", "0.588", "3", "[6]", False),
        ("264.6580", "17", "0.590", "2", "[6]", False),
        ("279.5431", "22", "0.250", "1", "[6]", False),
        ("400.6593", "13", "0.115", "1", "[6]", False),
    ],
    "85Sr": [
        ("514.0076", "22", "0.984", "4", "[5]", False),
    ],
    "88Y": [
        ("898.042", "4", "0.940", "3", "[8]", False),
        ("1836.063", "13", "0.9936", "3", "[8]", False),
    ],
    "94Nb": [
        ("702.645", "6", "0.9979", "5", "[9]", False),
        ("871.119", "4", "0.9986", "5", "[9]", False),
    ],
    "95Nb": [
        ("765.807", "6", "0.9981", "3", "[9]", False),
    ],
    "109Cd": [
        ("88.0341", "11", "0.0363", "2", "[8]", False),
    ],
    "111In": [
        ("171.28", "3", "0.9078", "10", "[5]", False),
        ("245.35", "4", "0.9416", "6", "[5]", False),
    ],
    "113Sn": [
        ("391.702", "4", "0.6489", "13", "[9]", False),
    ],
    "125Sb": [
        ("176.313", "1", "0.0685", "7", "[8]", False),
        ("380.452", "8", "0.01518", "16", "[8]", False),
        ("427.875", "6", "0.297", "3", "[8]", False),
        ("463.365", "5", "0.1048", "11", "[8]", False),
        ("600.600", "4", "0.1773", "18", "[8]", False),
        ("606.718", "3", "0.0500", "5", "[8]", False),
        ("635.954", "5", "0.1121", "12", "[8]", False),
    ],
    "125I": [
        ("35.4919", "5", "0.0658", "8", "[8]", False),
    ],
    "134Cs": [
        ("475.364", "3", "0.0149", "2", "[5]", False),
        ("563.240", "4", "0.0836", "3", "[5]", False),
        ("569.328", "3", "0.1539", "6", "[5]", False),
        ("604.720", "3", "0.9763", "6", "[5]", False),
        ("795.859", "5", "0.854", "3", "[5]", False),
        ("801.948", "5", "0.0869", "3", "[5]", False),
        ("1038.610", "7", "0.00990", "5", "[5]", False),
        ("1167.968", "5", "0.01792", "7", "[5]", False),
        ("1365.185", "7", "0.03016", "11", "[5]", False),
    ],
    "137Cs": [
        ("661.660", "3", "0.851", "2", "[8]", False),
    ],
    "133Ba": [
        ("80.998", "5", "0.3411", "28", "[7]", False),
        ("276.398", "1", "0.07147", "30", "[7]", False),
        ("302.853", "1", "0.1830", "6", "[7]", False),
        ("356.017", "2", "0.6194", "14", "[7]", False),
        ("383.851", "3", "0.08905", "29", "[7]", False),
    ],
    "139Ce": [
        ("165.857", "6", "0.7987", "6", "[8]", False),
    ],
    "152Eu": [
        ("121.7824", "4", "0.2837", "13", "[9]", False),
        ("244.6989", "10", "0.0753", "4", "[9]", False),
        ("344.2811", "19", "0.2657", "11", "[9]", False),
        ("411.126", "3", "0.02238", "10", "[9]", False),
        ("443.965", "4", "0.03125", "14", "[9]", False),
        ("778.903", "6", "0.1297", "6", "[9]", False),
        ("867.390", "6", "0.04214", "25", "[9]", False),
        ("964.055", "4", "0.1463", "6", "[9]", False),
        ("1085.842", "4", "0.1013", "5", "[9]", False),
        ("1089.767", "14", "0.01731", "9", "[9]", False),
        ("1112.087", "6", "0.1354", "6", "[9]", False),
        ("1212.970", "13", "0.01412", "8", "[9]", False),
        ("1299.152", "9", "0.01626", "11", "[9]", False),
        ("1408.022", "4", "0.2085", "9", "[9]", False),
    ],
    "154Eu": [
        ("123.071", "1", "0.412", "5", "[5]", False),
        ("247.930", "1", "0.0695", "9", "[5]", False),
        ("591.762", "5", "0.0499", "6", "[5]", False),
        ("692.425", "4", "0.0180", "3", "[5]", False),
        ("723.305", "5", "0.202", "2", "[5]", False),
        ("756.804", "5", "0.0458", "6", "[5]", False),
        ("873.190", "5", "0.1224", "15", "[5]", False),
        ("996.262", "6", "0.1048", "13", "[5]", False),
        ("1004.725", "7", "0.182", "2", "[5]", False),
        ("1274.436", "6", "0.350", "4", "[5]", False),
        ("1494.048", "9", "0.0071", "2", "[5]", False),
        ("1596.495", "18", "0.0181", "2", "[5]", False),
    ],
    "198Au": [
        ("411.8044", "11", "0.9557", "47", "[6]", False),
    ],
    "203Hg": [
        ("279.1967", "12", "0.8148", "8", "[9]", False),
    ],
    "207Bi": [
        ("569.702", "2", "0.9774", "3", "[5]", False),
        ("1063.662", "4", "0.745", "2", "[5]", False),
        ("1770.237", "9", "0.0687", "4", "[5]", False),
    ],
    "228Th": [
        ("84.373", "3", "0.0122", "2", "[8]", False),
        ("238.632", "2", "0.435", "4", "[8]", True),
        ("240.987", "6", "0.0410", "5", "[8]", True),
        ("277.358", "10", "0.0230", "3", "[8]", True),
        ("300.094", "10", "0.0325", "3", "[8]", True),
        ("510.77", "10", "0.0818", "10", "[8]", True),  # note: close to 511.003 keV annihilation radiation
        ("583.191", "2", "0.306", "2", "[8]", True),
        ("727.330", "9", "0.0669", "9", "[8]", True),
        ("860.564", "5", "0.0450", "4", "[8]", True),
        ("1620.735", "10", "0.0149", "5", "[8]", True),
        ("2614.533", "13", "0.3586", "6", "[8]", True),
    ],
    "239Np": [
        ("106.123", "2", "0.267", "4", "[10]", False),
        ("228.183", "1", "0.1112", "15", "[10]", False),
        ("277.599", "2", "0.1431", "20", "[10]", False),
    ],
    "241Am": [
        ("26.345", "1", "0.024", "1", "[3]", False),
        ("59.537", "1", "0.360", "4", "[3]", False),
    ],
    "243Am": [
        ("43.53", "1", "0.0594", "11", "[6]", False),
        ("74.66", "1", "0.674", "10", "[6]", False),
    ],
}


def combine_val_err(value_s, err_digits_s):
    """
    Combine a value string and its IAEA-style trailing-digit uncertainty
    into (value, error), e.g. ('1274.542', '7') -> (1274.542, 0.007).

    The uncertainty digits apply to the last decimal place(s) of the
    value. If err_digits_s is empty/None, or value_s can't be parsed as
    a plain float (e.g. an energy range like '4.95-5.43'), returns
    (value_s, None).
    """
    value_s = value_s.strip()
    if "." in value_s:
        decimals = len(value_s.split(".")[1])
    else:
        decimals = 0
    try:
        value = float(value_s)
    except ValueError:
        return value_s, None
    if not err_digits_s:
        return value, None
    err = round(int(err_digits_s) * (10 ** (-decimals)), decimals)
    return value, err


def format_error_fixed(err_digits_s, source_decimals, target_decimals):
    """
    Format a trailing-digit uncertainty (e.g. '22' from '42.310(22)') as a
    plain decimal string with target_decimals decimal places, given it was
    originally quoted against a value with source_decimals decimal places.

    Avoids scientific notation entirely (unlike round()+str()), and lets
    the error line up under a value column that has been padded to a
    common number of decimals for display alignment.
    """
    if not err_digits_s:
        return ""
    digits = err_digits_s.strip()
    if source_decimals > 0:
        frac = digits.rjust(source_decimals, "0")
    else:
        frac = ""
    if target_decimals > source_decimals:
        frac = frac + "0" * (target_decimals - source_decimals)
    if target_decimals == 0:
        return digits
    return f"0.{frac}"


def decimals_of(value_s):
    """Number of digits after the decimal point in a numeric string, or
    None if value_s isn't a plain number (e.g. an energy range)."""
    value_s = value_s.strip()
    try:
        float(value_s)
    except ValueError:
        return None
    return len(value_s.split(".")[1]) if "." in value_s else 0


def pad_columns(header, rows, sep="\t\t"):
    """
    Left-justify every column to the widest entry (header included) so
    columns line up regardless of differing text/number lengths, then
    join columns with `sep`.

    Returns (header_line, [row_line, ...]).
    """
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


def normalize_nuclide(name):
    """Normalize user input like ' 152eu ' / 'Eu152' / '152Eu' -> '152Eu'."""
    name = name.strip()
    # Try to find all keys case-insensitively first (handles metastable 'm' too)
    for key in list(XRAY_DATA.keys()) + list(GAMMA_DATA.keys()):
        if key.lower() == name.lower():
            return key
    return name  # fall back, will simply not be found


def print_report(nuclide, file=sys.stdout):
    key = normalize_nuclide(nuclide)
    xray_rows = XRAY_DATA.get(key)
    gamma_rows = GAMMA_DATA.get(key)

    if xray_rows is None and gamma_rows is None:
        print(f"No data found for nuclide '{nuclide}'.", file=file)
        print("Use --list to see all available nuclides.", file=file)
        return 1

    # --- X-ray table -------------------------------------------------
    if xray_rows:
        print(f"=== X-ray standards: {key} ===", file=file)
        header = ["Trans", "Energy(keV)", "Probability"]
        rows = [[trans, energy, prob] for trans, energy, prob in xray_rows]
        header_line, row_lines = pad_columns(header, rows)
        print(header_line, file=file)
        for line in row_lines:
            print(line, file=file)
        print("-" * 60, file=file)

    # --- Gamma-ray table ----------------------------------------------
    print(f"=== Gamma-ray standards: {key} ===", file=file)
    if gamma_rows:
        # Normalise each numeric column to the widest decimal count seen
        # in this report so the decimal points line up, then left-pad
        # every column to a common width.
        e_decimals = max(
            (decimals_of(e) or 0) for e, _, _, _, _, _ in gamma_rows
        )
        p_decimals = max(
            (decimals_of(p) or 0) for _, _, p, _, _, _ in gamma_rows
        )

        header = ["Energy(keV)", "Energy_error(keV)", "Probability", "Probability_error"]
        rows = []
        for energy_s, energy_err_s, prob_s, prob_err_s, ref, is_daughter in gamma_rows:
            src_e_dec = decimals_of(energy_s) or 0
            src_p_dec = decimals_of(prob_s) or 0
            e_str = f"{float(energy_s):.{e_decimals}f}"
            p_str = f"{float(prob_s):.{p_decimals}f}"
            e_err_str = format_error_fixed(energy_err_s, src_e_dec, e_decimals)
            p_err_str = format_error_fixed(prob_err_s, src_p_dec, p_decimals)
            if is_daughter:
                e_str += "*"
            rows.append([e_str, e_err_str, p_str, p_err_str])
        header_line, row_lines = pad_columns(header, rows)
        print(header_line, file=file)
        for line in row_lines:
            print(line, file=file)
    else:
        print("(no gamma-ray data in table for this nuclide)", file=file)

    return 0


# Atomic number for every element symbol appearing in the nuclide keys
# above, used to order --list output by Z then mass number.
ELEMENT_Z = {
    "Na": 11, "Sc": 21, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Zn": 30,
    "Se": 34, "Sr": 38, "Y": 39, "Nb": 41, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "I": 53, "Cs": 55, "Ba": 56, "Ce": 58, "Eu": 63, "Au": 79,
    "Hg": 80, "Bi": 83, "Th": 90, "Np": 93, "Am": 95,
}

_NUCLIDE_RE = re.compile(r"^(\d+)(m?)([A-Za-z]+)$")


def nuclide_sort_key(key):
    """Sort key giving (Z, mass number, metastable flag) for a nuclide
    string like '152Eu' or '93mNb', so --list can be ordered by Z then A."""
    m = _NUCLIDE_RE.match(key)
    if not m:
        return (999, 0, 0, key)  # unrecognised format, push to the end
    mass, meta, symbol = m.groups()
    z = ELEMENT_Z.get(symbol, 999)
    return (z, int(mass), 1 if meta else 0, key)


def list_nuclides():
    all_keys = sorted(
        set(XRAY_DATA.keys()) | set(GAMMA_DATA.keys()), key=nuclide_sort_key
    )
    print("Available nuclides (ordered by Z, then mass number):")
    for k in all_keys:
        tags = []
        if k in XRAY_DATA:
            tags.append("X-ray")
        if k in GAMMA_DATA:
            tags.append("gamma")
        print(f"  {k}\t({', '.join(tags)})")


def main():
    parser = argparse.ArgumentParser(
        description="Look up IAEA-TECDOC-619 X-ray/gamma-ray calibration "
        "standard energies and emission probabilities for a nuclide."
    )
    parser.add_argument(
        "nuclide",
        nargs="?",
        help="Nuclide, e.g. 152Eu, 60Co, 241Am, 93mNb",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available nuclides and exit"
    )
    args = parser.parse_args()

    if args.list or not args.nuclide:
        list_nuclides()
        if not args.nuclide:
            return 0
        return 0

    return print_report(args.nuclide)


if __name__ == "__main__":
    sys.exit(main())
