from pysotopelib import ZtoCS

def test_ZtoCS():
    assert ZtoCS.ZtoChemSym(0) == "shit"
    assert ZtoCS.ZtoChemSym(1) == "H"
    assert ZtoCS.ZtoChemSym(2) == "He"
    assert ZtoCS.ZtoChemSym(3) == "Li"
    assert ZtoCS.ZtoChemSym(4) == "Be"
    assert ZtoCS.ZtoChemSym(5) == "B"
    assert ZtoCS.ZtoChemSym(6) == "C"
    assert ZtoCS.ZtoChemSym(7) == "N"
    assert ZtoCS.ZtoChemSym(8) == "O"
    assert ZtoCS.ZtoChemSym(9) == "F"
    assert ZtoCS.ZtoChemSym(10) == "Ne"
    assert ZtoCS.ZtoChemSym(11) == "Na"
    assert ZtoCS.ZtoChemSym(12) == "Mg"
    assert ZtoCS.ZtoChemSym(13) == "Al"
    assert ZtoCS.ZtoChemSym(14) == "Si"
    assert ZtoCS.ZtoChemSym(15) == "P"
    assert ZtoCS.ZtoChemSym(16) == "S"
    assert ZtoCS.ZtoChemSym(17) == "Cl"
    assert ZtoCS.ZtoChemSym(18) == "Ar"
    assert ZtoCS.ZtoChemSym(19) == "K"
    assert ZtoCS.ZtoChemSym(20) == "Ca"
    assert ZtoCS.ZtoChemSym(21) == "Sc"
    assert ZtoCS.ZtoChemSym(22) == "Ti"
    assert ZtoCS.ZtoChemSym(23) == "V"
    assert ZtoCS.ZtoChemSym(24) == "Cr"
    assert ZtoCS.ZtoChemSym(25) == "Mn"
    assert ZtoCS.ZtoChemSym(26) == "Fe"
    assert ZtoCS.ZtoChemSym(27) == "Co"
    assert ZtoCS.ZtoChemSym(28) == "Ni"
    assert ZtoCS.ZtoChemSym(29) == "Cu"
    assert ZtoCS.ZtoChemSym(30) == "Zn"
    assert ZtoCS.ZtoChemSym(31) == "Ga"
    assert ZtoCS.ZtoChemSym(32) == "Ge"
    assert ZtoCS.ZtoChemSym(33) == "As"
    assert ZtoCS.ZtoChemSym(34) == "Se"
    assert ZtoCS.ZtoChemSym(35) == "Br"
    assert ZtoCS.ZtoChemSym(36) == "Kr"
    assert ZtoCS.ZtoChemSym(37) == "Rb"
    assert ZtoCS.ZtoChemSym(38) == "Sr"
    assert ZtoCS.ZtoChemSym(39) == "Y"
    assert ZtoCS.ZtoChemSym(40) == "Zr"
    assert ZtoCS.ZtoChemSym(41) == "Nb"
    assert ZtoCS.ZtoChemSym(42) == "Mo"
    assert ZtoCS.ZtoChemSym(43) == "Tc"
    assert ZtoCS.ZtoChemSym(44) == "Ru"
    assert ZtoCS.ZtoChemSym(45) == "Rh"
    assert ZtoCS.ZtoChemSym(46) == "Pd"
    assert ZtoCS.ZtoChemSym(47) == "Ag"
    assert ZtoCS.ZtoChemSym(48) == "Cd"
    assert ZtoCS.ZtoChemSym(49) == "In"
    assert ZtoCS.ZtoChemSym(50) == "Sn"
    assert ZtoCS.ZtoChemSym(51) == "Sb"
    assert ZtoCS.ZtoChemSym(52) == "Te"
    assert ZtoCS.ZtoChemSym(53) == "I"
    assert ZtoCS.ZtoChemSym(54) == "Xe"
    assert ZtoCS.ZtoChemSym(55) == "Cs"
    assert ZtoCS.ZtoChemSym(56) == "Ba"
    assert ZtoCS.ZtoChemSym(57) == "La"
    assert ZtoCS.ZtoChemSym(58) == "Ce"
    assert ZtoCS.ZtoChemSym(59) == "Pr"
    assert ZtoCS.ZtoChemSym(60) == "Nd"
    assert ZtoCS.ZtoChemSym(61) == "Pm"
    assert ZtoCS.ZtoChemSym(62) == "Sm"
    assert ZtoCS.ZtoChemSym(63) == "Eu"
    assert ZtoCS.ZtoChemSym(64) == "Gd"
    assert ZtoCS.ZtoChemSym(65) == "Tb"
    assert ZtoCS.ZtoChemSym(66) == "Dy"
    assert ZtoCS.ZtoChemSym(67) == "Ho"
    assert ZtoCS.ZtoChemSym(68) == "Er"
    assert ZtoCS.ZtoChemSym(69) == "Tm"
    assert ZtoCS.ZtoChemSym(70) == "Yb"
    assert ZtoCS.ZtoChemSym(71) == "Lu"
    assert ZtoCS.ZtoChemSym(72) == "Hf"
    assert ZtoCS.ZtoChemSym(73) == "Ta"
    assert ZtoCS.ZtoChemSym(74) == "W"
    assert ZtoCS.ZtoChemSym(75) == "Re"
    assert ZtoCS.ZtoChemSym(76) == "Os"
    assert ZtoCS.ZtoChemSym(77) == "Ir"
    assert ZtoCS.ZtoChemSym(78) == "Pt"
    assert ZtoCS.ZtoChemSym(79) == "Au"
    assert ZtoCS.ZtoChemSym(80) == "Hg"
    assert ZtoCS.ZtoChemSym(81) == "Tl"
    assert ZtoCS.ZtoChemSym(82) == "Pb"
    assert ZtoCS.ZtoChemSym(83) == "Bi"
    assert ZtoCS.ZtoChemSym(84) == "Po"
    assert ZtoCS.ZtoChemSym(85) == "At"
    assert ZtoCS.ZtoChemSym(86) == "Rn"
    assert ZtoCS.ZtoChemSym(87) == "Fr"
    assert ZtoCS.ZtoChemSym(88) == "Ra"
    assert ZtoCS.ZtoChemSym(89) == "Ac"
    assert ZtoCS.ZtoChemSym(90) == "Th"
    assert ZtoCS.ZtoChemSym(91) == "Pa"
    assert ZtoCS.ZtoChemSym(92) == "U"
    assert ZtoCS.ZtoChemSym(93) == "Np"
    assert ZtoCS.ZtoChemSym(94) == "Pu"
    assert ZtoCS.ZtoChemSym(95) == "Am"
    assert ZtoCS.ZtoChemSym(96) == "Cm"
    assert ZtoCS.ZtoChemSym(97) == "Bk"
    assert ZtoCS.ZtoChemSym(98) == "Cf"
    assert ZtoCS.ZtoChemSym(99) == "Es"
    assert ZtoCS.ZtoChemSym(100) == "Fm"
    assert ZtoCS.ZtoChemSym(101) == "Md"
    assert ZtoCS.ZtoChemSym(102) == "No"
    assert ZtoCS.ZtoChemSym(103) == "Lr"
    assert ZtoCS.ZtoChemSym(104) == "Rf"
    assert ZtoCS.ZtoChemSym(105) == "Db"
    assert ZtoCS.ZtoChemSym(106) == "Sg"
    assert ZtoCS.ZtoChemSym(107) == "Bh"
    assert ZtoCS.ZtoChemSym(108) == "Hs"
    assert ZtoCS.ZtoChemSym(109) == "Mt"
    assert ZtoCS.ZtoChemSym(110) == "Ds"
    assert ZtoCS.ZtoChemSym(111) == "Rg"
    assert ZtoCS.ZtoChemSym(112) == "Cn"
    assert ZtoCS.ZtoChemSym(113) == "Nh"
    assert ZtoCS.ZtoChemSym(114) == "Fl"
    assert ZtoCS.ZtoChemSym(115) == "Mc"
    assert ZtoCS.ZtoChemSym(116) == "Lv"
    assert ZtoCS.ZtoChemSym(117) == "Ts"
    assert ZtoCS.ZtoChemSym(118) == "Og"
    assert ZtoCS.ZtoChemSym(119) == "119"
