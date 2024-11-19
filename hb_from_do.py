# This program takes the input of dissolved oxygen concentration measured from an electrolysis probe and converts it to
# the matched hemoglobin saturation. This is used to validate image quantification in phantoms.

# Constants
R = 8.3145  # ideal gas constant; in J/molK


# Inputs from DO probe
o2_molarity = []
T = []

# Approximations
kH = 0  # Henry's law constant
p50 = 27  # 50% sat partial pressure for adults; in mmHg
h = 2.7  # Hill Constant

sO2 = (o2_molarity ** h) / ((p50 ** h) * (kH ** h) + o2_molarity ** h)