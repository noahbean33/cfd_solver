# Script to generate difference formulas for user-specified derivatives, scheme-type and order of accuracy

import numpy as np
import math
import matplotlib.pyplot as plt
from difference_forms.core import fd_coefficients
from difference_forms.formatting import format_latex

def findMultiplier(sol):
    # This function will seek to convert fractional coefficients to integers
    # so output looks neat
    multiplier = 1
    count = 1
    while True:
        solNew = sol*multiplier
        solNew_ints = (np.rint(solNew)).astype(int)
        diff = abs(solNew_ints - solNew)
        if max(diff)<1e-4:
            break
        if count == 1000:
            multiplier = 1
            break
        multiplier = multiplier+1
        count = count+1
    return multiplier

def printDifferenceFormula(derivativeOrder,scheme,accuracyOrder):
    # New implementation using refactored core/formatting modules
    scheme_use = 'centered' if scheme == 'central' else scheme
    offsets, coeffs = fd_coefficients(derivativeOrder, scheme_use, accuracyOrder, h=1.0)
    latex = format_latex(offsets, coeffs, derivative_order=derivativeOrder, h_symbol='\\Delta x')
    size = offsets.size
    plt.text(0.5 - 0.5*(size/12), 0.5, f'$%s$' % latex, fontsize=24)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    printDifferenceFormula(1,'backward',2)
