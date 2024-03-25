import sys
sys.path.append("../")

import autograd.numpy as np
import matplotlib.pyplot as plt

import BranchSwitching as bs

def PitchforkTest():
	G = lambda x, r: r*x[0] - x[0]**3
	dGdx = lambda x, r: np.array([[r - 3.0*x[0]**2]])
	dGdr = lambda x, r: np.array([x[0]])
	F = lambda x: np.append(G(x[0], x[1]), 0.0)
	
	x_prev = np.array([-0.02, 0.0])
	x_singular = np.array([0.0, 0.0])
	directions = bs.branchSwitching(F, dGdx, dGdr, x_singular, x_prev)
	print('directions', directions)

if __name__ == '__main__':
	PitchforkTest()
