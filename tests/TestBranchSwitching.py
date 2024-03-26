import sys
sys.path.append("../")

import autograd.numpy as np
import matplotlib.pyplot as plt

import BranchSwitching as bs

def PitchforkTest():
	G = lambda x, r: r*x[0] - x[0]**3
	dGdx = lambda x, r: np.array([[r - 3.0*x[0]**2]])
	dGdr = lambda x, r: np.array([x[0]])
	F = lambda x: np.append(G(x[0:1], x[1]), 0.0)
	
	x_prev = np.array([-np.sqrt(0.02), 0.02]) # x, r
	x_singular = np.array([0.0, 0.0])
	directions = bs.branchSwitching(F, dGdx, dGdr, x_singular, x_prev)
	print('directions', directions)

	r1_path = np.linspace(-5.0, 5.0, 10001)
	r2_path = np.linspace( 0.0, 5.0, 10001)
	plt.plot(r1_path, 0.0*r1_path, color='gray')
	plt.plot(r2_path,  np.sqrt(r2_path), color='gray')
	plt.plot(r2_path, -np.sqrt(r2_path), color='gray')
	plt.plot(x_singular[1], x_singular[0], 'bo')
	for q in directions:
		plt.plot(q[1], q[0], 'ro')
	plt.xlabel(r'$r$')
	plt.ylabel(r'$x$')
	plt.show()

def TranscriticalTest():
	G = lambda x, r: r*x[0] - x[0]**2
	dGdx = lambda x, r: np.array([[r - 2.0*x[0]]])
	dGdr = lambda x, r: np.array([x[0]])
	F = lambda x: np.append(G(x[0:1], x[1]), 0.0)

	x_prev = np.array([-0.02, -0.02]) # x, r
	x_singular = np.array([0.0, 0.0])
	directions = bs.branchSwitching(F, dGdx, dGdr, x_singular, x_prev)
	print('directions', directions)

	r_path = np.linspace(-5.0, 5.0, 10001)
	plt.plot(r_path, 0.0*r_path, color='gray')
	plt.plot(r_path, r_path, color='gray')
	plt.plot(x_singular[1], x_singular[0], 'bo')
	for q in directions:
		plt.plot(q[1], q[0], 'ro')
	plt.xlabel(r'$r$')
	plt.ylabel(r'$x$')
	plt.show()

if __name__ == '__main__':
	PitchforkTest()
