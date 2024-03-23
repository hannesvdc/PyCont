import sys
sys.path.append("../")

import autograd.numpy as np
import matplotlib.pyplot as plt

import Continuation as cont

def PitchforkTest():
	G = lambda x, r: r*x[0] + x[0]**3
	dGdx = lambda x, r: np.array([[r + 3.0*x[0]**2]])
	dGdr = lambda x, r: np.array([x[0]])

	u0 = np.array([0.0])
	p0 = -5.0

	ds_max = 0.001
	ds_min = 1.e-6
	ds = 0.1
	N = 10000
	u_path, r_path, bifurcation_points = cont.continuation(G, dGdx, dGdr, u0, p0, ds_min, ds_max, ds, N, max_it=10, sign=1.0)
	u_path = np.transpose(u_path)

	fig = plt.figure()
	ax = fig.gca()
	ax.plot(r_path, u_path[0], color='blue', label='Pitchfork Bifurcation')
	for n in range(len(bifurcation_points)):
		ax.plot(bifurcation_points[n][0], bifurcation_points[n][1], 'ro')
	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$u$')
	plt.xlim((np.min(r_path)-1.0, np.max(r_path)+1.0))
	plt.ylim((np.min(u_path)-1.0, np.max(u_path)+1.0))
	ax.legend()
	plt.show()	

if __name__ == '__main__':
	PitchforkTest()