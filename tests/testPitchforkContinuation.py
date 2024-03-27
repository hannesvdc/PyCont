import sys
sys.path.append("../")

import autograd.numpy as np
import matplotlib.pyplot as plt

import src.PseudoArclengthContinuation as cont

def PitchforkTest():
	G = lambda x, r: r*x[0] - x[0]**3
	dGdx = lambda x, r: np.array([[r - 3.0*x[0]**2]])
	dGdr = lambda x, r: np.array([x[0]])

	ds_max = 0.001
	ds_min = 1.e-6
	ds = 0.1

	N = 10000
	u0 = np.array([0.0])
	p0 = -5.0
	u_path_1, r_path_1, bifurcation_points_1 = cont.continuation(G, dGdx, dGdr, u0, p0, ds_min, ds_max, ds, N, max_it=10, sign=1.0)
	u_path_1 = np.transpose(u_path_1)

	N = 20000
	u0 = np.array([-3.0])
	p0 = 9.0
	u_path_2, r_path_2, bifurcation_points_2 = cont.continuation(G, dGdx, dGdr, u0, p0, ds_min, ds_max, ds, N, max_it=10, sign=-1.0)
	u_path_2 = np.transpose(u_path_2)

	fig = plt.figure()
	ax = fig.gca()
	ax.plot(r_path_1, u_path_1[0], color='blue', label='Pitchfork Bifurcation')
	ax.plot(r_path_2, u_path_2[0], color='blue')
	for n in range(len(bifurcation_points_1)):
		ax.plot(bifurcation_points_1[n][0], bifurcation_points_1[n][1], 'ro')
	for n in range(len(bifurcation_points_2)):
		ax.plot(bifurcation_points_2[n][0], bifurcation_points_2[n][1], 'ro')
	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$u$')
	plt.xlim((min(np.min(r_path_1), np.min(r_path_2))-1.0, max(np.max(r_path_1), np.max(r_path_2))+1.0))
	plt.ylim((min(np.min(u_path_1), np.min(u_path_2))-1.0, max(np.max(u_path_1), np.max(u_path_2))+1.0))
	ax.legend()
	plt.show()	

if __name__ == '__main__':
	PitchforkTest()