import sys
sys.path.append("../../")

import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt
from autograd import jacobian

import Continuation as cont
import NewtonRaphson as nr

def gradient_extremals_MB():
	# Muller-Brown Potential with Gradient and Hessian
	V = lambda x: -200.0 * np.exp(-(x[0]-1.0)**2 - 10.0*x[1]**2)\
				  -100.0 * np.exp(-x[0]**2 - 10.0*(x[1]-0.5)**2)\
				  -170.0 * np.exp(-6.5*(x[0]+0.5)**2 + 11.0*(x[0]+0.5)*(x[1]-1.5) - 6.5*(x[1]-1.5)**2)\
				  + 15.0 * np.exp( 0.7*(x[0]+1.0)**2 +  0.6*(x[0]+1.0)*(x[1]-1.0) + 0.7*(x[1]-1.0)**2)
	dV  = jacobian(V)
	ddV = jacobian(dV)

	# Objective Function and Partial Derivatives
	G_full = lambda y: np.append([V(y[0:2]) - y[3]], np.dot(ddV(y[0:2]), dV(y[0:2])) - y[2]*dV(y[0:2]))
	dG_full = jacobian(G_full)
	G = lambda u, L: G_full(np.append(u, L))
	dGdu = lambda u, L: dG_full(np.append(u, L))[0:3, 0:3]
	dGdL = lambda u, L: dG_full(np.append(u, L))[0:3, 3]

	# Initial Condition is Around - but enough far from - Local Minimum because 
    # Critical Points (and VRIs) are Bifurcation Points of G (i.e. G = 0 and dGdu = 0)
	rng = rd.RandomState()
	x0 = np.array([0.623499404930877, 0.0280377585286857]) + rng.normal(0.0, 0.1, 2)
	l0 = np.min(lg.eigvalsh(ddV(x0)))
	u0 = np.append(x0, l0)
	p0 = V(x0)
	u0 = nr.Newton(lambda u: G(u, p0), lambda u: dGdu(u, p0), u0, a_tol=1.e-14, max_it=100).x
	print('G Test', u0, G(u0, p0), dGdu(u0, p0), lg.det(dGdu(u0, p0)))

	# Continuation Parameters
	N = 10000
	ds_max = 0.001
	ds_min = 1.e-6
	ds = 0.1
	#continuation_result = cont.pseudoArclengthContinuation(G, dGdu, dGdL, u0, p0, ds_min, ds_max, ds, N, tolerance=1.e-10)
	continuation_result = cont.ContinuationResult()

	# Plot Branches and Bifurcation Points
	plotMullerBrown(continuation_result)

def plotMullerBrown(continuation_result):
	def MB2(X, Y):
		return -200.0 * np.exp(-(X-1.0)**2 - 10.0*Y**2)\
			   -100.0 * np.exp(-X**2 - 10.0*(Y-0.5)**2)\
			   -170.0 * np.exp(-6.5*(X+0.5)**2 + 11.0*(X+0.5)*(Y-1.5) - 6.5*(Y-1.5)**2)\
			   + 15.0 * np.exp( 0.7*(X+1.0)**2 +  0.6*(X+1.0)*(Y-1.0) + 0.7*(Y-1.0)**2)
	
	branches = continuation_result.branches
	bf_points = continuation_result.bifurcation_points

	_, ax=plt.figure(), plt.axes()

	x_lim = [-1.8, 1.2]
	y_lim = [-0.2, 2.0]

	xlist = np.linspace(x_lim[0], x_lim[1], 100)
	ylist = np.linspace(y_lim[0], y_lim[1], 100)
	X, Y = np.meshgrid(xlist, ylist)
	Z = MB2(X, Y)
	
	min_1 = np.array([-0.05001082, 0.4666941])
	min_2 = np.array([-0.55822363,  1.44172584])
	min_3 = np.array([0.6234994,  0.02803776])
	s_1 = np.array([0.21274942731, 0.29253349197])
	s_2 = np.array([-0.8314, 0.61276])
	cp = ax.contour(X, Y, Z, levels=np.linspace(-200.0, 200.0, 200))
	ax.scatter(min_1[0], min_1[1], marker='.', color='k', label='Local Minimum')
	ax.scatter(min_2[0], min_2[1], marker='.', color='k')
	ax.scatter(min_3[0], min_3[1], marker='.', color='k')
	ax.scatter(s_1[0], s_1[1], marker='x', color='k', label='Saddle Point')
	ax.scatter(s_2[0], s_2[1], marker='x', color='k')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_xlim(x_lim)
	ax.set_ylim(y_lim)

	plt.show()
	

if __name__ == '__main__':
	gradient_extremals_MB()