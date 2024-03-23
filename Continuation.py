import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg
import scipy.optimize as opt
from autograd import jacobian

import NewtonRaphson as nr

def continuation(G, dGdu, dGdp, u0, p0, ds_min, ds_max, ds, N, a_tol=1.e-8, max_it=10, verbose=False):
	M = u0.size
	u = np.copy(u0) # Always the previous point on the curve
	p = np.copy(p0)	# Always the previous point on the curve
	u_path = [u]
	p_path = [p]

	print(u, p)
	print_str = 'Step n :{0:3d}\t u :{1:4f}\t p :{2:4f}'.format(0, lg.norm(u), p)
	print(print_str)

	# Choose random initial for now.
	prev_tau = np.zeros(M+1)
	prev_tau[M] = -1.0
	print('intial norm',  lg.norm(prev_tau))
	for n in range(1, N+1):
		# Determine the tangent to the curve at current point
		# By solving an underdetermined system with quadratic constraint norm(tau)**2 = 1
		Gu = dGdu(u, p)
		Gp = dGdp(u, p)
		tau = _computeTangent(Gu, Gp, prev_tau, M, a_tol)

		# Our implementation uses adaptive timetepping
		while ds > ds_min:
			# Predictor step;
			# compute initial value for Newton-Raphson method
			u_p = u + tau[0:M] * ds
			p_p = p + tau[M]   * ds
			x_p = np.append(u_p, p_p)

			# Corrector step: 
			# create the system and solve it with the newton-raphson method.
			N = lambda x: np.dot(tau, x - np.append(u, p)) + ds
			F = lambda x: np.append(G(x[0:M], x[M]), N(x))
			dF = jacobian(F)
			result = nr.Newton(F, dF, x_p, a_tol=a_tol, max_it=max_it)

			# Adaptive timestepping
			if result.success:
				u = np.copy(result.x[0:M])
				p = result.x[M]
				u_path.append(u)
				p_path.append(p)

				# Updating the arclength step and tangent vector
				ds = min(1.2*ds, ds_max)
				prev_tau = np.copy(tau)
				break
			elif result.singular:
				# Find the bifurcation point by solving det(dF) = 0
				det_df = lambda x: lg.det(dF(x))
				x_singular = opt.fsolve(det_df, x_p)
				print('Bifurcation Point at', x_singular, '. Aborting')

				return np.array(u_path), np.array(p_path)

			# Decrease arclength if Newton routine needs more than max_it steps
			ds = max(0.5*ds, ds_min)
		
		print_str = 'Step n :{0:3d}\t u :{1:4f}\t p :{2:4f}'.format(n, lg.norm(u), p)
		print(print_str)

	return np.array(u_path), np.array(p_path)

def _computeTangent(Gu, Gp, prev_tau, M, a_tol):
	# Setup extended jacobian
	DG = np.zeros((M,M+1)) 
	DG[0:M, 0:M] = Gu
	DG[:,M] = Gp

	# Do a version of quadratic programming (can we implement QP?)
	g_tangent = lambda tau: np.append(np.dot(DG, tau), np.dot(tau, tau) - 1.0)
	dg_tangent = jacobian(g_tangent)
	tau = nr.Newton(g_tangent, dg_tangent, prev_tau, a_tol=a_tol).x

	return tau