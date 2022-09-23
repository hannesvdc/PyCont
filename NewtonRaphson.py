import numpy as np
import numpy.linalg as lg

class Result:
	def __init__(self, u, tol, m_it, err, it, suc):
		self.x = u
		self.a_tol = tol
		self.max_it = m_it
		self.error = err
		self.iterations = it
		self.success = suc

def NewtonRaphson(f, df, u0, a_tol=1.e-8, max_it=10, dt=1.0):
	u = u0
	fu = f(u)

	i = 0
	while lg.norm(fu) > a_tol and i < max_it:
		dfu = df(u)

		du = lg.solve(dfu, -fu)
		u = u + dt*dt

		fu = f(u)
		i += 1

	res = Result(u, a_tol, max_it, lg.norm(fu), i, i < max_it)
	return res.__dict__
