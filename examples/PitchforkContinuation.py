import autograd.numpy as np
import matplotlib.pyplot as plt

from pycont import continuation

def PitchforkTest():
	G = lambda x, r: np.array([r*x[0] - x[0]**3])
	u0 = np.array([-2.0])
	p0 = 4.0

	ds_max = 0.001
	ds_min = 1.e-6
	ds = 0.001
	N = 10000
	continuation_result = continuation.pseudoArclengthContinuation(G, u0, p0, ds_min, ds_max, ds, N, tolerance=1.e-10)

	# Print some Internal info
	print('\nNumber of Branches:', len(continuation_result.branches))
	print('Bifurcation Points:', continuation_result.bifurcation_points)

	fig = plt.figure()
	ax = fig.gca()
	x_grid = np.linspace(-16, 11, 1001)
	y_grid = np.linspace(-3.5, 3.5, 1001)
	ax.plot(x_grid, 0.0*x_grid, color='lightgray')
	ax.plot(0.0*y_grid, y_grid, color='lightgray')
	for n in range(len(continuation_result.branches)):
		branch = continuation_result.branches[n]
		ax.plot(branch['p'], branch['u'], color='blue')
	ax.plot(p0, u0, 'go', label='SP')
	for n in range(len(continuation_result.bifurcation_points)):
		ax.plot(continuation_result.bifurcation_points[n][1], continuation_result.bifurcation_points[n][0], 'ro', label='BP')
	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$u$')
	ax.legend(loc='upper left')
	plt.show()	

if __name__ == '__main__':
	PitchforkTest()