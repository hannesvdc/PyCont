import autograd.numpy as np
import autograd.numpy.linalg as lg

def is_stable(Gu, u_path, p_path):
    N = len(u_path)
    index = N // 2
    print(u_path.shape, p_path.shape)

    M = Gu(u_path[index], p_path[index])
    return np.all(np.real(lg.eigvals(M)) <= 0.0)