import numpy as np

def bialternate(A, B):
    if (A.shape[0] != A.shape[1]) or (B.shape[0] != B.shape[1]):
        raise ArithmeticError('Matrices A and B must be square.')
    if A.shape != B.shape:
        raise ArithmeticError('Matrices A and B do not have the same shape for bialternate product.')
    n = A.shape[0]
    m = n * (n-1) // 2
    
    C = np.zeros((m,m))
    row_index = 0
    column_index = 0
    for p in range(1, n):
        for q in range(p-1):
            for r in range(1, n):
                for s in range(r-1):
                    det_1 = A[p,r] * B[q,s] - A[p,s] * B[q,r]
                    det_2 = B[p,r] * A[q,s] - B[p,s] * A[q,r]
                    C[row_index,column_index] = 0.5*(det_1 + det_2)

                    # Update column index
                    column_index += 1

            # Update row index
            row_index += 1

    return C