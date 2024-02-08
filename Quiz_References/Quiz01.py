from numpy.linalg import svd
import numpy as np


#LSA/LSI--Latent Semantic Analysis/Indexing
# Define a matrix
# Don't forget to verify that you have the correct size matrix
A = np.array([[2,1,0,0,0], [1,1,1,0,0], [0,0,1,1,1], [0,0,0,1,2]])

print("Matrix A:\n", A)

# Calculate the singular value decomposition
U, S, VT = np.linalg.svd(A)
# U.round(2), S.round(2), VT.round(2)

print("U:\n", U.round(2) )
print("S:\n", S.round(2))
print("VT:\n", VT.round(2))

# Define the rank for the reduced approximation
k = 2

# Keep only the first k singular values/vectors
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

print("U_k:\n", U_k.round(2))
print("S_k:\n", S_k.round(2))
print("VT_k:\n", VT_k.round(2))

# Calculate the reduced-rank approximation of A
A_k = U_k @ S_k @ VT_k
print("\n ----------------------------" )
print("Reduced-rank approximation of A using a K value of " + str(k) + ":\n", A_k.round(2))
print("\nAccuracy goes down as the depending on the size of K" )

