# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

import dune.common
from dune.istl import blockVector, BlockVector, bcrsMatrix, BCRSMatrix, SeqJacobi, CGSolver, BuildMode

def identity(n, buildMode=BuildMode.random):
    if buildMode is BuildMode.random:
        mat = bcrsMatrix((n, n), n, BuildMode.random)
        for i in range(n):
            mat.setRowSize(i, 1)
        mat.endRowSizes()
        for i in range(n):
            mat.setIndices(i, [i])
        mat.endIndices()
        for i in range(n):
            mat[i, i] = 1
        return mat
    elif buildMode is BuildMode.implicit:
        mat = bcrsMatrix((n, n), n, 0, BuildMode.implicit)
        for i in range(n):
            mat[i, i] = 1
        mat.compress()
        return mat
    else:
        raise Exception("buildMode " + str(buildMode) + " not supported by identity")

def isIdentity(mat):
    if mat.rows != mat.cols:
        return False
    for i in range(mat.rows):
        if not (i, i) in mat:
            return False
        for j in range(mat.cols):
            if (i, j) in mat and mat[i, j] != (1 if i == j else 0):
                return False
    return True

mat = bcrsMatrix((5, 5), 5, BuildMode.random)
if mat.shape != (5, 5) or mat.nonZeroes != 5 or mat.buildMode != BuildMode.random:
    raise Exception("Matrix does not return constructor arguments correctly")

mat = bcrsMatrix((5, 5), 5, BuildMode.implicit, blockType=(2,3))
if mat.shape != (5, 5) or mat.nonZeroes != 5 or mat.buildMode != BuildMode.implicit \
   or mat[0,0].rows != 2 or mat[0,0].cols != 3:
    raise Exception("Matrix does not return constructor arguments correctly")

Matrix = BCRSMatrix((2,3))
mat = Matrix((5, 5), 5, Matrix.BuildMode.implicit)
if mat.shape != (5, 5) or mat.nonZeroes != 5 or mat.buildMode != BuildMode.implicit \
   or mat[0,0].rows != 2 or mat[0,0].cols != 3:
    raise Exception("Matrix does not return constructor arguments correctly")

# the following not working yet since the matrix market exported fails
# BMatrix = BCRSMatrix(Matrix)
# mat = BMatrix((5,5), 5, BuildMode.implicit)

# store identity matrix
mat = identity(5)
if not isIdentity(mat):
    raise Exception("Identity matrix not setup correctly")
mat.store("bcrsmatrix.mm", "matrixmarket")

for i, row in mat.enumerate:
    for j, col in row.enumerate:
        if i != j:
            raise Exception("Wrong sparsity pattern")
        if col != 1:
            raise Exception("Diagonal entry is not 1")

mat = identity(5, BuildMode.implicit)
if not isIdentity(mat):
    raise Exception("Identity matrix not setup correctly")

# manipulate diagonal to 2
for i in range(mat.rows):
    mat[i, i] *= 2
if isIdentity(mat):
    raise Exception("Matrix not manipulated")

# reload identity matrix
mat = bcrsMatrix()
mat.load("bcrsmatrix.mm", "matrixmarket")
if not isIdentity(mat):
    raise Exception("Matrix not loaded correctly")

# store in matlab format
mat.store("bcrsmatrix.txt", "matlab")

# solve trivial linear system
x = blockVector(5)
for i in range(5):
    x[i] = i+1 # float(i+1) # dune.common.FieldVector([i+1])

y1, y2 = blockVector(5), blockVector(5)
mat.mv(x, y1)
mat.asLinearOperator().apply(x, y2)
if (y1 - y2).two_norm > 1e-12:
    raise Exception("mat.mv != mat.asLinearOperator().apply")

S = CGSolver(mat.asLinearOperator(), SeqJacobi(mat), 1e-10)
mat = None
z = blockVector(5)
_, _, converged, _, _ = S(z, y1)
if not converged:
    raise Exception("CGSolver has not converged")
if (z - x).two_norm > 1e-8:
    raise Exception("CGSolver unable to solve identity")

s = "(" + ", ".join(str(v) for v in x) + ")"

str_x = "("
for i in range(0,5):
    str_x = str_x + "(" +  "{0:.6f}".format(x[i][0]) + "), "
str_x = str_x[:-2]
str_x = str_x +")"

if str_x != s:
    raise Exception(str(x) + " = str(x) != " + s)
