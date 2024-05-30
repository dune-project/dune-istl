// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

/** \file \brief Tests the DILU preconditioner.
 */

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>



void seqDILUApplyIsCorrect1()
{
    /*
        Tests that applying the dilu preconditioner matches the expected result
        for a 2x2 matrix, with block size 2x2.
                 A               x     =     b
        | | 3  1| | 1  0| |   | |1| |     | |2| |
        | | 2  1| | 0  1| |   | |2| |     | |1| |
        |                 | x |     |  =  |     |
        | | 0  0| |-1  0| |   | |1| |     | |3| |
        | | 0  0| | 0 -1| |   | |1| |     | |4| |
    */


    const int N = 2;
    const int bz = 2;
    const int nonZeroes = 3;
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, bz, bz>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<double, bz>>;

    using ft = typename Vector::field_type;
    const ft myEps((ft)1e-6);
    Dune::TestSuite t;

    Matrix A(N, N, nonZeroes, Matrix::row_wise);
    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        row.insert(row.index());
        if (row.index() == 0) {
            row.insert(row.index() + 1);
        }
    }

    A[0][0][0][0]=3.0;
    A[0][0][0][1]=1.0;
    A[0][0][1][0]=2.0;
    A[0][0][1][1]=1.0;

    A[0][1][0][0]=1.0;
    A[0][1][1][1]=1.0;

    A[1][1][0][0]=-1.0;
    A[1][1][1][1]=-1.0;


    Vector x(2);
    x[0][0] = 1.0;
    x[0][1] = 2.0;
    x[1][0] = 1.0;
    x[1][1] = 1.0;

    Vector b(2);
    b[0][0] = 2.0;
    b[0][1] = 1.0;
    b[1][0] = 3.0;
    b[1][1] = 4.0;


    auto D_00 = A[0][0];
    auto D_00_inv = D_00;
    D_00_inv.invert();
    // D_11 = A_11 - L_10 D_0_inv U_01 = A_11
    auto D_11 = A[1][1];
    auto D_11_inv = D_11;
    D_11_inv.invert();

    // define: z = M^-1(b - Ax)
    // where: M = (D + L_A) A^-1 (D + U_A)
    // lower triangular solve: (D + L) y = b - Ax
    // y_0 = D_00_inv*[b_0 - (A_00*x_0 + A_01*x_1)]
    Vector y(2);

    auto rhs = b[0];
    A[0][0].mmv(x[0], rhs);
    A[0][1].mmv(x[1], rhs);
    D_00_inv.mv(rhs, y[0]);

    // y_1 = D_11_inv*(b_1 - (A_10*x_0 + A_11*x_1) - A_10*y_0) = D_11_inv*(b_1 - A_11*x_1)
    rhs = b[1];
    A[1][1].mmv(x[1], rhs);
    D_11_inv.mv(rhs, y[1]);


    // upper triangular solve: (D + U) z = Dy
    // z_1 = y_1
    Vector z(2);
    z[1] = y[1];

    // z_0 = y_0 - D_00_inv*A_01*z_1
    z[0] = y[0];
    auto temp = D_00_inv*A[0][1];
    temp.mmv(z[1], z[0]);

    // z is now the update x_k+1 - x_k

    Dune::SeqDILU<Matrix,Vector,Vector> seqDILU(A, 1);
    Vector v(2);
    Vector d(b);
    A.mmv(x, d);
    seqDILU.apply(v, d);

    // compare calculated update z from equations to update v from DILU preconditioner
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            t.check(std::abs(v[i][j] - z[i][j]) <= myEps)
             << " Error in SeqDILUApplyIsCorrect1, v[" << i <<"][" << j << "]=" << v[i][j] << " != z[" << i <<"][" << j << "]=" << z[i][j];
        }
    }
}


void seqDILUApplyIsCorrect2()
{
    /*
        Tests that applying the DILU preconditioner matches the expected result
        for a 3x3 matrix, with block size 3x3.

                          A                       x    =     b
        | | 3  1  2| | 0  0  0| | 0  0  0| |   | |1| |    | |2| |
        | | 2  3  1| | 0  0  0| | 0  0  0| |   | |2| |    | |1| |
        | | 2  1  0| | 0  0  0| | 0  0  0| |   | |3| |    | |2| |
        |                                  |   |     |    |     |
        | | 0  0  0| | 1  0  1| | 1  0  2| |   | |1| |    | |2| |
        | | 0  0  0| | 4  1  0| | 0  1  1| | x | |3| |  = | |3| |
        | | 0  0  0| | 3  1  3| | 0  1  3| |   | |2| |    | |2| |
        |                                  |   |     |    |     |
        | | 0  0  0| | 1  0  2| | 1  3  2| |   | |1| |    | |0| |
        | | 0  0  0| | 0  1  4| | 2  1  3| |   | |0| |    | |2| |
        | | 0  0  0| | 5  1  1| | 3  1  2| |   | |2| |    | |1| |
    */

    const int N = 3;
    const int bz = 3;
    const int nonZeroes = 5;
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, bz, bz>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<double, bz>>;

    using ft = typename Vector::field_type;
    const ft myEps((ft)1e-6);
    Dune::TestSuite t;

    Matrix A(N, N, nonZeroes, Matrix::row_wise);
    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        if (row.index() == 0) {
            row.insert(row.index());
        }
        else if (row.index() == 1) {
            row.insert(row.index());
            row.insert(row.index() + 1);
        }
        else if (row.index() == 2) {
            row.insert(row.index() - 1);
            row.insert(row.index());
        }
    }

    A[0][0][0][0]=3.0;    A[1][1][0][0]=1.0;    A[1][2][0][0]=1.0;
    A[0][0][0][1]=1.0;    A[1][1][0][1]=0.0;    A[1][2][0][1]=0.0;
    A[0][0][0][2]=2.0;    A[1][1][0][2]=1.0;    A[1][2][0][2]=2.0;
    A[0][0][1][0]=2.0;    A[1][1][1][0]=4.0;    A[1][2][1][0]=0.0;
    A[0][0][1][1]=3.0;    A[1][1][1][1]=1.0;    A[1][2][1][1]=1.0;
    A[0][0][1][2]=1.0;    A[1][1][1][2]=0.0;    A[1][2][1][2]=1.0;
    A[0][0][2][0]=2.0;    A[1][1][2][0]=3.0;    A[1][2][2][0]=0.0;
    A[0][0][2][1]=1.0;    A[1][1][2][1]=1.0;    A[1][2][2][1]=1.0;
    A[0][0][2][2]=0.0;    A[1][1][2][2]=3.0;    A[1][2][2][2]=3.0;

    A[2][1][0][0]=1.0;    A[2][2][0][0]=1.0;
    A[2][1][0][1]=0.0;    A[2][2][0][1]=3.0;
    A[2][1][0][2]=2.0;    A[2][2][0][2]=2.0;
    A[2][1][1][0]=0.0;    A[2][2][1][0]=2.0;
    A[2][1][1][1]=1.0;    A[2][2][1][1]=1.0;
    A[2][1][1][2]=4.0;    A[2][2][1][2]=3.0;
    A[2][1][2][0]=5.0;    A[2][2][2][0]=3.0;
    A[2][1][2][1]=1.0;    A[2][2][2][1]=1.0;
    A[2][1][2][2]=1.0;    A[2][2][2][2]=2.0;

    Vector x(3);
    x[0][0] = 1.0;    x[1][0] = 1.0;    x[2][0] = 1.0;
    x[0][1] = 2.0;    x[1][1] = 3.0;    x[2][1] = 0.0;
    x[0][2] = 3.0;    x[1][2] = 2.0;    x[2][2] = 2.0;

    Vector b(3);
    b[0][0] = 2.0;    b[1][0] = 2.0;    b[2][0] = 0.0;
    b[0][1] = 1.0;    b[1][1] = 3.0;    b[2][1] = 2.0;
    b[0][2] = 2.0;    b[1][2] = 2.0;    b[2][2] = 1.0;


    // D_00 = A_00
    auto D_00 = A[0][0];
    auto D_00_inv = D_00;
    D_00_inv.invert();
    // D_11 = A_11 - L_10 D_00_inv U_01
    //      = A_11
    auto D_11 = A[1][1];
    auto D_11_inv = D_11;
    D_11_inv.invert();
    // D_22 = A_22 - A_20 D_00_inv A_02 - A_21 D_11_inv A_12
    //      = A_22 - A_21 D_11_inv A_12
    auto D_22 = A[2][2] - A[2][1]*D_11_inv*A[1][2];
    auto D_22_inv = D_22;
    D_22_inv.invert();

    // define: z = M^-1(b - Ax)
    // where: M = (D + L_A) A^-1 (D + U_A)
    // lower triangular solve: (D + L) y = b - Ax

    Vector y(3);
    // y_0 = D_00_inv*[b_0 - (A_00*x_0 + A_01*x_1)]
    //     = D_00_inv*[b_0 - A_00*x_0]
    auto rhs = b[0];
    A[0][0].mmv(x[0], rhs);
    D_00_inv.mv(rhs, y[0]);

    // y_1 = D_11_inv*(b_1 - (A_10*x_0 + A_11*x_1 + A_12*x_2) - A_10*y_0)
    //     = D_11_inv*(b_1 - A_11*x_1)
    rhs = b[1];
    A[1][1].mmv(x[1], rhs);
    A[1][2].mmv(x[2], rhs);
    D_11_inv.mv(rhs, y[1]);

    // y_2 = D_22_inv*(b_2 - (A_20*x_0 + A_21*x_1 + A_22*x_2) - (A_20*y_0 + A_21*y_1))
    //     = D_22_inv*(b_2 - (A_21*x_1 + A_22*x_2) - (A_21*y_1))
    rhs = b[2];
    A[2][1].mmv(x[1], rhs);
    A[2][2].mmv(x[2], rhs);
    A[2][1].mmv(y[1], rhs);
    D_22_inv.mv(rhs, y[2]);


    // upper triangular solve: (D + U) z = Dy
    Vector z(3);
    // z_2 = y_2
    z[2] = y[2];

    // z_1 = y_1 - D_11_inv*A_12*z_2
    z[1] = y[1];
    auto temp = D_11_inv*A[1][2];
    temp.mmv(z[2], z[1]);

    // z_0 = y_0 - D_00_inv(A_01*z_1 + A_02*z_2)
    // z_0 = y_0
    z[0] = y[0];

    // z is now the update x_k+1 - x_k

    Dune::SeqDILU<Matrix,Vector,Vector> seqDILU(A, 1);
    Vector v(3);
    Vector d(b);
    A.mmv(x, d);
    seqDILU.apply(v, d);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            t.check(std::abs(v[i][j] - z[i][j]) <= myEps)
             << " Error in SeqDILUApplyIsCorrect2, v[" << i <<"][" << j << "]=" << v[i][j] << " != z[" << i <<"][" << j << "]=" << z[i][j];
        }
    }
}


void seqDILUApplyIsEqualToSeqILUApply()
{
    /*
        Tests that applying the DILU preconditioner is equivalent to applying an ILU preconditioner
        for a block diagonal matrix.

                 A               x     =     b
        | | 3  1| | 0  0| |   | |1| |     | |2| |
        | | 2  1| | 0  0| |   | |2| |     | |1| |
        |                 | x |     |  =  |     |
        | | 0  0| |-1  0| |   | |1| |     | |3| |
        | | 0  0| | 0 -1| |   | |1| |     | |4| |
    */

    const int N = 2;
    const int bz = 2;
    const int nonZeroes = 2;
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, bz, bz>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<double, bz>>;

    using ft = typename Vector::field_type;
    const ft myEps((ft)1e-6);
    Dune::TestSuite t;

    Matrix A(N, N, nonZeroes, Matrix::row_wise);
    for (auto row = A.createbegin(); row != A.createend(); ++row) {
        row.insert(row.index());
    }

    A[0][0][0][0]=3.0;
    A[0][0][0][1]=1.0;
    A[0][0][1][0]=2.0;
    A[0][0][1][1]=1.0;

    A[1][1][0][0]=-1.0;
    A[1][1][1][1]=-1.0;

    Vector x(2);
    x[0][0] = 1.0;
    x[0][1] = 2.0;
    x[1][0] = 1.0;
    x[1][1] = 1.0;

    Vector b(2);
    b[0][0] = 2.0;
    b[0][1] = 1.0;
    b[1][0] = 3.0;
    b[1][1] = 4.0;

    Dune::SeqDILU<Matrix,Vector,Vector> seqDILU(A, 1);
    Vector v_DILU(2);
    Vector d_DILU(b);
    A.mmv(x, d_DILU);
    seqDILU.apply(v_DILU, d_DILU);

    Dune::SeqILU<Matrix,Vector,Vector> seqILU(A, 1);
    Vector v_ILU(2);
    Vector d_ILU(b);
    A.mmv(x, d_ILU);
    seqDILU.apply(v_ILU, d_ILU);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            t.check(std::abs(v_DILU[i][j] - v_ILU[i][j]) <= myEps)
            << " Error in SeqDILUApplyIsEqualToSeqILUApply, v_DILU[" << i <<"][" << j << "]=" << v_DILU[i][j] << " != v_ILU[" << i <<"][" << j << "]=" << v_ILU[i][j];
        }
    }
}






int main() try
{
  seqDILUApplyIsCorrect1();
  seqDILUApplyIsCorrect2();
  seqDILUApplyIsEqualToSeqILUApply();

  return 0;
}
catch (std::exception& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
