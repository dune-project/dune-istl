// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
/**
 * \author: Marian Piatkowski, Steffen Müthing
 * \file
 * \brief Test MINRES and GMRes for complex matrices and complex rhs.
 */

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <complex>

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include "complexdata.hh"

typedef std::complex<double> FIELD_TYPE;


int main(int argc, char** argv)
{

  try {

    std::size_t N = 3;
    const int maxIter = int(N*N*N*N);
    const double reduction = 1e-16;

    std::cout << "============================================" << '\n'
              << "starting solver tests with complex matrices and complex rhs... "
              << std::endl
              << std::endl;

    std::cout << "============================================" << '\n'
              << "solving system with Hilbert matrix of size 10 times imaginary unit... "
              << std::endl
              << std::endl;

    Dune::FieldMatrix<std::complex<double>,10,10> hilbertmatrix;
    for(int i=0; i<10; i++) {
      for(int j=0; j<10; j++) {
        std::complex<double> temp(0.0,1./(i+j+1));
        hilbertmatrix[i][j] = temp;
      }
    }

    Dune::FieldVector<std::complex<double>,10> hilbertsol(1.0);
    Dune::FieldVector<std::complex<double>,10> hilbertiter(0.0);
    Dune::MatrixAdapter<Dune::FieldMatrix<std::complex<double>,10,10>,Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > hilbertadapter(hilbertmatrix);

    Dune::FieldVector<std::complex<double>,10> hilbertrhs(0.0);
    hilbertadapter.apply(hilbertsol,hilbertrhs);

    Dune::Richardson<Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > noprec(1.0);

    Dune::RestartedGMResSolver<Dune::FieldVector<std::complex<double>,10> > realgmrestest(hilbertadapter,noprec,reduction,maxIter,maxIter,2);
    Dune::InverseOperatorResult stat;
    realgmrestest.apply(hilbertiter,hilbertrhs,stat);

    std::cout << hilbertiter << std::endl;
    // error of solution
    hilbertiter -= hilbertsol;
    std::cout << "error of solution with GMRes:" << std::endl;
    std::cout << hilbertiter.two_norm() << std::endl;

    std::cout << "============================================" << '\n'
              << "solving system with complex matrix of size 10" << '\n'
              << "randomly generated with the Eigen library... "
              << std::endl << std::endl;

    Dune::FieldMatrix<std::complex<double>,10,10> complexmatrix(0.0), hermitianmatrix(0.0);
    Dune::FieldVector<std::complex<double>,10> complexsol, complexrhs(0.0), complexiter(0.0);

    // assemble randomly generated matrices from Eigen
    assemblecomplexmatrix(complexmatrix);
    assemblecomplexsol(complexsol);
    assemblehermitianmatrix(hermitianmatrix);

    Dune::MatrixAdapter<Dune::FieldMatrix<std::complex<double>,10,10>,Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > complexadapter(complexmatrix), hermitianadapter(hermitianmatrix);

    Dune::SeqJac<Dune::FieldMatrix<std::complex<double>,10,10>,Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10>,0> complexjacprec(complexmatrix,1,1.0);
    Dune::Richardson<Dune::FieldVector<std::complex<double>,10>,Dune::FieldVector<std::complex<double>,10> > complexnoprec(1.0);

    Dune::RestartedGMResSolver<Dune::FieldVector<std::complex<double>,10> > complexgmrestest(complexadapter,complexnoprec,1e-12,maxIter,maxIter*maxIter,2);

    complexadapter.apply(complexsol,complexrhs);
    complexgmrestest.apply(complexiter,complexrhs,stat);

    std::cout << complexiter << std::endl;
    // error of solution
    complexiter -= complexsol;
    std::cout << "error of solution with GMRes: " << complexiter.two_norm() << std::endl;

    std::cout << "============================================" << '\n'
              << "solving system with hermitian matrix of size 10" << '\n'
              << "randomly generated with the Eigen library... "
              << std::endl << std::endl;

    Dune::RestartedGMResSolver<Dune::FieldVector<std::complex<double>,10> > hermitiangmrestest(hermitianadapter,complexnoprec,1e-12,maxIter,maxIter*maxIter,2);
    Dune::MINRESSolver<Dune::FieldVector<std::complex<double>,10> > complexminrestest(hermitianadapter,complexnoprec,1e-12,maxIter,2);

    complexiter = 0.0;
    hermitianadapter.apply(complexsol,complexrhs);
    hermitiangmrestest.apply(complexiter,complexrhs,stat);

    std::cout << complexiter << std::endl;
    // error of solution
    complexiter -= complexsol;
    std::cout << "error of solution with GMRes: " << complexiter.two_norm() << std::endl;

    complexiter = 0.0;
    hermitianadapter.apply(complexsol,complexrhs);
    complexminrestest.apply(complexiter,complexrhs,stat);

    std::cout << complexiter << std::endl;
    // error of solution
    complexiter-= complexsol;
    std::cout << "error of solution with MinRes: " << complexiter.two_norm() << std::endl;

    return 0;

  } catch (Dune::Exception& e) {
    std::cerr << "DUNE reported an exception: " << e << std::endl;
    return 1;
  } catch (std::exception& e) {
    std::cerr << "C++ reported an exception: " << e.what() << std::endl;
    return 2;
  } catch (...) {
    std::cerr << "Unknown exception encountered!" << std::endl;
    return 3;
  }
}
