// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_EIGENVALUE_TEST_MATRIXINFO_HH
#define DUNE_ISTL_EIGENVALUE_TEST_MATRIXINFO_HH

#include <cmath>    // provides std::abs and std::sqrt
#include <cassert>  // provides assert
#include <limits>
#include <iostream>  // provides std::cout, std::endl

#include <dune/common/exceptions.hh>  // provides DUNE_THROW(...), Dune::Exception
#include <dune/common/fvector.hh>     // provides Dune::FieldVector

#include <dune/istl/blocklevel.hh>       // provides Dune::blockLevel
#include <dune/istl/bvector.hh>          // provides Dune::BlockVector
#include <dune/istl/superlu.hh>          // provides Dune::SuperLU
#include <dune/istl/preconditioners.hh>  // provides Dune::SeqGS
#include <dune/istl/solvers.hh>          // provides Dune::BiCGSTABSolver
#include <dune/istl/matrixmatrix.hh>     // provides Dune::transposeMatMultMat(...)

#include "../arpackpp.hh"        // provides Dune::ArPackPlusPlus_Algorithms
#include "../poweriteration.hh"  // provides Dune::PowerIteration_Algorithms


/**
 * \brief Class template which yields information related to a square
 *        matrix like its spectral (i.e. 2-norm) condition number.
 *
 * \todo The current implementation is limited to DUNE-ISTL
 *       BCRSMatrix types with blocklevel 2. An extension to
 *       blocklevel >= 2 might be provided in a future version.
 *
 * \tparam BCRSMatrix Type of a DUNE-ISTL BCRSMatrix whose properties
 *                    shall be considered; is assumed to have blocklevel
 *                    2 with square blocks.
 *
 * \author Sebastian Westerheide.
 */
template <typename BCRSMatrix>
class MatrixInfo
{
public:
  //! Type of the underlying field of the matrix
  typedef typename BCRSMatrix::field_type Real;

public:
  /**
   * \brief Construct from required parameters.
   *
   * \param[in] m                       The DUNE-ISTL BCRSMatrix
   *                                    whose properties shall be
   *                                    considered; is assumed to
   *                                    be square.
   * \param[in] verbose                 Verbosity setting.
   * \param[in] arppp_a_verbosity_level Verbosity setting of the
   *                                    underlying ARPACK++ algorithms.
   * \param[in] pia_verbosity_level     Verbosity setting of the
   *                                    underlying power iteration
   *                                    based algorithms.
   */
  MatrixInfo (const BCRSMatrix& m,
              const bool verbose = false,
              const unsigned int arppp_a_verbosity_level = 0,
              const unsigned int pia_verbosity_level = 0)
    : m_(m),
      verbose_(verbose),
      arppp_a_verbosity_level_(arppp_a_verbosity_level*verbose),
      pia_verbosity_level_(pia_verbosity_level*verbose),
      cond_2_(-1.0), symmetricity_assumed_(false)
  {
    // assert that BCRSMatrix type has blocklevel 2
    static_assert
      (Dune::blockLevel<BCRSMatrix>() == 2,
       "Only BCRSMatrices with blocklevel 2 are supported.");

    // assert that BCRSMatrix type has square blocks
    static_assert
      (BCRSMatrix::block_type::rows == BCRSMatrix::block_type::cols,
       "Only BCRSMatrices with square blocks are supported.");

    // assert that m_ is square
    const int nrows = m_.M() * BCRSMatrix::block_type::rows;
    const int ncols = m_.N() * BCRSMatrix::block_type::cols;
    if (nrows != ncols)
      DUNE_THROW(Dune::Exception,"Matrix is not square ("
                 << nrows << "x" << ncols << ").");
  }

  //! return spectral (i.e. 2-norm) condition number of the matrix
  inline Real getCond2 (const bool assume_symmetric = true) const
  {
    if (cond_2_ == -1.0 || symmetricity_assumed_ != assume_symmetric)
    {
      if (verbose_)
        std::cout << "    MatrixInfo: Computing 2-norm condition number"
                  << (assume_symmetric ? " (assuming that matrix is symmetric)." : ".")
                  << std::endl;

      if (assume_symmetric)
        cond_2_ = computeSymCond2();     // assume that m_ is symmetric
      else
        cond_2_ = computeNonSymCond2();  // don't assume that m_ is symmetric

      symmetricity_assumed_ = assume_symmetric;
    }
    return cond_2_;
  }

protected:
  //! Type of block vectors compatible with the rows of a BCRSMatrix
  //! object and its columns
  static const int bvBlockSize = BCRSMatrix::block_type::rows;
  typedef Dune::FieldVector<Real,bvBlockSize> BlockVectorBlock;
  typedef Dune::BlockVector<BlockVectorBlock> BlockVector;

protected:
  //! compute spectral (i.e. 2-norm) condition number of the matrix,
  //! while assuming that it is symmetric such that its largest/smallest
  //! magnitude eigenvalue can be used instead of its largest/smallest
  //! singular value to compute the spectral condition number
  inline Real computeSymCond2 () const
  {
    // 1) allocate memory for largest and smallest magnitude eigenvalue
    //    as well as the spectral (i.e. 2-norm) condition number
    Real lambda_max{}, lambda_min{}, cond_2{};

    // 2) allocate memory for starting vectors and approximated
    //    eigenvectors
    BlockVector x(m_.M());

#if HAVE_ARPACKPP
    // 3) setup ARPACK++ eigenvalue algorithms
    typedef Dune::ArPackPlusPlus_Algorithms<BCRSMatrix,BlockVector> ARPPP_A;
    const ARPPP_A arppp_a(m_,100000,arppp_a_verbosity_level_);
#endif  // HAVE_ARPACKPP

    // 4) setup power iteration based iterative eigenvalue algorithms
    typedef Dune::PowerIteration_Algorithms<BCRSMatrix,BlockVector> PIA;
    PIA pia(m_,20000,pia_verbosity_level_);
    static const bool avoidLinSolverCrime = true;

#if HAVE_SUPERLU
    // 5) select a linear solver for power iteration based iterative
    //    eigenvalue algorithms
    typedef Dune::SuperLU<BCRSMatrix> PIALS;
    const unsigned int piaLS_verbosity = 0;
    PIALS piaLS(pia.getIterationMatrix(),piaLS_verbosity);
#else
    // 5) select a linear solver for power iteration based iterative
    //    eigenvalue algorithms
    typedef Dune::SeqGS<BCRSMatrix,
                        typename PIA::IterationOperator::domain_type,
                        typename PIA::IterationOperator::range_type> PIAPC;
    PIAPC piaPC(pia.getIterationMatrix(),2,1.0);
    const double piaLS_reduction = 1e-02;
    const unsigned int piaLS_max_iter = 1000;
    const unsigned int piaLS_verbosity = 0;
    typedef Dune::BiCGSTABSolver<typename PIA::IterationOperator::domain_type> PIALS;
    PIALS piaLS(pia.getIterationOperator(),piaPC,
                piaLS_reduction,piaLS_max_iter,piaLS_verbosity);
#endif  // HAVE_SUPERLU

#if HAVE_ARPACKPP
    // 6) get largest magnitude eigenvalue via ARPACK++
    //    (assume that m_ is symmetric)
    {
      const Real epsilon = 0.0;
      // x = 1.0; (not supported yet)
      arppp_a.computeSymMaxMagnitude(epsilon,x,lambda_max);
    }
#else
    // 6) get largest magnitude eigenvalue via a combination of
    //    power and TLIME iteration (assume that m_ is symmetric)
    {
      const Real epsilonPI    = 1e-02;
      const Real epsilonTLIME = 1e-08;
      x = 1.0;
      // 6.1) perform power iteration for largest magnitude
      //       eigenvalue (predictor)
      pia.applyPowerIteration(epsilonPI,x,lambda_max);
      // 6.2) perform TLIME iteration to improve result (corrector)
      const Real gamma = m_.infinity_norm();
      const Real eta = 0.0;
      const Real delta = 1e-03;
      bool external;
      pia.template applyTLIMEIteration<PIALS,avoidLinSolverCrime>
        (gamma,eta,epsilonTLIME,piaLS,delta,2,external,x,lambda_max);
      assert(external);
    }
#endif  // HAVE_ARPACKPP

    // 7) get smallest magnitude eigenvalue via TLIME iteration
    //    (assume that m_ is symmetric)
    {
      const Real epsilon = std::sqrt(std::numeric_limits<Real>::epsilon());
      x = 1.0;
      // 7.1) perform TLIME iteration for smallest magnitude
      //      eigenvalue
      const Real gamma = 0.0;
      const Real eta = 0.0;
      const Real delta = 1e-03;
      bool external;
      pia.template applyTLIMEIteration<PIALS,avoidLinSolverCrime>
        (gamma,eta,epsilon,piaLS,delta,2,external,x,lambda_min);
      assert(external);
    }

    // 8) check largest magnitude eigenvalue (we have
    //    ||m|| >= |lambda| for each eigenvalue lambda
    //    of a matrix m and each matrix norm ||.||)
    if (std::abs(lambda_max) > m_.infinity_norm())
      DUNE_THROW(Dune::Exception,"Absolute value of approximated "
                 << "largest magnitude eigenvalue is greater than "
                 << "infinity norm of the matrix!");

    // 9) output largest magnitude eigenvalue
    if (verbose_)
      std::cout << "    Largest magnitude eigenvalue λ_max = "
                << lambda_max << std::endl;

    // 10) output smallest magnitude eigenvalue
    if (verbose_)
      std::cout << "    Smallest magnitude eigenvalue λ_min = "
                << lambda_min << std::endl;

    // 11) compute spectral (i.e. 2-norm) condition number
    //     (assume that m_ is symmetric)
    cond_2 = std::abs(lambda_max / lambda_min);

    // 12) output spectral (i.e. 2-norm) condition number
    if (verbose_)
      std::cout << "    2-norm condition number cond_2 = "
                << cond_2 << std::endl;

    // 13) return spectral (i.e. 2-norm) condition number
    return cond_2;
  }

  //! compute spectral (i.e. 2-norm) condition number of the matrix,
  //! without assuming that it is symmetric
  inline Real computeNonSymCond2 () const
  {
    // 1) allocate memory for largest and smallest singular value
    //    as well as the spectral (i.e. 2-norm) condition number
    Real sigma_max, sigma_min, cond_2;

    // 2) allocate memory for starting vectors and approximated
    //    eigenvectors respectively singular vectors
    BlockVector x(m_.M());

#if HAVE_ARPACKPP
    // 3) setup ARPACK++ eigenvalue algorithms
    typedef Dune::ArPackPlusPlus_Algorithms<BCRSMatrix,BlockVector> ARPPP_A;
    const ARPPP_A arppp_a(m_,100000,arppp_a_verbosity_level_);
#endif  // HAVE_ARPACKPP

    // 4) compute m^t*m
    BCRSMatrix mtm;
    Dune::transposeMatMultMat(mtm,m_,m_);

    // 5) allocate memory for largest and smallest magnitude
    //    eigenvalue of m^t*m
    Real lambda_max{}, lambda_min{};

    // 6) setup power iteration based iterative eigenvalue algorithms
    //    for m^t*m
    typedef Dune::PowerIteration_Algorithms<BCRSMatrix,BlockVector> PIA;
    PIA pia(mtm,20000,pia_verbosity_level_);
    static const bool avoidLinSolverCrime = true;

#if HAVE_SUPERLU
    // 7) select a linear solver for power iteration based iterative
    //    eigenvalue algorithms for m^t*m
    typedef Dune::SuperLU<BCRSMatrix> PIALS;
    const unsigned int piaLS_verbosity = 0;
    PIALS piaLS(pia.getIterationMatrix(),piaLS_verbosity);
#else
    // 7) select a linear solver for power iteration based iterative
    //    eigenvalue algorithms for m^t*m
    typedef Dune::SeqGS<BCRSMatrix,
                        typename PIA::IterationOperator::domain_type,
                        typename PIA::IterationOperator::range_type> PIAPC;
    PIAPC piaPC(pia.getIterationMatrix(),2,1.0);
    const double piaLS_reduction = 1e-02;
    const unsigned int piaLS_max_iter = 1000;
    const unsigned int piaLS_verbosity = 0;
    typedef Dune::BiCGSTABSolver<typename PIA::IterationOperator::domain_type> PIALS;
    PIALS piaLS(pia.getIterationOperator(),piaPC,
                piaLS_reduction,piaLS_max_iter,piaLS_verbosity);
#endif  // HAVE_SUPERLU

#if HAVE_ARPACKPP
    // 8) get largest singular value via ARPACK++
    {
      const Real epsilon = 0.0;
      // x = 1.0; (not supported yet)
      arppp_a.computeNonSymMax(epsilon,x,sigma_max);
    }
#else
    // 8) get largest singular value as square root of the largest
    //    magnitude eigenvalue of m^t*m via a combination of power
    //    and TLIME iteration
    {
      const Real epsilonPI    = 1e-02;
      const Real epsilonTLIME = 1e-08;
      x = 1.0;
      // 8.1) perform power iteration for largest magnitude
      //      eigenvalue of m^t*m (predictor)
      pia.applyPowerIteration(epsilonPI,x,lambda_max);
      // 8.2) perform TLIME iteration to improve result (corrector)
      const Real gamma = mtm.infinity_norm();
      const Real eta = 0.0;
      const Real delta = 1e-03;
      bool external;
      pia.template applyTLIMEIteration<PIALS,avoidLinSolverCrime>
        (gamma,eta,epsilonTLIME,piaLS,delta,2,external,x,lambda_max);
      assert(external);
      // 8.3) get largest singular value
      sigma_max = std::sqrt(lambda_max);
    }
#endif  // HAVE_ARPACKPP

    // 9) get smallest singular value as square root of the smallest
    //    magnitude eigenvalue of m^t*m via TLIME iteration
    {
      const Real epsilon = std::sqrt(std::numeric_limits<Real>::epsilon());
      x = 1.0;
      // 9.1) perform TLIME iteration for smallest magnitude
      //      eigenvalue of m^t*m
      const Real gamma = 0.0;
      const Real eta = 0.0;
      const Real delta = 1e-03;
      bool external;
      pia.template applyTLIMEIteration<PIALS,avoidLinSolverCrime>
        (gamma,eta,epsilon,piaLS,delta,2,external,x,lambda_min);
      assert(external);
      // 9.2) get smallest singular value
      sigma_min = std::sqrt(lambda_min);
    }

    // 10) check largest magnitude eigenvalue (we have
    //     ||m|| >= |lambda| for each eigenvalue lambda
    //     of a matrix m and each matrix norm ||.||)
    if (std::abs(lambda_max) > mtm.infinity_norm())
      DUNE_THROW(Dune::Exception,"Absolute value of approximated "
                 << "largest magnitude eigenvalue is greater than "
                 << "infinity norm of the matrix!");

    // 11) output largest singular value
    if (verbose_)
      std::cout << "    Largest singular value σ_max = "
                << sigma_max << std::endl;

    // 12) output smallest singular value
    if (verbose_)
      std::cout << "    Smallest singular value σ_min = "
                << sigma_min << std::endl;

    // 13) compute spectral (i.e. 2-norm) condition number
    cond_2 = sigma_max / sigma_min;

    // 14) output spectral (i.e. 2-norm) condition number
    if (verbose_)
      std::cout << "    2-norm condition number cond_2 = "
                << cond_2 << std::endl;

    // 15) return spectral (i.e. 2-norm) condition number
    return cond_2;
  }

protected:
  // parameters related to computation of matrix information
  const BCRSMatrix& m_;

  // verbosity setting
  const bool verbose_;
  const unsigned int arppp_a_verbosity_level_;
  const unsigned int pia_verbosity_level_;

  // memory for storing matrix information
  // (mutable as matrix information is computed on demand)
  mutable Real cond_2_;
  mutable bool symmetricity_assumed_;
};


#endif  // DUNE_ISTL_EIGENVALUE_TEST_MATRIXINFO_HH
