// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_EIGENVALUE_ARPACKPP_HH
#define DUNE_ISTL_EIGENVALUE_ARPACKPP_HH

#if HAVE_ARPACKPP || defined DOXYGEN

#include <cmath>  // provides std::abs, std::pow, std::sqrt

#include <iostream>  // provides std::cout, std::endl
#include <string>    // provides std::string

#include <dune/common/fvector.hh>     // provides Dune::FieldVector
#include <dune/common/exceptions.hh>  // provides DUNE_THROW(...)

#include <dune/istl/blocklevel.hh>     // provides Dune::blockLevel
#include <dune/istl/bvector.hh>        // provides Dune::BlockVector
#include <dune/istl/istlexception.hh>  // provides Dune::ISTLError
#include <dune/istl/io.hh>             // provides Dune::printvector(...)

#ifdef Status
#undef Status        // prevent preprocessor from damaging the ARPACK++
                     // code when "X11/Xlib.h" is included (the latter
                     // defines Status as "#define Status int" and
                     // ARPACK++ provides a class with a method called
                     // Status)
#endif
#include "arssym.h"  // provides ARSymStdEig

namespace Dune
{

  /** @addtogroup ISTL_Eigenvalue
    @{
  */

  namespace Impl {
    /**
     *        Wrapper for a DUNE-ISTL BCRSMatrix which can be used
     *        together with those algorithms of the ARPACK++ library
     *        which solely perform the products A*v and/or A^T*A*v
     *        and/or A*A^T*v.
     *
     * \todo The current implementation is limited to DUNE-ISTL
     *       BCRSMatrix types with blocklevel 2. An extension to
     *       blocklevel >= 2 might be provided in a future version.
     *
     * \tparam BCRSMatrix Type of a DUNE-ISTL BCRSMatrix;
     *                    is assumed to have blocklevel 2.
     *
     * \author Sebastian Westerheide.
     */
    template <class BCRSMatrix>
    class ArPackPlusPlus_BCRSMatrixWrapper
    {
    public:
      //! Type of the underlying field of the matrix
      typedef typename BCRSMatrix::field_type Real;

    public:
      //! Construct from BCRSMatrix A
      ArPackPlusPlus_BCRSMatrixWrapper (const BCRSMatrix& A)
        : A_(A),
          m_(A_.M() * mBlock), n_(A_.N() * nBlock)
      {
        // assert that BCRSMatrix type has blocklevel 2
        static_assert
          (blockLevel<BCRSMatrix>() == 2,
            "Only BCRSMatrices with blocklevel 2 are supported.");

        // allocate memory for auxiliary block vector objects
        // which are compatible to matrix rows / columns
        domainBlockVector.resize(A_.N());
        rangeBlockVector.resize(A_.M());
      }

      //! Perform matrix-vector product w = A*v
      inline void multMv (Real* v, Real* w)
      {
        // get vector v as an object of appropriate type
        arrayToDomainBlockVector(v,domainBlockVector);

        // perform matrix-vector product
        A_.mv(domainBlockVector,rangeBlockVector);

        // get vector w from object of appropriate type
        rangeBlockVectorToArray(rangeBlockVector,w);
      };

      //! Perform matrix-vector product w = A^T*A*v
      inline void multMtMv (Real* v, Real* w)
      {
        // get vector v as an object of appropriate type
        arrayToDomainBlockVector(v,domainBlockVector);

        // perform matrix-vector product
        A_.mv(domainBlockVector,rangeBlockVector);
        A_.mtv(rangeBlockVector,domainBlockVector);

        // get vector w from object of appropriate type
        domainBlockVectorToArray(domainBlockVector,w);
      };

      //! Perform matrix-vector product w = A*A^T*v
      inline void multMMtv (Real* v, Real* w)
      {
        // get vector v as an object of appropriate type
        arrayToRangeBlockVector(v,rangeBlockVector);

        // perform matrix-vector product
        A_.mtv(rangeBlockVector,domainBlockVector);
        A_.mv(domainBlockVector,rangeBlockVector);

        // get vector w from object of appropriate type
        rangeBlockVectorToArray(rangeBlockVector,w);
      };

      //! Return number of rows in the matrix
      inline int nrows () const { return m_; }

      //! Return number of columns in the matrix
      inline int ncols () const { return n_; }

    protected:
      // Number of rows and columns in each block of the matrix
      constexpr static int mBlock = BCRSMatrix::block_type::rows;
      constexpr static int nBlock = BCRSMatrix::block_type::cols;

      // Type of vectors in the domain of the linear map associated with
      // the matrix, i.e. block vectors compatible to matrix rows
      constexpr static int dbvBlockSize = nBlock;
      typedef Dune::FieldVector<Real,dbvBlockSize> DomainBlockVectorBlock;
      typedef Dune::BlockVector<DomainBlockVectorBlock> DomainBlockVector;

      // Type of vectors in the range of the linear map associated with
      // the matrix, i.e. block vectors compatible to matrix columns
      constexpr static int rbvBlockSize = mBlock;
      typedef Dune::FieldVector<Real,rbvBlockSize> RangeBlockVectorBlock;
      typedef Dune::BlockVector<RangeBlockVectorBlock> RangeBlockVector;

      // Types for vector index access
      typedef typename DomainBlockVector::size_type dbv_size_type;
      typedef typename RangeBlockVector::size_type rbv_size_type;
      typedef typename DomainBlockVectorBlock::size_type dbvb_size_type;
      typedef typename RangeBlockVectorBlock::size_type rbvb_size_type;

      // Get vector v from a block vector object which is compatible to
      // matrix rows
      static inline void
      domainBlockVectorToArray (const DomainBlockVector& dbv, Real* v)
      {
        for (dbv_size_type block = 0; block < dbv.N(); ++block)
          for (dbvb_size_type iBlock = 0; iBlock < dbvBlockSize; ++iBlock)
            v[block*dbvBlockSize + iBlock] = dbv[block][iBlock];
      }

      // Get vector v from a block vector object which is compatible to
      // matrix columns
      static inline void
      rangeBlockVectorToArray (const RangeBlockVector& rbv, Real* v)
      {
        for (rbv_size_type block = 0; block < rbv.N(); ++block)
          for (rbvb_size_type iBlock = 0; iBlock < rbvBlockSize; ++iBlock)
            v[block*rbvBlockSize + iBlock] = rbv[block][iBlock];
      }

    public:
      //! Get vector v as a block vector object which is compatible to
      //! matrix rows
      static inline void arrayToDomainBlockVector (const Real* v,
        DomainBlockVector& dbv)
      {
        for (dbv_size_type block = 0; block < dbv.N(); ++block)
          for (dbvb_size_type iBlock = 0; iBlock < dbvBlockSize; ++iBlock)
            dbv[block][iBlock] = v[block*dbvBlockSize + iBlock];
      }

      //! Get vector v as a block vector object which is compatible to
      //! matrix columns
      static inline void arrayToRangeBlockVector (const Real* v,
        RangeBlockVector& rbv)
      {
        for (rbv_size_type block = 0; block < rbv.N(); ++block)
          for (rbvb_size_type iBlock = 0; iBlock < rbvBlockSize; ++iBlock)
            rbv[block][iBlock] = v[block*rbvBlockSize + iBlock];
      }

    protected:
      // The DUNE-ISTL BCRSMatrix
      const BCRSMatrix& A_;

      // Number of rows and columns in the matrix
      const int m_, n_;

      // Auxiliary block vector objects which are
      // compatible to matrix rows / columns
      mutable DomainBlockVector domainBlockVector;
      mutable RangeBlockVector rangeBlockVector;
    };
  } // end namespace Impl

  /**
   * \brief Wrapper to use a range of ARPACK++ eigenvalue solvers
   *
   *        A class template for performing some eigenvalue algorithms
   *        provided by the ARPACK++ library which is based on the implicitly
   *        restarted Arnoldi/Lanczos method (IRAM/IRLM), a synthesis of the
   *        Arnoldi/Lanczos process with the implicitily shifted QR technique.
   *        The method is designed to compute eigenvalue-eigenvector pairs of
   *        large scale sparse nonsymmetric/symmetric matrices. This class
   *        template uses the algorithms to compute the dominant (i.e. largest
   *        magnitude) and least dominant (i.e. smallest magnitude) eigenvalue
   *        as well as the spectral condition number of square, symmetric
   *        matrices and to compute the largest and smallest singular value as
   *        well as the spectral condition number of nonsymmetric matrices.
   *
   * \note For a recent version of the ARPACK++ library working with recent
   *       compiler versions see "http://reuter.mit.edu/software/arpackpatch/"
   *       or the git repository "https://github.com/m-reuter/arpackpp.git".
   *
   * \note Note that the Arnoldi/Lanczos process currently is initialized
   *       using a vector which is randomly generated by ARPACK++. This
   *       could be changed in a future version, since ARPACK++ supports
   *       manual initialization of this vector.
   *
   * \todo The current implementation is limited to DUNE-ISTL BCRSMatrix types
   *       with blocklevel 2. An extension to blocklevel >= 2 might be provided
   *       in a future version.
   *
   * \todo Maybe make ARPACK++ parameter ncv available to the user.
   *
   * \tparam BCRSMatrix  Type of a DUNE-ISTL BCRSMatrix whose eigenvalues
   *                     respectively singular values shall be considered;
   *                     is assumed to have blocklevel 2.
   * \tparam BlockVector Type of the associated vectors; compatible with the
   *                     rows of a BCRSMatrix object (if #rows >= #ncols) or
   *                     its columns (if #rows < #ncols).
   *
   * \author Sebastian Westerheide.
   */
  template <typename BCRSMatrix, typename BlockVector>
  class ArPackPlusPlus_Algorithms
  {
  public:
    typedef typename BlockVector::field_type Real;

  public:
    /**
     * \brief Construct from required parameters.
     *
     * \param[in] m               The DUNE-ISTL BCRSMatrix whose eigenvalues
     *                            resp. singular values shall be considered.
     * \param[in] nIterationsMax  Influences the maximum number of Arnoldi
     *                            update iterations allowed; depending on the
     *                            algorithm, c*nIterationsMax iterations may
     *                            be performed, where c is a natural number.
     * \param[in] verbosity_level Verbosity setting;
     *                            >= 1: algorithms print a preamble and
     *                                  the final result,
     *                            >= 2: algorithms print information about
     *                                  the problem solved using ARPACK++,
     *                            >= 3: the final result output includes
     *                                  the approximated eigenvector,
     *                            >= 4: sets the ARPACK(++) verbosity mode.
     */
    ArPackPlusPlus_Algorithms (const BCRSMatrix& m,
                               const unsigned int nIterationsMax = 100000,
                               const unsigned int verbosity_level = 0)
      : m_(m), nIterationsMax_(nIterationsMax),
        verbosity_level_(verbosity_level),
        nIterations_(0),
        title_("    ArPackPlusPlus_Algorithms: "),
        blank_(title_.length(),' ')
    {}

    /**
     * \brief Assume the matrix to be square, symmetric and perform IRLM
     *        to compute an approximation lambda of its dominant
     *        (i.e. largest magnitude) eigenvalue and the corresponding
     *        approximation x of an associated eigenvector.
     *
     * \param[in]  epsilon The target relative accuracy of Ritz values
     *                     (0 == machine precision).
     * \param[out] lambda  The approximated dominant eigenvalue.
     * \param[out] x       The associated approximated eigenvector.
     */
    inline void computeSymMaxMagnitude (const Real& epsilon,
                                        BlockVector& x, Real& lambda) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_ << "Computing an approximation of "
                  << "the dominant eigenvalue of a matrix which "
                  << "is assumed to be symmetric." << std::endl;

      // use type ArPackPlusPlus_BCRSMatrixWrapper to store matrix information
      // and to perform the product A*v (LU decomposition is not used)
      typedef Impl::ArPackPlusPlus_BCRSMatrixWrapper<BCRSMatrix> WrappedMatrix;
      WrappedMatrix A(m_);

      // get number of rows and columns in A
      const int nrows = A.nrows();
      const int ncols = A.ncols();

      // assert that A is square
      if (nrows != ncols)
        DUNE_THROW(Dune::ISTLError,"Matrix is not square ("
                   << nrows << "x" << ncols << ").");

      // allocate memory for variables, set parameters
      const int nev = 1;                     // Number of eigenvalues to compute
      int ncv = std::min(20, nrows);         // Number of Arnoldi vectors generated at each iteration (0 == auto)
      const Real tol = epsilon;              // Stopping tolerance (relative accuracy of Ritz values) (0 == machine precision)
      const int maxit = nIterationsMax_*nev; // Maximum number of Arnoldi update iterations allowed   (0 == 100*nev)
      Real* ev = new Real[nev];              // Computed eigenvalues of A
      const bool ivec = true;                // Flag deciding if eigenvectors shall be determined
      int nconv;                             // Number of converged eigenvalues

      // define what we need: eigenvalues with largest magnitude
      char which[] = "LM";
      ARSymStdEig<Real,WrappedMatrix>
        dprob(nrows, nev, &A, &WrappedMatrix::multMv, which, ncv, tol, maxit);

      // set ARPACK verbosity mode if requested
      if (verbosity_level_ > 3) dprob.Trace();

      // find eigenvalues and eigenvectors of A, obtain the eigenvalues
      nconv = dprob.Eigenvalues(ev,ivec);

      // obtain approximated dominant eigenvalue of A
      lambda = ev[nev-1];

      // obtain associated approximated eigenvector of A
      Real* x_raw = dprob.RawEigenvector(nev-1);
      WrappedMatrix::arrayToDomainBlockVector(x_raw,x);

      // obtain number of Arnoldi update iterations actually taken
      nIterations_ = dprob.GetIter();

      // compute residual norm
      BlockVector r(x);
      Real* Ax_raw = new Real[nrows];
      A.multMv(x_raw,Ax_raw);
      WrappedMatrix::arrayToDomainBlockVector(Ax_raw,r);
      r.axpy(-lambda,x);
      const Real r_norm = r.two_norm();

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        if (verbosity_level_ > 1)
        {
          // print some information about the problem
          std::cout << blank_ << "Obtained eigenvalues of A by solving "
                    << "A*x = λ*x using the ARPACK++ class ARSym"
                    << "StdEig:" << std::endl;
          std::cout << blank_ << "       converged eigenvalues of A: "
                    << nconv << " / " << nev << std::endl;
          std::cout << blank_ << "         dominant eigenvalue of A: "
                    << lambda << std::endl;
        }
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "λ = " << lambda << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated eigenvector via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }

      // free dynamically allocated memory
      delete[] Ax_raw;
      delete[] ev;
    }

    /**
     * \brief Assume the matrix to be square, symmetric and perform IRLM
     *        to compute an approximation lambda of its least dominant
     *        (i.e. smallest magnitude) eigenvalue and the corresponding
     *        approximation x of an associated eigenvector.
     *
     * \param[in]  epsilon The target relative accuracy of Ritz values
     *                     (0 == machine precision).
     * \param[out] lambda  The approximated least dominant eigenvalue.
     * \param[out] x       The associated approximated eigenvector.
     */
    inline void computeSymMinMagnitude (const Real& epsilon,
                                        BlockVector& x, Real& lambda) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_ << "Computing an approximation of the "
                  << "least dominant eigenvalue of a matrix which "
                  << "is assumed to be symmetric." << std::endl;

      // use type ArPackPlusPlus_BCRSMatrixWrapper to store matrix information
      // and to perform the product A*v (LU decomposition is not used)
      typedef Impl::ArPackPlusPlus_BCRSMatrixWrapper<BCRSMatrix> WrappedMatrix;
      WrappedMatrix A(m_);

      // get number of rows and columns in A
      const int nrows = A.nrows();
      const int ncols = A.ncols();

      // assert that A is square
      if (nrows != ncols)
        DUNE_THROW(Dune::ISTLError,"Matrix is not square ("
                   << nrows << "x" << ncols << ").");

      // allocate memory for variables, set parameters
      const int nev = 1;                     // Number of eigenvalues to compute
      int ncv = std::min(20, nrows);         // Number of Arnoldi vectors generated at each iteration (0 == auto)
      const Real tol = epsilon;              // Stopping tolerance (relative accuracy of Ritz values) (0 == machine precision)
      const int maxit = nIterationsMax_*nev; // Maximum number of Arnoldi update iterations allowed   (0 == 100*nev)
      Real* ev = new Real[nev];              // Computed eigenvalues of A
      const bool ivec = true;                // Flag deciding if eigenvectors shall be determined
      int nconv;                             // Number of converged eigenvalues

      // define what we need: eigenvalues with smallest magnitude
      char which[] = "SM";
      ARSymStdEig<Real,WrappedMatrix>
        dprob(nrows, nev, &A, &WrappedMatrix::multMv, which, ncv, tol, maxit);

      // set ARPACK verbosity mode if requested
      if (verbosity_level_ > 3) dprob.Trace();

      // find eigenvalues and eigenvectors of A, obtain the eigenvalues
      nconv = dprob.Eigenvalues(ev,ivec);

      // obtain approximated least dominant eigenvalue of A
      lambda = ev[nev-1];

      // obtain associated approximated eigenvector of A
      Real* x_raw = dprob.RawEigenvector(nev-1);
      WrappedMatrix::arrayToDomainBlockVector(x_raw,x);

      // obtain number of Arnoldi update iterations actually taken
      nIterations_ = dprob.GetIter();

      // compute residual norm
      BlockVector r(x);
      Real* Ax_raw = new Real[nrows];
      A.multMv(x_raw,Ax_raw);
      WrappedMatrix::arrayToDomainBlockVector(Ax_raw,r);
      r.axpy(-lambda,x);
      const Real r_norm = r.two_norm();

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        if (verbosity_level_ > 1)
        {
          // print some information about the problem
          std::cout << blank_ << "Obtained eigenvalues of A by solving "
                    << "A*x = λ*x using the ARPACK++ class ARSym"
                    << "StdEig:" << std::endl;
          std::cout << blank_ << "       converged eigenvalues of A: "
                    << nconv << " / " << nev << std::endl;
          std::cout << blank_ << "   least dominant eigenvalue of A: "
                    << lambda << std::endl;
        }
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "λ = " << lambda << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated eigenvector via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }

      // free dynamically allocated memory
      delete[] Ax_raw;
      delete[] ev;
    }

    /**
     * \brief Assume the matrix to be square, symmetric and perform IRLM
     *        to compute an approximation of its spectral condition number
     *        which, for symmetric matrices, can be expressed as the ratio
     *        of the dominant eigenvalue's magnitude and the least dominant
     *        eigenvalue's magnitude.
     *
     * \param[in]  epsilon The target relative accuracy of Ritz values
     *                     (0 == machine precision).
     * \param[out] cond_2  The approximated spectral condition number.
     */
    inline void computeSymCond2 (const Real& epsilon, Real& cond_2) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_ << "Computing an approximation of the "
                  << "spectral condition number of a matrix which "
                  << "is assumed to be symmetric." << std::endl;

      // use type ArPackPlusPlus_BCRSMatrixWrapper to store matrix information
      // and to perform the product A*v (LU decomposition is not used)
      typedef Impl::ArPackPlusPlus_BCRSMatrixWrapper<BCRSMatrix> WrappedMatrix;
      WrappedMatrix A(m_);

      // get number of rows and columns in A
      const int nrows = A.nrows();
      const int ncols = A.ncols();

      // assert that A is square
      if (nrows != ncols)
        DUNE_THROW(Dune::ISTLError,"Matrix is not square ("
                   << nrows << "x" << ncols << ").");

      // allocate memory for variables, set parameters
      const int nev = 2;                     // Number of eigenvalues to compute
      int ncv = std::min(20, nrows);         // Number of Arnoldi vectors generated at each iteration (0 == auto)
      const Real tol = epsilon;              // Stopping tolerance (relative accuracy of Ritz values) (0 == machine precision)
      const int maxit = nIterationsMax_*nev; // Maximum number of Arnoldi update iterations allowed   (0 == 100*nev)
      Real* ev = new Real[nev];              // Computed eigenvalues of A
      const bool ivec = true;                // Flag deciding if eigenvectors shall be determined
      int nconv;                             // Number of converged eigenvalues

      // define what we need: eigenvalues from both ends of the spectrum
      char which[] = "BE";
      ARSymStdEig<Real,WrappedMatrix>
        dprob(nrows, nev, &A, &WrappedMatrix::multMv, which, ncv, tol, maxit);

      // set ARPACK verbosity mode if requested
      if (verbosity_level_ > 3) dprob.Trace();

      // find eigenvalues and eigenvectors of A, obtain the eigenvalues
      nconv = dprob.Eigenvalues(ev,ivec);

      // obtain approximated dominant and least dominant eigenvalue of A
      const Real& lambda_max = ev[nev-1];
      const Real& lambda_min = ev[0];

      // obtain associated approximated eigenvectors of A
      Real* x_max_raw = dprob.RawEigenvector(nev-1);
      Real* x_min_raw = dprob.RawEigenvector(0);

      // obtain approximated spectral condition number of A
      cond_2 = std::abs(lambda_max / lambda_min);

      // obtain number of Arnoldi update iterations actually taken
      nIterations_ = dprob.GetIter();

      // compute each residual norm
      Real* Ax_max_raw = new Real[nrows];
      Real* Ax_min_raw = new Real[nrows];
      A.multMv(x_max_raw,Ax_max_raw);
      A.multMv(x_min_raw,Ax_min_raw);
      Real r_max_norm = 0.0;
      Real r_min_norm = 0.0;
      for (int i = 0; i < nrows; ++i)
      {
        r_max_norm += std::pow(Ax_max_raw[i] - lambda_max * x_max_raw[i],2);
        r_min_norm += std::pow(Ax_min_raw[i] - lambda_min * x_min_raw[i],2);
      }
      r_max_norm = std::sqrt(r_max_norm);
      r_min_norm = std::sqrt(r_min_norm);

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        if (verbosity_level_ > 1)
        {
          // print some information about the problem
          std::cout << blank_ << "Obtained eigenvalues of A by solving "
                    << "A*x = λ*x using the ARPACK++ class ARSym"
                    << "StdEig:" << std::endl;
          std::cout << blank_ << "       converged eigenvalues of A: "
                    << nconv << " / " << nev << std::endl;
          std::cout << blank_ << "         dominant eigenvalue of A: "
                    << lambda_max << std::endl;
          std::cout << blank_ << "   least dominant eigenvalue of A: "
                    << lambda_min << std::endl;
          std::cout << blank_ << "   spectral condition number of A: "
                    << cond_2 << std::endl;
        }
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = {" << r_max_norm << ","
                  << r_min_norm << "}, " << "λ = {"
                  << lambda_max << "," << lambda_min
                  << "}): cond_2 = " << cond_2 << std::endl;
      }

      // free dynamically allocated memory
      delete[] Ax_min_raw;
      delete[] Ax_max_raw;
      delete[] ev;
    }

    /**
     * \brief Assume the matrix to be nonsymmetric and perform IRLM
     *        to compute an approximation sigma of its largest
     *        singlar value and the corresponding approximation x of
     *        an associated singular vector.
     *
     * \param[in]  epsilon The target relative accuracy of Ritz values
     *                     (0 == machine precision).
     * \param[out] sigma   The approximated largest singlar value.
     * \param[out] x       The associated approximated right-singular
     *                     vector (if #rows >= #ncols) respectively
     *                     left-singular vector (if #rows < #ncols).
     */
    inline void computeNonSymMax (const Real& epsilon,
                                  BlockVector& x, Real& sigma) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_ << "Computing an approximation of the "
                  << "largest singular value of a matrix which "
                  << "is assumed to be nonsymmetric." << std::endl;

      // use type ArPackPlusPlus_BCRSMatrixWrapper to store matrix information
      // and to perform the product A^T*A*v (LU decomposition is not used)
      typedef Impl::ArPackPlusPlus_BCRSMatrixWrapper<BCRSMatrix> WrappedMatrix;
      WrappedMatrix A(m_);

      // get number of rows and columns in A
      const int nrows = A.nrows();
      const int ncols = A.ncols();

      // assert that A has more rows than columns (extend code later to the opposite case!)
      if (nrows < ncols)
        DUNE_THROW(Dune::ISTLError,"Matrix has less rows than "
                   << "columns (" << nrows << "x" << ncols << ")."
                   << " This case is not implemented, yet.");

      // allocate memory for variables, set parameters
      const int nev = 1;                     // Number of eigenvalues to compute
      int ncv = std::min(20, nrows);         // Number of Arnoldi vectors generated at each iteration (0 == auto)
      const Real tol = epsilon;              // Stopping tolerance (relative accuracy of Ritz values) (0 == machine precision)
      const int maxit = nIterationsMax_*nev; // Maximum number of Arnoldi update iterations allowed   (0 == 100*nev)
      Real* ev = new Real[nev];              // Computed eigenvalues of A^T*A
      const bool ivec = true;                // Flag deciding if eigenvectors shall be determined
      int nconv;                             // Number of converged eigenvalues

      // define what we need: eigenvalues with largest algebraic value
      char which[] = "LA";
      ARSymStdEig<Real,WrappedMatrix>
        dprob(ncols, nev, &A, &WrappedMatrix::multMtMv, which, ncv, tol, maxit);

      // set ARPACK verbosity mode if requested
      if (verbosity_level_ > 3) dprob.Trace();

      // find eigenvalues and eigenvectors of A^T*A, obtain the eigenvalues
      nconv = dprob.Eigenvalues(ev,ivec);

      // obtain approximated largest eigenvalue of A^T*A
      const Real& lambda = ev[nev-1];

      // obtain associated approximated eigenvector of A^T*A
      Real* x_raw = dprob.RawEigenvector(nev-1);
      WrappedMatrix::arrayToDomainBlockVector(x_raw,x);

      // obtain number of Arnoldi update iterations actually taken
      nIterations_ = dprob.GetIter();

      // compute residual norm
      BlockVector r(x);
      Real* AtAx_raw = new Real[ncols];
      A.multMtMv(x_raw,AtAx_raw);
      WrappedMatrix::arrayToDomainBlockVector(AtAx_raw,r);
      r.axpy(-lambda,x);
      const Real r_norm = r.two_norm();

      // calculate largest singular value of A (note that
      // x is right-singular / left-singular vector of A)
      sigma = std::sqrt(lambda);

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        if (verbosity_level_ > 1)
        {
          // print some information about the problem
          std::cout << blank_ << "Obtained singular values of A by sol"
                    << "ving (A^T*A)*x = σ²*x using the ARPACK++ "
                    << "class ARSymStdEig:" << std::endl;
          std::cout << blank_ << "   converged eigenvalues of A^T*A: "
                    << nconv << " / " << nev << std::endl;
          std::cout << blank_ << "      largest eigenvalue of A^T*A: "
                    << lambda << std::endl;
          std::cout << blank_ << "   => largest singular value of A: "
                    << sigma << std::endl;
        }
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "σ = " << sigma << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated right-singular / left-singular vector
          // via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }

      // free dynamically allocated memory
      delete[] AtAx_raw;
      delete[] ev;
    }

    /**
     * \brief Assume the matrix to be nonsymmetric and perform IRLM
     *        to compute an approximation sigma of its smallest
     *        singlar value and the corresponding approximation x of
     *        an associated singular vector.
     *
     * \param[in]  epsilon The target relative accuracy of Ritz values
     *                     (0 == machine precision).
     * \param[out] sigma   The approximated smallest singlar value.
     * \param[out] x       The associated approximated right-singular
     *                     vector (if #rows >= #ncols) respectively
     *                     left-singular vector (if #rows < #ncols).
     */
    inline void computeNonSymMin (const Real& epsilon,
                                  BlockVector& x, Real& sigma) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_ << "Computing an approximation of the "
                  << "smallest singular value of a matrix which "
                  << "is assumed to be nonsymmetric." << std::endl;

      // use type ArPackPlusPlus_BCRSMatrixWrapper to store matrix information
      // and to perform the product A^T*A*v (LU decomposition is not used)
      typedef Impl::ArPackPlusPlus_BCRSMatrixWrapper<BCRSMatrix> WrappedMatrix;
      WrappedMatrix A(m_);

      // get number of rows and columns in A
      const int nrows = A.nrows();
      const int ncols = A.ncols();

      // assert that A has more rows than columns (extend code later to the opposite case!)
      if (nrows < ncols)
        DUNE_THROW(Dune::ISTLError,"Matrix has less rows than "
                   << "columns (" << nrows << "x" << ncols << ")."
                   << " This case is not implemented, yet.");

      // allocate memory for variables, set parameters
      const int nev = 1;                     // Number of eigenvalues to compute
      int ncv = std::min(20, nrows);         // Number of Arnoldi vectors generated at each iteration (0 == auto)
      const Real tol = epsilon;              // Stopping tolerance (relative accuracy of Ritz values) (0 == machine precision)
      const int maxit = nIterationsMax_*nev; // Maximum number of Arnoldi update iterations allowed   (0 == 100*nev)
      Real* ev = new Real[nev];              // Computed eigenvalues of A^T*A
      const bool ivec = true;                // Flag deciding if eigenvectors shall be determined
      int nconv;                             // Number of converged eigenvalues

      // define what we need: eigenvalues with smallest algebraic value
      char which[] = "SA";
      ARSymStdEig<Real,WrappedMatrix>
        dprob(ncols, nev, &A, &WrappedMatrix::multMtMv, which, ncv, tol, maxit);

      // set ARPACK verbosity mode if requested
      if (verbosity_level_ > 3) dprob.Trace();

      // find eigenvalues and eigenvectors of A^T*A, obtain the eigenvalues
      nconv = dprob.Eigenvalues(ev,ivec);

      // obtain approximated smallest eigenvalue of A^T*A
      const Real& lambda = ev[nev-1];

      // obtain associated approximated eigenvector of A^T*A
      Real* x_raw = dprob.RawEigenvector(nev-1);
      WrappedMatrix::arrayToDomainBlockVector(x_raw,x);

      // obtain number of Arnoldi update iterations actually taken
      nIterations_ = dprob.GetIter();

      // compute residual norm
      BlockVector r(x);
      Real* AtAx_raw = new Real[ncols];
      A.multMtMv(x_raw,AtAx_raw);
      WrappedMatrix::arrayToDomainBlockVector(AtAx_raw,r);
      r.axpy(-lambda,x);
      const Real r_norm = r.two_norm();

      // calculate smallest singular value of A (note that
      // x is right-singular / left-singular vector of A)
      sigma = std::sqrt(lambda);

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        if (verbosity_level_ > 1)
        {
          // print some information about the problem
          std::cout << blank_ << "Obtained singular values of A by sol"
                    << "ving (A^T*A)*x = σ²*x using the ARPACK++ "
                    << "class ARSymStdEig:" << std::endl;
          std::cout << blank_ << "   converged eigenvalues of A^T*A: "
                    << nconv << " / " << nev << std::endl;
          std::cout << blank_ << "     smallest eigenvalue of A^T*A: "
                    << lambda << std::endl;
          std::cout << blank_ << "  => smallest singular value of A: "
                    << sigma << std::endl;
        }
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = " << r_norm << "): "
                  << "σ = " << sigma << std::endl;
        if (verbosity_level_ > 2)
        {
          // print approximated right-singular / left-singular vector
          // via DUNE-ISTL I/O methods
          Dune::printvector(std::cout,x,blank_+"x",blank_+"row");
        }
      }

      // free dynamically allocated memory
      delete[] AtAx_raw;
      delete[] ev;
    }

    /**
     * \brief Assume the matrix to be nonsymmetric and perform IRLM
     *        to compute an approximation of its spectral condition
     *        number which can be expressed as the ratio of the
     *        largest singular value and the smallest singular value.
     *
     * \param[in]  epsilon The target relative accuracy of Ritz values
     *                     (0 == machine precision).
     * \param[out] cond_2  The approximated spectral condition number.
     */
    inline void computeNonSymCond2 (const Real& epsilon, Real& cond_2) const
    {
      // print verbosity information
      if (verbosity_level_ > 0)
        std::cout << title_ << "Computing an approximation of the "
                  << "spectral condition number of a matrix which "
                  << "is assumed to be nonsymmetric." << std::endl;

      // use type ArPackPlusPlus_BCRSMatrixWrapper to store matrix information
      // and to perform the product A^T*A*v (LU decomposition is not used)
      typedef Impl::ArPackPlusPlus_BCRSMatrixWrapper<BCRSMatrix> WrappedMatrix;
      WrappedMatrix A(m_);

      // get number of rows and columns in A
      const int nrows = A.nrows();
      const int ncols = A.ncols();

      // assert that A has more rows than columns (extend code later to the opposite case!)
      if (nrows < ncols)
        DUNE_THROW(Dune::ISTLError,"Matrix has less rows than "
                   << "columns (" << nrows << "x" << ncols << ")."
                   << " This case is not implemented, yet.");

      // allocate memory for variables, set parameters
      const int nev = 2;                     // Number of eigenvalues to compute
      int ncv = std::min(20, nrows);         // Number of Arnoldi vectors generated at each iteration (0 == auto)
      const Real tol = epsilon;              // Stopping tolerance (relative accuracy of Ritz values) (0 == machine precision)
      const int maxit = nIterationsMax_*nev; // Maximum number of Arnoldi update iterations allowed   (0 == 100*nev)
      Real* ev = new Real[nev];              // Computed eigenvalues of A^T*A
      const bool ivec = true;                // Flag deciding if eigenvectors shall be determined
      int nconv;                             // Number of converged eigenvalues

      // define what we need: eigenvalues from both ends of the spectrum
      char which[] = "BE";
      ARSymStdEig<Real,WrappedMatrix>
        dprob(ncols, nev, &A, &WrappedMatrix::multMtMv, which, ncv, tol, maxit);

      // set ARPACK verbosity mode if requested
      if (verbosity_level_ > 3) dprob.Trace();

      // find eigenvalues and eigenvectors of A^T*A, obtain the eigenvalues
      nconv = dprob.Eigenvalues(ev,ivec);

      // obtain approximated largest and smallest eigenvalue of A^T*A
      const Real& lambda_max = ev[nev-1];
      const Real& lambda_min = ev[0];

      // obtain associated approximated eigenvectors of A^T*A
      Real* x_max_raw = dprob.RawEigenvector(nev-1);
      Real* x_min_raw = dprob.RawEigenvector(0);

      // obtain number of Arnoldi update iterations actually taken
      nIterations_ = dprob.GetIter();

      // compute each residual norm
      Real* AtAx_max_raw = new Real[ncols];
      Real* AtAx_min_raw = new Real[ncols];
      A.multMtMv(x_max_raw,AtAx_max_raw);
      A.multMtMv(x_min_raw,AtAx_min_raw);
      Real r_max_norm = 0.0;
      Real r_min_norm = 0.0;
      for (int i = 0; i < ncols; ++i)
      {
        r_max_norm += std::pow(AtAx_max_raw[i] - lambda_max * x_max_raw[i],2);
        r_min_norm += std::pow(AtAx_min_raw[i] - lambda_min * x_min_raw[i],2);
      }
      r_max_norm = std::sqrt(r_max_norm);
      r_min_norm = std::sqrt(r_min_norm);

      // calculate largest and smallest singular value of A
      const Real sigma_max = std::sqrt(lambda_max);
      const Real sigma_min = std::sqrt(lambda_min);

      // obtain approximated spectral condition number of A
      cond_2 = sigma_max / sigma_min;

      // print verbosity information
      if (verbosity_level_ > 0)
      {
        if (verbosity_level_ > 1)
        {
          // print some information about the problem
          std::cout << blank_ << "Obtained singular values of A by sol"
                    << "ving (A^T*A)*x = σ²*x using the ARPACK++ "
                    << "class ARSymStdEig:" << std::endl;
          std::cout << blank_ << "   converged eigenvalues of A^T*A: "
                    << nconv << " / " << nev << std::endl;
          std::cout << blank_ << "      largest eigenvalue of A^T*A: "
                    << lambda_max << std::endl;
          std::cout << blank_ << "     smallest eigenvalue of A^T*A: "
                    << lambda_min << std::endl;
          std::cout << blank_ << "  =>  largest singular value of A: "
                    << sigma_max << std::endl;
          std::cout << blank_ << "  => smallest singular value of A: "
                    << sigma_min << std::endl;
        }
        std::cout << blank_ << "Result ("
                  << "#iterations = " << nIterations_ << ", "
                  << "║residual║_2 = {" << r_max_norm << ","
                  << r_min_norm << "}, " << "σ = {"
                  << sigma_max << "," << sigma_min
                  << "}): cond_2 = " << cond_2 << std::endl;
      }

      // free dynamically allocated memory
      delete[] AtAx_min_raw;
      delete[] AtAx_max_raw;
      delete[] ev;
    }

    /**
     * \brief Return the number of iterations in last application of
     *        an algorithm.
     */
    inline unsigned int getIterationCount () const
    {
      if (nIterations_ == 0)
        DUNE_THROW(Dune::ISTLError,"No algorithm applied, yet.");

      return nIterations_;
    }

  protected:
    // parameters related to iterative eigenvalue algorithms
    const BCRSMatrix& m_;
    const unsigned int nIterationsMax_;

    // verbosity setting
    const unsigned int verbosity_level_;

    // memory for storing temporary variables (mutable as they shall
    // just be effectless auxiliary variables of the const apply*(...)
    // methods)
    mutable unsigned int nIterations_;

    // constants for printing verbosity information
    const std::string title_;
    const std::string blank_;
  };

  /** @} */

}  // namespace Dune

#endif  // HAVE_ARPACKPP

#endif  // DUNE_ISTL_EIGENVALUE_ARPACKPP_HH
