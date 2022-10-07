// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SPQR_HH
#define DUNE_ISTL_SPQR_HH

#if HAVE_SUITESPARSE_SPQR || defined DOXYGEN

#include <complex>
#include <type_traits>

#include <SuiteSparseQR.hpp>

#include <dune/common/exceptions.hh>

#include <dune/istl/bccsmatrixinitializer.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/solvertype.hh>
#include <dune/istl/solverfactory.hh>

namespace Dune {
  /**
   * @addtogroup ISTL
   *
   * @{
   */
  /**
   * @file
   * @author Marco Agnese, Andrea Sacconi
   * @brief Class for using SPQR with ISTL matrices.
   */

  // forward declarations
  template<class M, class T, class TM, class TD, class TA>
  class SeqOverlappingSchwarz;

  template<class T, bool tag>
  struct SeqOverlappingSchwarzAssemblerHelper;

  /** @brief Use the %SPQR package to directly solve linear systems -- empty default class
   * @tparam Matrix the matrix type defining the system
   * Details on SPQR can be found on
   * http://www.cise.ufl.edu/research/sparse/spqr/
   */
  template<class Matrix>
  class SPQR
  {};

  /** @brief The %SPQR direct sparse solver for matrices of type BCRSMatrix
   *
   * Specialization for the Dune::BCRSMatrix. %SPQR will always go double
   * precision and supports complex numbers
   * too (use std::complex<double> for that).
   *
   * \tparam T Number type.  Only double and std::complex<double> is supported
   * \tparam A STL-compatible allocator type
   * \tparam n Number of rows in a matrix block
   * \tparam m Number of columns in a matrix block
   *
   * \note This will only work if dune-istl has been configured to use SPQR
   */
  template<typename T, typename A, int n, int m>
  class SPQR<BCRSMatrix<FieldMatrix<T,n,m>,A > >
    : public InverseOperator<BlockVector<FieldVector<T,m>, typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T,m> > >,
                             BlockVector<FieldVector<T,n>, typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T,n> > > >
  {
    public:
    /** @brief The matrix type. */
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
    typedef Dune::BCRSMatrix<FieldMatrix<T,n,m>,A> matrix_type;
    /** @brief The corresponding SuperLU Matrix type.*/
    typedef ISTL::Impl::BCCSMatrix<T,int> SPQRMatrix;
    /** @brief Type of an associated initializer class. */
    typedef ISTL::Impl::BCCSMatrixInitializer<BCRSMatrix<FieldMatrix<T,n,m>,A>, int> MatrixInitializer;
    /** @brief The type of the domain of the solver. */
    typedef Dune::BlockVector<FieldVector<T,m>, typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T,m> > > domain_type;
    /** @brief The type of the range of the solver. */
    typedef Dune::BlockVector<FieldVector<T,n>, typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T,n> > > range_type;

    //! Category of the solver (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::Category::sequential;
    }

    /** @brief Construct a solver object from a BCRSMatrix
     *
     * This computes the matrix decomposition, and may take a long time
     * (and use a lot of memory).
     *
     *  @param matrix the matrix to solve for
     *  @param verbose, 0 or 1, set the verbosity level, defaults to 0
     */
    SPQR(const Matrix& matrix, int verbose=0) : matrixIsLoaded_(false), verbose_(verbose)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in SPQR (only double and std::complex<double> supported)");
      cc_ = new cholmod_common();
      cholmod_l_start(cc_);
      setMatrix(matrix);
    }

     /** @brief Constructor for compatibility with SuperLU standard constructor
     *
     * This computes the matrix decomposition, and may take a long time
     * (and use a lot of memory).
     *
     * @param matrix the matrix to solve for
     * @param verbose, 0 or 1, set the verbosity level, defaults to 0
     */
    SPQR(const Matrix& matrix, int verbose, bool) : matrixIsLoaded_(false), verbose_(verbose)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in SPQR (only double and std::complex<double> supported)");
      cc_ = new cholmod_common();
      cholmod_l_start(cc_);
      setMatrix(matrix);
    }

    /** @brief Constructs the SPQR solver.
     *
     * @param matrix  The matrix of the system to solve.
     * @param config  ParameterTree containing solver parameters.
     *
     * ParameterTree Key | Meaning
     * ------------------|------------
     * verbose           | The verbosity level. default=0
    */
    SPQR(const Matrix& matrix, const ParameterTree& config)
      : SPQR(matrix, config.get<int>("verbose", 0))
    {}

    /** @brief Default constructor. */
    SPQR() : matrixIsLoaded_(false), verbose_(0)
    {
      //check whether T is a supported type
      static_assert((std::is_same<T,double>::value) || (std::is_same<T,std::complex<double> >::value),
                    "Unsupported Type in SPQR (only double and std::complex<double> supported)");
      cc_ = new cholmod_common();
      cholmod_l_start(cc_);
    }

    /** @brief Destructor. */
    virtual ~SPQR()
    {
      if ((spqrMatrix_.N() + spqrMatrix_.M() > 0) || matrixIsLoaded_)
        free();
      cholmod_l_finish(cc_);
    }

    /** \copydoc InverseOperator::apply(X&, Y&, InverseOperatorResult&) */
    virtual void apply(domain_type& x, range_type& b, InverseOperatorResult& res)
    {
      const std::size_t numRows(spqrMatrix_.N());
      // fill B
      for(std::size_t k = 0; k != numRows/n; ++k)
        for (int l = 0; l < n; ++l)
          (static_cast<T*>(B_->x))[n*k+l] = b[k][l];

      cholmod_dense* BTemp = B_;
      B_ = SuiteSparseQR_qmult<T>(0, spqrfactorization_, B_, cc_);
      cholmod_dense* X = SuiteSparseQR_solve<T>(1, spqrfactorization_, B_, cc_);
      cholmod_l_free_dense(&BTemp, cc_);

      const std::size_t numCols(spqrMatrix_.M());
      // fill x
      for(std::size_t k = 0; k != numCols/m; ++k)
        for (int l = 0; l < m; ++l)
          x[k][l] = (static_cast<T*>(X->x))[m*k+l];

      cholmod_l_free_dense(&X, cc_);
      // this is a direct solver
      res.iterations = 1;
      res.converged = true;
      if(verbose_ > 0)
      {
        std::cout<<std::endl<<"Solving with SuiteSparseQR"<<std::endl;
        std::cout<<"Flops Taken: "<<cc_->SPQR_flopcount<<std::endl;
        std::cout<<"Analysis Time: "<<cc_->SPQR_analyze_time<<" s"<<std::endl;
        std::cout<<"Factorize Time: "<<cc_->SPQR_factorize_time<<" s"<<std::endl;
        std::cout<<"Backsolve Time: "<<cc_->SPQR_solve_time<<" s"<<std::endl;
        std::cout<<"Peak Memory Usage: "<<cc_->memory_usage<<" bytes"<<std::endl;
        std::cout<<"Rank Estimate: "<<cc_->SPQR_istat[4]<<std::endl<<std::endl;
      }
    }

    /** \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&) */
    virtual void apply (domain_type& x, range_type& b, [[maybe_unused]] double reduction, InverseOperatorResult& res)
    {
      apply(x, b, res);
    }

    void setOption([[maybe_unused]] unsigned int option, [[maybe_unused]] double value)
    {}

    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& matrix)
    {
      if ((spqrMatrix_.N() + spqrMatrix_.M() > 0) || matrixIsLoaded_)
        free();

      if (spqrMatrix_.N() + spqrMatrix_.M() + spqrMatrix_.nonzeroes() != 0)
        spqrMatrix_.free();
      spqrMatrix_.setSize(MatrixDimension<Matrix>::rowdim(matrix),
                          MatrixDimension<Matrix>::coldim(matrix));
      ISTL::Impl::BCCSMatrixInitializer<Matrix, int> initializer(spqrMatrix_);

      copyToBCCSMatrix(initializer, matrix);

      decompose();
    }

    template<class S>
    void setSubMatrix(const Matrix& matrix, const S& rowIndexSet)
    {
      if ((spqrMatrix_.N() + spqrMatrix_.M() > 0) || matrixIsLoaded_)
        free();

      if (spqrMatrix_.N() + spqrMatrix_.M() + spqrMatrix_.nonzeroes() != 0)
        spqrMatrix_.free();

      spqrMatrix_.setSize(rowIndexSet.size()*MatrixDimension<Matrix>::rowdim(matrix) / matrix.N(),
                          rowIndexSet.size()*MatrixDimension<Matrix>::coldim(matrix) / matrix.M());
      ISTL::Impl::BCCSMatrixInitializer<Matrix, int> initializer(spqrMatrix_);

      copyToBCCSMatrix(initializer, ISTL::Impl::MatrixRowSubset<Matrix,std::set<std::size_t> >(matrix,rowIndexSet));

      decompose();
    }

    /**
     * @brief Sets the verbosity level for the solver.
     * @param v verbosity level: 0 only error messages, 1 a bit of statistics.
     */
    inline void setVerbosity(int v)
    {
      verbose_=v;
    }

    /**
     * @brief Return the matrix factorization.
     * @warning It is up to the user to keep consistency.
     */
    inline SuiteSparseQR_factorization<T>* getFactorization()
    {
      return spqrfactorization_;
    }

    /**
     * @brief Return the column coppressed matrix.
     * @warning It is up to the user to keep consistency.
     */
    inline SPQRMatrix& getInternalMatrix()
    {
      return spqrMatrix_;
    }

    /**
     * @brief Free allocated space.
     * @warning Later calling apply will result in an error.
     */
    void free()
    {
      cholmod_l_free_sparse(&A_, cc_);
      cholmod_l_free_dense(&B_, cc_);
      SuiteSparseQR_free<T>(&spqrfactorization_, cc_);
      spqrMatrix_.free();
      matrixIsLoaded_ = false;
    }

    /** @brief Get method name. */
    inline const char* name()
    {
      return "SPQR";
    }

    private:
    template<class M,class X, class TM, class TD, class T1>
    friend class SeqOverlappingSchwarz;

    friend struct SeqOverlappingSchwarzAssemblerHelper<SPQR<Matrix>,true>;

    /** @brief Computes the QR decomposition. */
    void decompose()
    {
      const std::size_t nrows(spqrMatrix_.N());
      const std::size_t ncols(spqrMatrix_.M());
      const std::size_t nnz(spqrMatrix_.getColStart()[ncols]);

      // initialise the matrix A (sorted, packed, unsymmetric, real entries)
      A_ = cholmod_l_allocate_sparse(nrows, ncols, nnz, 1, 1, 0, 1, cc_);

      // copy all the entries of Ap, Ai, Ax
      for(std::size_t k = 0; k != (ncols+1); ++k)
        (static_cast<long int *>(A_->p))[k] = spqrMatrix_.getColStart()[k];

      for(std::size_t k = 0; k != nnz; ++k)
      {
        (static_cast<long int*>(A_->i))[k] = spqrMatrix_.getRowIndex()[k];
        (static_cast<T*>(A_->x))[k] = spqrMatrix_.getValues()[k];
      }

      // initialise the vector B
      B_ = cholmod_l_allocate_dense(nrows, 1, nrows, A_->xtype, cc_);
      // compute factorization of A
      spqrfactorization_=SuiteSparseQR_factorize<T>(SPQR_ORDERING_DEFAULT,SPQR_DEFAULT_TOL,A_,cc_);
    }

    SPQRMatrix spqrMatrix_;
    bool matrixIsLoaded_;
    int verbose_;
    cholmod_common* cc_;
    cholmod_sparse* A_;
    cholmod_dense* B_;
    SuiteSparseQR_factorization<T>* spqrfactorization_;
  };

  template<typename T, typename A>
  struct IsDirectSolver<SPQR<BCRSMatrix<T,A> > >
  {
    enum {value = true};
  };

  template<typename T, typename A>
  struct StoresColumnCompressed<SPQR<BCRSMatrix<T,A> > >
  {
    enum {value = true};
  };

  struct SPQRCreator {
    template<class> struct isValidBlock : std::false_type{};

    template<typename TL, typename M>
    std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                          typename Dune::TypeListElement<2, TL>::type>>
    operator() (TL /*tl*/, const M& mat, const Dune::ParameterTree& config,
      std::enable_if_t<
                isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value,int> = 0) const
    {
      int verbose = config.get("verbose", 0);
      return std::make_shared<Dune::SPQR<M>>(mat,verbose);
    }

    // second version with SFINAE to validate the template parameters of SPQR
    template<typename TL, typename M>
    std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                          typename Dune::TypeListElement<2, TL>::type>>
    operator() (TL /*tl*/, const M& /*mat*/, const Dune::ParameterTree& /*config*/,
      std::enable_if_t<!isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value,int> = 0) const
    {
      DUNE_THROW(UnsupportedType,
        "Unsupported Type in SPQR (only double and std::complex<double> supported)");
    }
  };
  template<> struct SPQRCreator::isValidBlock<FieldVector<double,1>> : std::true_type{};
  // std::complex is temporary disabled, because it fails if libc++ is used
  //template<> struct SPQRCreator::isValidMatrixBlock<FieldMatrix<std::complex<double>,1,1>> : std::true_type{};
  DUNE_REGISTER_DIRECT_SOLVER("spqr", Dune::SPQRCreator());

} // end namespace Dune


#endif //HAVE_SUITESPARSE_SPQR
#endif //DUNE_ISTL_SPQR_HH
