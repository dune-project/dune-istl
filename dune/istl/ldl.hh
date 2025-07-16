// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_LDL_HH
#define DUNE_ISTL_LDL_HH

#if HAVE_SUITESPARSE_LDL || defined DOXYGEN

#include <memory>
#include <type_traits>

#ifdef __cplusplus
extern "C"
{
#include <ldl.h>
#include <amd.h>
}
#endif

#include <dune/common/exceptions.hh>

#include <dune/istl/bccsmatrixinitializer.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/matrixtraits.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/solvertype.hh>
#include <dune/istl/solverregistry.hh>

namespace Dune {
  /**
   * @addtogroup ISTL
   *
   * @{
   */
  /**
   * @file
   * @author Marco Agnese, Andrea Sacconi
   * @brief Class for using LDL with ISTL matrices.
   */

  // forward declarations
  template<class M, class T, class TM, class TD, class TA>
  class SeqOverlappingSchwarz;

  template<class T, bool tag>
  struct SeqOverlappingSchwarzAssemblerHelper;

  /**
   * @brief The %LDL direct sparse solver for ISTL matrices
   *
   * %LDL will always go double precision.
   *
   * Details on UMFPack can be found on
   * https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/LDL
   *
   * @tparam M  the matrix type defining the system
   *
   * \note This will only work if dune-istl has been configured to use LDL
   */
  template<typename M>
  class LDL : public InverseOperator<
    typename MatrixTraits<M>::domain_type,
    typename MatrixTraits<M>::range_type>
  {
    public:
    /** @brief The matrix type. */
    typedef M Matrix;
    typedef M matrix_type;
    typedef typename FieldTraits<M>::field_type field_type;

    /** @brief The corresponding SuperLU Matrix type. */
    typedef Dune::ISTL::Impl::BCCSMatrix<field_type,int> LDLMatrix;
    /** @brief Type of an associated initializer class. */
    typedef ISTL::Impl::BCCSMatrixInitializer<M, int> MatrixInitializer;
    /** @brief The type of the domain of the solver. */
    typedef typename MatrixTraits<M>::domain_type domain_type;
    /** @brief The type of the range of the solver. */
    typedef typename MatrixTraits<M>::range_type range_type;

    //! Category of the solver (see SolverCategory::Category)
    SolverCategory::Category category() const override
    {
      return SolverCategory::Category::sequential;
    }

    /**
     * @brief Construct a solver object from a BCRSMatrix.
     *
     * This computes the matrix decomposition, and may take a long time
     * (and use a lot of memory).
     *
     * @param matrix the matrix to solve for
     * @param verbose 0 or 1 set the verbosity level, defaults to 0
     */
    LDL(const Matrix& matrix, int verbose=0) : matrixIsLoaded_(false), verbose_(verbose)
    {
      //check whether T is a supported type
      static_assert(std::is_same<field_type,double>::value,"Unsupported Type in LDL (only double supported)");
      setMatrix(matrix);
    }

    /**
     * @brief Constructor for compatibility with SuperLU standard constructor
     *
     * This computes the matrix decomposition, and may take a long time
     * (and use a lot of memory).
     *
     * @param matrix the matrix to solve for
     * @param verbose 0 or 1 set the verbosity level, defaults to 0
     */
    LDL(const Matrix& matrix, int verbose, bool) : matrixIsLoaded_(false), verbose_(verbose)
    {
      //check whether T is a supported type
      static_assert(std::is_same<field_type,double>::value,"Unsupported Type in LDL (only double supported)");
      setMatrix(matrix);
    }

    /** @brief Constructs the LDL solver.
     *
     * @param matrix  The matrix of the system to solve.
     * @param config  ParameterTree containing solver parameters.
     *
     * ParameterTree Key | Meaning
     * ------------------|------------
     * verbose           | The verbosity level. default=0
    */
    LDL(const Matrix& matrix, const ParameterTree& config)
      : LDL(matrix, config.get<int>("verbose", 0))
    {}

    /** @brief Default constructor. */
    LDL() : matrixIsLoaded_(false), verbose_(0)
    {}

    /** @brief Default constructor. */
    virtual ~LDL()
    {
      if ((ldlMatrix_.N() + ldlMatrix_.M() > 0) || matrixIsLoaded_)
        free();
    }

  public:
    /** \copydoc InverseOperator::apply(X&, Y&, InverseOperatorResult&) */
    void apply(domain_type& x, range_type& b, InverseOperatorResult& res) override
    {
      BlockVector<field_type> B(ldlMatrix_.N()), X(ldlMatrix_.M());
      flatVectorForEachMasked(b, maskVector_, [&](auto const& b_i, std::size_t i) { B[i] = b_i; });
      apply(X.data(), B.data());
      flatVectorForEachMasked(x, maskVector_, [&](auto& x_i, std::size_t i) { x_i = X[i]; });

      // this is a direct solver
      res.iterations = 1;
      res.converged = true;
    }

    /** \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&) */
    void apply(domain_type& x, range_type& b, [[maybe_unused]] double reduction, InverseOperatorResult& res) override
    {
      apply(x,b,res);
    }

    /**
     * @brief Additional apply method with c-arrays in analogy to superlu.
     * @param x solution array
     * @param b rhs array
     */
    void apply(field_type* x, field_type* b)
    {
      const int dimMat(ldlMatrix_.N());
      ldl_perm(dimMat, Y_, b, P_);
      ldl_lsolve(dimMat, Y_, Lp_, Li_, Lx_);
      ldl_dsolve(dimMat, Y_, D_);
      ldl_ltsolve(dimMat, Y_, Lp_, Li_, Lx_);
      ldl_permt(dimMat, x, Y_, P_);
    }

    void setOption([[maybe_unused]] unsigned int option, [[maybe_unused]] double value)
    {}

    /** @brief Initialize data from given matrix. */
    void setMatrix(const Matrix& matrix)
    {
      if ((ldlMatrix_.N() + ldlMatrix_.M() > 0) || matrixIsLoaded_)
        free();

      if (ldlMatrix_.N() + ldlMatrix_.M() + ldlMatrix_.nonzeroes() != 0)
        ldlMatrix_.free();
      ldlMatrix_.setSize(MatrixDimension<Matrix>::rowdim(matrix),
                         MatrixDimension<Matrix>::coldim(matrix));
      MatrixInitializer initializer(ldlMatrix_);

      copyToBCCSMatrix(initializer, matrix);

      maskVector_.clear();
      maskVector_.resize(matrix.N(), true);

      decompose();
    }

    template<class S>
    void setSubMatrix(const Matrix& matrix, const S& rowIndexSet)
    {
      if ((ldlMatrix_.N() + ldlMatrix_.M() > 0) || matrixIsLoaded_)
        free();

      if (ldlMatrix_.N() + ldlMatrix_.M() + ldlMatrix_.nonzeroes() != 0)
        ldlMatrix_.free();

      ldlMatrix_.setSize(rowIndexSet.size()*MatrixDimension<Matrix>::rowdim(matrix) / matrix.N(),
                         rowIndexSet.size()*MatrixDimension<Matrix>::coldim(matrix) / matrix.M());
      MatrixInitializer initializer(ldlMatrix_);

      copyToBCCSMatrix(initializer, ISTL::Impl::MatrixRowSubset<Matrix,std::set<std::size_t> >(matrix,rowIndexSet));

      maskVector_.clear();
      maskVector_.resize(matrix.N(), false);
      for (auto i : rowIndexSet)
        maskVector_[i] = true;

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
     * @brief Return the column compress matrix.
     * @warning It is up to the user to keep consistency.
     */
    inline LDLMatrix& getInternalMatrix()
    {
      return ldlMatrix_;
    }

    /**
     * @brief Free allocated space.
     * @warning Later calling apply will result in an error.
     */
    void free()
    {
      delete [] D_;
      delete [] Y_;
      delete [] Lp_;
      delete [] Lx_;
      delete [] Li_;
      delete [] P_;
      delete [] Pinv_;
      ldlMatrix_.free();
      matrixIsLoaded_ = false;
    }

    /** @brief Get method name. */
    inline const char* name()
    {
      return "LDL";
    }

    /**
     * @brief Get factorization diagonal matrix D.
     * @warning It is up to the user to preserve consistency.
     */
    inline double* getD()
    {
      return D_;
    }

    /**
     * @brief Get factorization Lp.
     * @warning It is up to the user to preserve consistency.
     */
    inline int* getLp()
    {
      return Lp_;
    }

    /**
     * @brief Get factorization Li.
     * @warning It is up to the user to preserve consistency.
     */
    inline int* getLi()
    {
      return Li_;
    }

    /**
     * @brief Get factorization Lx.
     * @warning It is up to the user to preserve consistency.
     */
    inline double* getLx()
    {
      return Lx_;
    }

    private:
    template<class,class,class,class,class>
    friend class SeqOverlappingSchwarz;

    friend struct SeqOverlappingSchwarzAssemblerHelper<LDL<Matrix>,true>;

    /** @brief Computes the LDL decomposition. */
    void decompose()
    {
      // allocate vectors
      const int dimMat(ldlMatrix_.N());
      D_ = new double [dimMat];
      Y_ = new double [dimMat];
      Lp_ = new int [dimMat + 1];
      Parent_ = new int [dimMat];
      Lnz_ = new int [dimMat];
      Flag_ = new int [dimMat];
      Pattern_ = new int [dimMat];
      P_ = new int [dimMat];
      Pinv_ = new int [dimMat];

      double Info [AMD_INFO];
      if(amd_order (dimMat, ldlMatrix_.getColStart(), ldlMatrix_.getRowIndex(), P_, (double *) NULL, Info) < AMD_OK)
        DUNE_THROW(InvalidStateException,"Error: AMD failed!");
      if(verbose_ > 0)
        amd_info (Info);
      // compute the symbolic factorisation
      ldl_symbolic(dimMat, ldlMatrix_.getColStart(), ldlMatrix_.getRowIndex(), Lp_, Parent_, Lnz_, Flag_, P_, Pinv_);
      // initialise those entries of additionalVectors_ whose dimension is known only now
      Lx_ = new double [Lp_[dimMat]];
      Li_ = new int [Lp_[dimMat]];
      // compute the numeric factorisation
      const int rank(ldl_numeric(dimMat, ldlMatrix_.getColStart(), ldlMatrix_.getRowIndex(), ldlMatrix_.getValues(),
                                 Lp_, Parent_, Lnz_, Li_, Lx_, D_, Y_, Pattern_, Flag_, P_, Pinv_));
      // free temporary vectors
      delete [] Flag_;
      delete [] Pattern_;
      delete [] Parent_;
      delete [] Lnz_;

      if(rank!=dimMat)
        DUNE_THROW(InvalidStateException,"Error: LDL factorisation failed!");
    }

    LDLMatrix ldlMatrix_;
    bool matrixIsLoaded_;
    int verbose_;
    int* Lp_;
    int* Parent_;
    int* Lnz_;
    int* Flag_;
    int* Pattern_;
    int* P_;
    int* Pinv_;
    double* D_;
    double* Y_;
    double* Lx_;
    int* Li_;

    std::vector<bool> maskVector_;
  };

  template<typename M>
  struct IsDirectSolver<LDL<M> >
  {
    enum {value = true};
  };

  template<typename T, typename A>
  struct StoresColumnCompressed<LDL<BCRSMatrix<T,A> > >
  {
    enum {value = true};
  };

  DUNE_REGISTER_SOLVER("ldl",
                       [](auto opTraits, const auto& op, const Dune::ParameterTree& config)
                       -> std::shared_ptr<typename decltype(opTraits)::solver_type>
                       {
                         using OpTraits = decltype(opTraits);
                         using M = typename OpTraits::matrix_type;
                         // works only for sequential operators
                         if constexpr (OpTraits::isParallel){
                           if(opTraits.getCommOrThrow(op).communicator().size() > 1)
                             DUNE_THROW(Dune::InvalidStateException, "LDL works only for sequential operators.");
                         }
                         // check if LDL<M>* is convertible to
                         // InverseOperator*. This allows only the explicit
                         // specialized variants of LDL
                         if constexpr (std::is_convertible_v<LDL<M>*,
                                       Dune::InverseOperator<typename OpTraits::domain_type,
                                       typename OpTraits::range_type>*> &&
                                       std::is_same_v<typename FieldTraits<M>::field_type, double>
                                       ){
                           const auto& A = opTraits.getAssembledOpOrThrow(op);
                           const M& mat = A->getmat();
                           int verbose = config.get("verbose", 0);
                           return std::make_shared<LDL<M>>(mat,verbose);
                         }
                         DUNE_THROW(UnsupportedType,
                                    "Unsupported Type in LDL (only FieldMatrix<double,...> supported)");
                       });

} // end namespace Dune


#endif //HAVE_SUITESPARSE_LDL
#endif //DUNE_ISTL_LDL_HH
