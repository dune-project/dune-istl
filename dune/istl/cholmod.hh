// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#pragma once

#if HAVE_SUITESPARSE_CHOLMOD

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include<dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/foreach.hh>

#include <vector>
#include <memory>

#include <cholmod.h>

namespace Dune {

namespace Impl{

  /** @brief Dummy class for empty ignore nodes
   *
   * This class implements "no" ignore nodes with a
   * compatible interface for the "setMatrix" method.
   *
   * It should be optimized out by the compiler, so no
   * overhead should be produced
   */
  struct NoIgnore
  {
    const NoIgnore& operator[](std::size_t) const { return *this; }
    explicit operator bool() const { return false; }
    static constexpr std::size_t size() { return 0; }

  };


  template<class BlockedVector, class FlatVector>
  void copyToFlatVector(const BlockedVector& blockedVector, FlatVector& flatVector)
  {
    // traverse the vector once just to compute the size
    std::size_t len = flatVectorForEach(blockedVector, [&](auto&&, auto...){});
    flatVector.resize(len);

    flatVectorForEach(blockedVector, [&](auto&& entry, auto offset){
      flatVector[offset] = entry;
    });
  }

  // special (dummy) case for NoIgnore
  template<class FlatVector>
  void copyToFlatVector(const NoIgnore&, FlatVector&)
  {
    // just do nothing
    return;
  }

  template<class FlatVector, class BlockedVector>
  void copyToBlockedVector(const FlatVector& flatVector, BlockedVector& blockedVector)
  {
    flatVectorForEach(blockedVector, [&](auto& entry, auto offset){
      entry = flatVector[offset];
    });
  }


} //namespace Impl

/** @brief Dune wrapper for SuiteSparse/CHOLMOD solver
  *
  * This class implements an InverseOperator between Vector types
  */
template<class Vector>
class Cholmod : public InverseOperator<Vector, Vector>
{
public:

  /** @brief Default constructor
   *
   *  Calls the the cholmod runtime,
   *  see CHOLMOD doc
   */
  Cholmod()
  {
    cholmod_start(&c_);
  }

  /** @brief Destructor
   *
   *  Free space and calls cholmod_finish,
   *  see CHOLMOD doc
   */
  ~Cholmod()
  {
    if (L_)
      cholmod_free_factor(&L_, &c_);
    cholmod_finish(&c_);
  }

  // forbid copying to avoid freeing memory twice
  Cholmod(const Cholmod&) = delete;
  Cholmod& operator=(const Cholmod&) = delete;


  /** @brief simple forward to apply(X&, Y&, InverseOperatorResult&)
    */
  void apply (Vector& x, Vector& b, [[maybe_unused]] double reduction, InverseOperatorResult& res)
  {
    apply(x,b,res);
  }

  /** @brief solve the linear system Ax=b (possibly with respect to some ignore field)
   *
   * The method assumes that setMatrix() was called before
   * In the case of a given ignore field the corresponding entries of both in x and b will stay untouched in this method.
  */
  void apply(Vector& x, Vector& b, InverseOperatorResult& res)
  {
    // do nothing if N=0
    if ( nIsZero_ )
    {
        return;
    }

    if (x.size() != b.size())
      DUNE_THROW(Exception, "Error in apply(): sizes of x and b do not match!");

    // cast to double array
    auto b2 = std::make_unique<double[]>(L_->n);
    auto x2 = std::make_unique<double[]>(L_->n);

    // copy to cholmod
    auto bp = b2.get();

    flatVectorForEach(b, [&](auto&& entry, auto&& flatIndex){
      if ( subIndices_.empty() )
        bp[ flatIndex ] = entry;
      else
        if( subIndices_[ flatIndex ] != std::numeric_limits<std::size_t>::max() )
          bp[ subIndices_[ flatIndex ] ] = entry;
    });

      // create a cholmod dense object
    auto b3 = make_cholmod_dense(cholmod_allocate_dense(L_->n, 1, L_->n, CHOLMOD_REAL, &c_), &c_);
    // cast because void-ptr
    auto b4 = static_cast<double*>(b3->x);
    std::copy(b2.get(), b2.get() + L_->n, b4);

    // solve for a cholmod x object
    auto x3 = make_cholmod_dense(cholmod_solve(CHOLMOD_A, L_, b3.get(), &c_), &c_);
    // cast because void-ptr
    auto xp = static_cast<double*>(x3->x);

    // copy into x
    flatVectorForEach(x, [&](auto&& entry, auto&& flatIndex){
      if ( subIndices_.empty() )
        entry = xp[ flatIndex ];
      else
        if( subIndices_[ flatIndex ] != std::numeric_limits<std::size_t>::max() )
          entry = xp[ subIndices_[ flatIndex ] ];
    });

    // statistics for a direct solver
    res.iterations = 1;
    res.converged = true;
  }


  /** @brief Set matrix without ignore nodes
   *
   * This method forwards a nullptr to the setMatrix method
   * with ignore nodes
   */
  template<class Matrix>
  void setMatrix(const Matrix& matrix)
  {
    const Impl::NoIgnore* noIgnore = nullptr;
    setMatrix(matrix, noIgnore);
  }

  /** @brief Set matrix and ignore nodes
   *
   * The input matrix is copied to CHOLMOD compatible form.
   * It is possible to ignore some degrees of freedom, provided an ignore field is given with same block structure
   * like the BlockVector template of the class.
   *
   * The ignore field causes the method to ignore both rows and cols of the matrix and therefore operates only
   * on the reduced quadratic matrix. In case of, e.g., Dirichlet values the user has to take care of proper
   * adjusting of the rhs before calling apply().
   *
   * Decomposing the matrix at the end takes a lot of time
   * \param [in] matrix Matrix to be decomposed. In BCRS compatible form
   * \param [in] ignore Pointer to a compatible BitVector
   */
  template<class Matrix, class Ignore>
  void setMatrix(const Matrix& matrix, const Ignore* ignore)
  {
    // count the number of entries and diagonal entries
    int nonZeros = 0;
    int numberOfIgnoredDofs = 0;


    auto [flatRows,flatCols] = flatMatrixForEach( matrix, [&](auto&& /*entry*/, auto&& flatRowIndex, auto&& flatColIndex){
      if( flatRowIndex <= flatColIndex )
        nonZeros++;
    });

    std::vector<bool> flatIgnore;

    if ( ignore )
    {
      Impl::copyToFlatVector(*ignore,flatIgnore);
      numberOfIgnoredDofs = std::count(flatIgnore.begin(),flatIgnore.end(),true);
    }

    // Total number of rows
    int N = flatRows - numberOfIgnoredDofs;

    nIsZero_ = (N <= 0);

    if ( nIsZero_ )
    {
        return;
    }

    /*
    * CHOLMOD uses compressed-column sparse matrices, but for symmetric
    * matrices this is the same as the compressed-row sparse matrix used
    * by DUNE.  So we can just store Mᵀ instead of M (as M = Mᵀ).
    */
    const auto deleter = [c = &this->c_](auto* p) {
      cholmod_free_sparse(&p, c);
    };
    auto M = std::unique_ptr<cholmod_sparse, decltype(deleter)>(
      cholmod_allocate_sparse(N,             // # rows
                              N,             // # cols
                              nonZeros,      // # of nonzeroes
                              1,             // indices are sorted ( 1 = true)
                              1,             // matrix is "packed" ( 1 = true)
                              -1,            // stype of matrix ( -1 = consider the lower part only )
                              CHOLMOD_REAL,  // xtype of matrix ( CHOLMOD_REAL = single array, no complex numbers)
                              &c_            // cholmod_common ptr
                             ), deleter);

    // copy the data of BCRS matrix to Cholmod Sparse matrix
    int* Ap = static_cast<int*>(M->p);
    int* Ai = static_cast<int*>(M->i);
    double* Ax = static_cast<double*>(M->x);


    if ( ignore )
    {
      // init the mapping
      subIndices_.resize(flatRows,std::numeric_limits<std::size_t>::max());

      std::size_t subIndexCounter = 0;

      for ( std::size_t i=0; i<flatRows; i++ )
      {
        if ( not  flatIgnore[ i ] )
        {
          subIndices_[ i ] = subIndexCounter++;
        }
      }
    }

    // at first, we need to compute the row starts "Ap"
    // therefore, we count all (not ignored) entries in each row and in the end we accumulate everything
    flatMatrixForEach(matrix, [&](auto&& /*entry*/, auto&& flatRowIndex, auto&& flatColIndex){

      // stop if ignored
      if ( ignore and ( flatIgnore[flatRowIndex] or flatIgnore[flatColIndex] ) )
        return;

      // stop if in lower half
      if ( flatRowIndex > flatColIndex )
        return;

      // ok, count the entry
      auto idx = ignore ? subIndices_[flatRowIndex] : flatRowIndex;
      Ap[idx+1]++;

    });

    // now accumulate
    Ap[0] = 0;
    for ( int i=0; i<N; i++ )
    {
      Ap[i+1] += Ap[i];
    }

    // we need a compressed row position counter
    std::vector<std::size_t> rowPosition(N,0);

    // now we can set the entries
    flatMatrixForEach(matrix, [&](auto&& entry, auto&& flatRowIndex, auto&& flatColIndex){

      // stop if ignored
      if ( ignore and ( flatIgnore[flatRowIndex] or flatIgnore[flatColIndex] ) )
        return;

      // stop if in lower half
      if ( flatRowIndex > flatColIndex )
        return;

      // ok, set the entry
      auto rowIdx = ignore ? subIndices_[flatRowIndex] : flatRowIndex;
      auto colIdx = ignore ? subIndices_[flatColIndex] : flatColIndex;
      auto rowStart = Ap[rowIdx];
      auto rowPos   = rowPosition[rowIdx];
      Ai[ rowStart + rowPos ] = colIdx;
      Ax[ rowStart + rowPos ] = entry;
      rowPosition[rowIdx]++;

    });

    // Now analyse the pattern and optimal row order
    L_ = cholmod_analyze(M.get(), &c_);

    // Do the factorization (this may take some time)
    cholmod_factorize(M.get(), L_, &c_);
  }

  virtual SolverCategory::Category category() const
  {
    return SolverCategory::Category::sequential;
  }

  /** \brief return a reference to the CHOLMOD common object for advanced option settings
   *
   *  The CHOLMOD common object stores all parameters and options for the solver to run
   *  and can be modified in several ways, see CHOLMOD Userguide for further information
   */
  cholmod_common& cholmodCommonObject()
  {
      return c_;
  }

  /** \brief The CHOLMOD data structure that stores the factorization
   *
   * Access to this is necessary for the more advanced features of CHOLMOD.
   * You need to know what you are doing!
   */
  cholmod_factor& cholmodFactor()
  {
    return *L_;
  }

  /** \brief The CHOLMOD data structure that stores the factorization
   *
   * Access to this is necessary for the more advanced features of CHOLMOD.
   * You need to know what you are doing!
   */
  const cholmod_factor& cholmodFactor() const
  {
    return *L_;
  }
private:

  // create a std::unique_ptr to a cholmod_dense object with a deleter
  // that calls the appropriate cholmod cleanup routine
  auto make_cholmod_dense(cholmod_dense* x, cholmod_common* c)
  {
    const auto deleter = [c](auto* p) {
      cholmod_free_dense(&p, c);
    };
    return std::unique_ptr<cholmod_dense, decltype(deleter)>(x, deleter);
  }

  cholmod_common c_;
  cholmod_factor* L_ = nullptr;

  // indicator for a 0x0 problem (due to ignore dof's)
  bool nIsZero_ = false;

  // vector mapping all indices in flat order to the not ignored indices
  std::vector<std::size_t> subIndices_;
};

  struct CholmodCreator{
    template<class F> struct isValidBlock : std::false_type{};
    template<int k> struct isValidBlock<FieldVector<double,k>> : std::true_type{};
    template<int k> struct isValidBlock<FieldVector<float,k>> : std::true_type{};

    template<class TL, typename M>
    std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                          typename Dune::TypeListElement<2, TL>::type>>
    operator()(TL /*tl*/, const M& mat, const Dune::ParameterTree& /*config*/,
               std::enable_if_t<isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value,int> = 0) const
    {
      using D = typename Dune::TypeListElement<1, TL>::type;
      auto solver = std::make_shared<Dune::Cholmod<D>>();
      solver->setMatrix(mat);
      return solver;
    }

    // second version with SFINAE to validate the template parameters of Cholmod
    template<typename TL, typename M>
    std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                          typename Dune::TypeListElement<2, TL>::type>>
    operator() (TL /*tl*/, const M& /*mat*/, const Dune::ParameterTree& /*config*/,
                std::enable_if_t<!isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value,int> = 0) const
    {
      DUNE_THROW(UnsupportedType, "Unsupported Type in Cholmod");
    }
  };
  DUNE_REGISTER_DIRECT_SOLVER("cholmod", Dune::CholmodCreator());

} /* namespace Dune */

#endif // HAVE_SUITESPARSE_CHOLMOD
