#pragma once

#if HAVE_SUITESPARSE_CHOLMOD

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include<dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>

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
   * overhead should be preduced
   */
  struct NoIgnore
  {
    const NoIgnore& operator[](std::size_t) const { return *this; }
    explicit operator bool() const { return false; }
    std::size_t count() const { return 0; }
  };

} //namespace Impl

template<class T>
class Cholmod;

/** @brief Dune wrapper for SuiteSparse/CHOLMOD solver
  *
  * This class implements an InverseOperator between BlockVector types
  */
template<class T, class A, int k>
class Cholmod<BlockVector<FieldVector<T,k>, A>>
  : public InverseOperator<BlockVector<FieldVector<T,k>, A>, BlockVector<FieldVector<T,k>, A>>
{
public:

  // type of unknown
  using X = BlockVector<FieldVector<T,k>, A>;
  // type of rhs
  using B = BlockVector<FieldVector<T,k>, A>;


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
  void apply (X& x, B& b, [[maybe_unused]] double reduction, InverseOperatorResult& res)
  {
    apply(x,b,res);
  }

  /** @brief solve the linear system Ax=b (possibly with respect to some ignore field)
   *
   * The method assumes that setMatrix() was called before
   * In the case of a given ignore field the corresponding entries of both in x and b will stay untouched in this method.
  */
  void apply(X& x, B& b, InverseOperatorResult& res)
  {
    // do nothing if N=0
    if ( nIsZero_ )
    {
        return;
    }

    if (x.size() != b.size())
      DUNE_THROW(Exception, "Error in apply(): sizes of x and b do not match!");

    const auto& blocksize = k;

    // cast to double array
    auto b2 = std::make_unique<double[]>(L_->n);
    auto x2 = std::make_unique<double[]>(L_->n);

    // copy to cholmod
    auto bp = b2.get();
    if (inverseSubIndices_.empty()) // no ignore field given
    {
      // simply copy all values
      for (const auto& bi : b)
        for (const auto& bii : bi)
          *bp++ = bii;
    }
    else // use the mapping from not ignored entries
    {
      // iterate over inverseSubIndices and resolve the block indices
      for (const auto& idx : inverseSubIndices_)
        *bp++ = b[ idx / blocksize ][ idx % blocksize ];
    }

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
    if (inverseSubIndices_.empty()) // no ignore field given
    {
      // simply copy all values
      for (int i=0, s=x.size(); i<s; i++)
        for (int ii=0, ss=x[i].size(); ii<ss; ii++)
          x[i][ii] = *xp++;
    }
    else // use the mapping from not ignored entries
    {
      // iterate over inverseSubIndices and resolve the block indices
      for (const auto& idx : inverseSubIndices_)
        x[ idx / blocksize ][ idx % blocksize ] = *xp++;
    }

    // statistics for a direct solver
    res.iterations = 1;
    res.converged = true;
  }


  /** @brief Set matrix without ignore nodes
   *
   * This method forwards a nullptr to the setMatrix method
   * with ignore nodes
   */
  void setMatrix(const BCRSMatrix<FieldMatrix<T,k,k>>& matrix)
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
  template<class Ignore>
  void setMatrix(const BCRSMatrix<FieldMatrix<T,k,k>>& matrix, const Ignore* ignore)
  {

    const auto blocksize = k;

    // Total number of rows
    int N = blocksize * matrix.N();
    if ( ignore )
      N -= ignore->count();

    nIsZero_ = (N <= 0);

    if ( nIsZero_ )
    {
        return;
    }

    // number of nonzeroes
    const int nnz = blocksize * blocksize * matrix.nonzeroes();
    // number of diagonal entries
    const int nDiag = blocksize * blocksize * matrix.N();
    // number of nonzeroes in the dialgonal
    const int nnzDiag = (blocksize * (blocksize+1)) / 2 * matrix.N();
    // number of nonzeroes in triangular submatrix (including diagonal)
    const int nnzTri = (nnz - nDiag) / 2 + nnzDiag;

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
                              nnzTri,        // # of nonzeroes
                              1,             // indices are sorted ( 1 = true)
                              1,             // matrix is "packed" ( 1 = true)
                              -1,            // stype of matrix ( -1 = cosider the lower part only )
                              CHOLMOD_REAL,  // xtype of matrix ( CHOLMOD_REAL = single array, no complex numbers)
                              &c_            // cholmod_common ptr
                             ), deleter);

    // copy the data of BCRS matrix to Cholmod Sparse matrix
    int* Ap = static_cast<int*>(M->p);
    int* Ai = static_cast<int*>(M->i);
    double* Ax = static_cast<double*>(M->x);

    // vector mapping all indices in flat order to the not ignored indices
    std::vector<std::size_t> subIndices;

    if ( ignore )
    {
      // init the mappings
      inverseSubIndices_.resize(N);            // size = number of not ignored entries
      subIndices.resize(matrix.M()*blocksize); // size = number of all entries

      std::size_t j=0;
      for( std::size_t block=0; block<matrix.N(); block++ )
      {
        for( std::size_t i=0; i<blocksize; i++ )
        {
          if( not (*ignore)[block][i] )
          {
            subIndices[ block*blocksize + i ] = j;
            inverseSubIndices_[j] = block*blocksize + i;
            j++;
          }
        }
      }
    }

    // Copy the data to the CHOLMOD array
    int n = 0;
    for (auto rowIt = matrix.begin(); rowIt != matrix.end(); rowIt++)
    {
      const auto row = rowIt.index();
      for (std::size_t i=0; i<blocksize; i++)
      {
        if( ignore and (*ignore)[row][i] )
          continue;

        // col start
        *Ap++ = n;

        for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
        {
          const auto col = colIt.index();

          // are we already in the lower part?
          if (col < row)
            continue;

          for (auto j = (row == col ? i : 0); j<blocksize; j++)
          {
            if( ignore and (*ignore)[col][j] )
              continue;

            const auto jj = ignore ? subIndices[  blocksize*col + j ] : blocksize*col + j;

            // set the current index and entry
            *Ai++ = jj;
            *Ax++ = (*colIt)[i][j];
            // increase number of set values
            n++;
          }
        }
      }
    }
    // set last col start
    *Ap = n;

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

private:

  // create a destrucable unique_ptr
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

  // mapping from the not ignored indices in flat order to all indices in flat order
  // it also holds the info about ignore nodes: if it is empty => no ignore field
  std::vector<std::size_t> inverseSubIndices_;

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
