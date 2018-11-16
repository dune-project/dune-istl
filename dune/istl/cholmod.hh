#pragma once

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include<dune/istl/solver.hh>


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


template<class T, class A, int k>
class Cholmod<BlockVector<FieldVector<T,k>, A>>
  : public InverseOperator<
      BlockVector<FieldVector<T,k>, A>,
      BlockVector<FieldVector<T,k>, A>>
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

  // forbit this to avoid freeing memory twice
  Cholmod(const Cholmod&) = delete;
  Cholmod& operator=(const Cholmod&) = delete;


  /**
    *  \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)
    */
  void apply (X& x, B& b, double reduction, InverseOperatorResult& res) override
  {
    DUNE_UNUSED_PARAMETER(reduction);
    apply(x,b,res);
  }

  /**
  *  \copydoc InverseOperator::apply(X&, Y&, InverseOperatorResult&)
  */
  void apply(X& x, B& b, InverseOperatorResult& res) override
  {
    // cast to double array
    auto b2 = std::make_unique<double[]>(L_->n);
    auto x2 = std::make_unique<double[]>(L_->n);

    // copy to cholmod
    auto bp = b2.get();
    for (const auto& bi : b)
      for (const auto& bii : bi)
        *bp++ = bii;

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
    x.resize(L_->n);
    for (auto& xi : x)
      for (auto& xii : xi)
        xii = *xp++;

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
   * The ignore argument consists of a compatible BitVector,
   * indicating the dof's which has to be deleted from the matrix
   * Decomposing the matrix at the end takes a lot of time
   * \param [in] matrix Matrix to be decomposed. In BCRS compatible form
   * \param [in] ignore Pointer to a compatible BitVector
   */
  template<class Matrix, class Ignore>
  void setMatrix(const Matrix& matrix, const Ignore* ignore)
  {
    const auto blocksize = Matrix::block_type::rows;

    // Total number of rows
    int N = blocksize * matrix.N();
    if ( ignore )
      N -= ignore->count();

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

    // create a vector that maps each remaining matrix index to it's number in the condensed matrix
    std::vector<size_t> subIndices;

    if ( ignore )
    {
      subIndices.resize(matrix.M()*blocksize);

      size_t j=0;
      for( size_t block=0; block<matrix.N(); block++ )
      {
        for( size_t i=0; i<blocksize; i++ )
        {
          if( not (*ignore)[block][i] )
          {
            subIndices[ block*blocksize + i ] = j++;
          }
        }
      }
    }

    // Copy the data to the CHOLMOD array
    int n = 0;
    for (auto rowIt = matrix.begin(); rowIt != matrix.end(); rowIt++)
    {
      const auto row = rowIt.index();
      for (auto i=0; i<blocksize; i++)
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
};


} /* namespace Dune */
