// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MATRIXMATRIX_HH
#define DUNE_ISTL_MATRIXMATRIX_HH

#include <tuple>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/timer.hh>
namespace Dune
{

  /**
   * @addtogroup ISTL_SPMV
   *
   * @{
   */
  /** @file
   * @author Markus Blatt
   * @brief provides functions for sparse matrix matrix multiplication.
   */

  namespace
  {

    /**
     * @brief Traverses over the nonzero pattern of the matrix-matrix product.
     *
     * Template parameter b is used to select the matrix product:
     * <dt>0</dt><dd>\f$A\cdot B\f$</dd>
     * <dt>1</dt><dd>\f$A^T\cdot B\f$</dd>
     * <dt>2</dt><dd>\f$A\cdot B^T\f$</dd>
     */
    template<int b>
    struct NonzeroPatternTraverser
    {};


    template<>
    struct NonzeroPatternTraverser<0>
    {
      template<class T,class A1, class A2, class F, int n, int m, int k>
      static void traverse(const Dune::BCRSMatrix<Dune::FieldMatrix<T,n,k>,A1>& A,
                           const Dune::BCRSMatrix<Dune::FieldMatrix<T,k,m>,A2>& B,
                           F& func)
      {
        if(A.M()!=B.N())
          DUNE_THROW(ISTLError, "The sizes of the matrices do not match: "<<A.M()<<"!="<<B.N());

        typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<T,n,k>,A1>::ConstRowIterator Row;
        typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<T,n,k>,A1>::ConstColIterator Col;
        typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<T,k,m>,A2>::ConstColIterator BCol;
        for(Row row= A.begin(); row != A.end(); ++row) {
          // Loop over all column entries
          for(Col col = row->begin(); col != row->end(); ++col) {
            // entry at i,k
            // search for all nonzeros in row k
            for(BCol bcol = B[col.index()].begin(); bcol != B[col.index()].end(); ++bcol) {
              func(*col, *bcol, row.index(), bcol.index());
            }
          }
        }
      }

    };

    template<>
    struct NonzeroPatternTraverser<1>
    {
      template<class T, class A1, class A2, class F, int n, int m, int k>
      static void traverse(const Dune::BCRSMatrix<Dune::FieldMatrix<T,k,n>,A1>& A,
                           const Dune::BCRSMatrix<Dune::FieldMatrix<T,k,m>,A2>& B,
                           F& func)
      {

        if(A.N()!=B.N())
          DUNE_THROW(ISTLError, "The sizes of the matrices do not match: "<<A.N()<<"!="<<B.N());

        typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<T,k,n>,A1>::ConstRowIterator Row;
        typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<T,k,n>,A1>::ConstColIterator Col;
        typedef typename Dune::BCRSMatrix<Dune::FieldMatrix<T,k,m>,A2>::ConstColIterator BCol;

        for(Row row=A.begin(); row!=A.end(); ++row) {
          for(Col col=row->begin(); col!=row->end(); ++col) {
            for(BCol bcol  = B[row.index()].begin(); bcol !=  B[row.index()].end(); ++bcol) {
              func(*col, *bcol, col.index(), bcol.index());
            }
          }
        }
      }
    };

    template<>
    struct NonzeroPatternTraverser<2>
    {
      template<class T, class A1, class A2, class F, int n, int m, int k>
      static void traverse(const BCRSMatrix<FieldMatrix<T,n,m>,A1>& mat,
                           const BCRSMatrix<FieldMatrix<T,k,m>,A2>& matt,
                           F& func)
      {
        if(mat.M()!=matt.M())
          DUNE_THROW(ISTLError, "The sizes of the matrices do not match: "<<mat.M()<<"!="<<matt.M());

        typedef typename BCRSMatrix<FieldMatrix<T,n,m>,A1>::ConstRowIterator row_iterator;
        typedef typename BCRSMatrix<FieldMatrix<T,n,m>,A1>::ConstColIterator col_iterator;
        typedef typename BCRSMatrix<FieldMatrix<T,k,m>,A2>::ConstRowIterator row_iterator_t;
        typedef typename BCRSMatrix<FieldMatrix<T,k,m>,A2>::ConstColIterator col_iterator_t;

        for(row_iterator mrow=mat.begin(); mrow != mat.end(); ++mrow) {
          //iterate over the column entries
          // mt is a transposed matrix crs therefore it is treated as a ccs matrix
          // and the row_iterator iterates over the columns of the transposed matrix.
          // search the row of the transposed matrix for an entry with the same index
          // as the mcol iterator

          for(row_iterator_t mtcol=matt.begin(); mtcol != matt.end(); ++mtcol) {
            //Search for col entries in mat that have a corrsponding row index in matt
            // (i.e. corresponding col index in the as this is the transposed matrix
            col_iterator_t mtrow=mtcol->begin();
            bool funcCalled = false;
            for(col_iterator mcol=mrow->begin(); mcol != mrow->end(); ++mcol) {
              // search
              // TODO: This should probably be substituted by a binary search
              for( ; mtrow != mtcol->end(); ++mtrow)
                if(mtrow.index()>=mcol.index())
                  break;
              if(mtrow != mtcol->end() && mtrow.index()==mcol.index()) {
                func(*mcol, *mtrow, mtcol.index());
                funcCalled = true;
                // In some cases we only search for one pair, then we break here
                // and continue with the next column.
                if(F::do_break)
                  break;
              }
            }
            // move on with func only if func was called, otherwise they might
            // get out of sync
            if (funcCalled)
              func.nextCol();
          }
          func.nextRow();
        }
      }
    };



    template<class T, class A, int n, int m>
    class SparsityPatternInitializer
    {
    public:
      enum {do_break=true};
      typedef typename BCRSMatrix<FieldMatrix<T,n,m>,A>::CreateIterator CreateIterator;
      typedef typename BCRSMatrix<FieldMatrix<T,n,m>,A>::size_type size_type;

      SparsityPatternInitializer(CreateIterator iter)
        : rowiter(iter)
      {}

      template<class T1, class T2>
      void operator()(const T1&, const T2&, size_type j)
      {
        rowiter.insert(j);
      }

      void nextRow()
      {
        ++rowiter;
      }
      void nextCol()
      {}

    private:
      CreateIterator rowiter;
    };


    template<int transpose, class T, class TA, int n, int m>
    class MatrixInitializer
    {
    public:
      enum {do_break=true};
      typedef typename Dune::BCRSMatrix<FieldMatrix<T,n,m>,TA> Matrix;
      typedef typename Matrix::CreateIterator CreateIterator;
      typedef typename Matrix::size_type size_type;

      MatrixInitializer(Matrix& A_, size_type)
        : count(0), A(A_)
      {}
      template<class T1, class T2>
      void operator()(const T1&, const T2&, int)
      {
        ++count;
      }

      void nextCol()
      {}

      void nextRow()
      {}

      std::size_t nonzeros()
      {
        return count;
      }

      template<class A1, class A2, int n2, int m2, int n3, int m3>
      void initPattern(const BCRSMatrix<FieldMatrix<T,n2,m2>,A1>& mat1,
                       const BCRSMatrix<FieldMatrix<T,n3,m3>,A2>& mat2)
      {
        SparsityPatternInitializer<T, TA, n, m> sparsity(A.createbegin());
        NonzeroPatternTraverser<transpose>::traverse(mat1,mat2,sparsity);
      }

    private:
      std::size_t count;
      Matrix& A;
    };

    template<class T, class TA, int n, int m>
    class MatrixInitializer<1,T,TA,n,m>
    {
    public:
      enum {do_break=false};
      typedef Dune::BCRSMatrix<Dune::FieldMatrix<T,n,m>,TA> Matrix;
      typedef typename Matrix::CreateIterator CreateIterator;
      typedef typename Matrix::size_type size_type;

      MatrixInitializer(Matrix& A_, size_type rows)
        :  A(A_), entries(rows)
      {}

      template<class T1, class T2>
      void operator()(const T1&, const T2&, size_type i, size_type j)
      {
        entries[i].insert(j);
      }

      void nextCol()
      {}

      size_type nonzeros()
      {
        size_type nnz=0;
        typedef typename std::vector<std::set<size_t> >::const_iterator Iter;
        for(Iter iter = entries.begin(); iter != entries.end(); ++iter)
          nnz+=(*iter).size();
        return nnz;
      }
      template<class A1, class A2, int n2, int m2, int n3, int m3>
      void initPattern(const BCRSMatrix<FieldMatrix<T,n2,m2>,A1>&,
                       const BCRSMatrix<FieldMatrix<T,n3,m3>,A2>&)
      {
        typedef typename std::vector<std::set<size_t> >::const_iterator Iter;
        CreateIterator citer = A.createbegin();
        for(Iter iter = entries.begin(); iter != entries.end(); ++iter, ++citer) {
          typedef std::set<size_t>::const_iterator SetIter;
          for(SetIter index=iter->begin(); index != iter->end(); ++index)
            citer.insert(*index);
        }
      }

    private:
      Matrix& A;
      std::vector<std::set<size_t> > entries;
    };

    template<class T, class TA, int n, int m>
    struct MatrixInitializer<0,T,TA,n,m>
      : public MatrixInitializer<1,T,TA,n,m>
    {
      MatrixInitializer(Dune::BCRSMatrix<Dune::FieldMatrix<T,n,m>,TA>& A_,
                        typename Dune::BCRSMatrix<Dune::FieldMatrix<T,n,m>,TA>::size_type rows)
        : MatrixInitializer<1,T,TA,n,m>(A_,rows)
      {}
    };


    template<class T, class T1, class T2, int n, int m, int k>
    void addMatMultTransposeMat(FieldMatrix<T,n,k>& res, const FieldMatrix<T1,n,m>& mat,
                                const FieldMatrix<T2,k,m>& matt)
    {
      typedef typename FieldMatrix<T,n,k>::size_type size_type;

      for(size_type row=0; row<n; ++row)
        for(size_type col=0; col<k; ++col) {
          for(size_type i=0; i < m; ++i)
            res[row][col]+=mat[row][i]*matt[col][i];
        }
    }

    template<class T, class T1, class T2, int n, int m, int k>
    void addTransposeMatMultMat(FieldMatrix<T,n,k>& res, const FieldMatrix<T1,m,n>& mat,
                                const FieldMatrix<T2,m,k>& matt)
    {
      typedef typename FieldMatrix<T,n,k>::size_type size_type;
      for(size_type i=0; i<m; ++i)
        for(size_type row=0; row<n; ++row) {
          for(size_type col=0; col < k; ++col)
            res[row][col]+=mat[i][row]*matt[i][col];
        }
    }

    template<class T, class T1, class T2, int n, int m, int k>
    void addMatMultMat(FieldMatrix<T,n,m>& res, const FieldMatrix<T1,n,k>& mat,
                       const FieldMatrix<T2,k,m>& matt)
    {
      typedef typename FieldMatrix<T,n,k>::size_type size_type;
      for(size_type row=0; row<n; ++row)
        for(size_type col=0; col<m; ++col) {
          for(size_type i=0; i < k; ++i)
            res[row][col]+=mat[row][i]*matt[i][col];
        }
    }


    template<class T, class A, int n, int m>
    class EntryAccumulatorFather
    {
    public:
      enum {do_break=false};
      typedef BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
      typedef typename Matrix::RowIterator Row;
      typedef typename Matrix::ColIterator Col;

      EntryAccumulatorFather(Matrix& mat_)
        : mat(mat_), row(mat.begin())
      {
        mat=0;
        col=row->begin();
      }
      void nextRow()
      {
        ++row;
        if(row!=mat.end())
          col=row->begin();
      }

      void nextCol()
      {
        ++this->col;
      }
    protected:
      Matrix& mat;
    private:
      Row row;
    protected:
      Col col;
    };

    template<class T, class A, int n, int m, int transpose>
    class EntryAccumulator
      : public EntryAccumulatorFather<T,A,n,m>
    {
    public:
      typedef BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
      typedef typename Matrix::size_type size_type;

      EntryAccumulator(Matrix& mat_)
        : EntryAccumulatorFather<T,A,n,m>(mat_)
      {}

      template<class T1, class T2>
      void operator()(const T1& t1, const T2& t2, size_type i)
      {
        assert(this->col.index()==i);
        addMatMultMat(*(this->col),t1,t2);
      }
    };

    template<class T, class A, int n, int m>
    class EntryAccumulator<T,A,n,m,0>
      : public EntryAccumulatorFather<T,A,n,m>
    {
    public:
      typedef BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
      typedef typename Matrix::size_type size_type;

      EntryAccumulator(Matrix& mat_)
        : EntryAccumulatorFather<T,A,n,m>(mat_)
      {}

      template<class T1, class T2>
      void operator()(const T1& t1, const T2& t2, size_type i, size_type j)
      {
        addMatMultMat(this->mat[i][j], t1, t2);
      }
    };

    template<class T, class A, int n, int m>
    class EntryAccumulator<T,A,n,m,1>
      : public EntryAccumulatorFather<T,A,n,m>
    {
    public:
      typedef BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
      typedef typename Matrix::size_type size_type;

      EntryAccumulator(Matrix& mat_)
        : EntryAccumulatorFather<T,A,n,m>(mat_)
      {}

      template<class T1, class T2>
      void operator()(const T1& t1, const T2& t2, size_type i, size_type j)
      {
        addTransposeMatMultMat(this->mat[i][j], t1, t2);
      }
    };

    template<class T, class A, int n, int m>
    class EntryAccumulator<T,A,n,m,2>
      : public EntryAccumulatorFather<T,A,n,m>
    {
    public:
      typedef BCRSMatrix<FieldMatrix<T,n,m>,A> Matrix;
      typedef typename Matrix::size_type size_type;

      EntryAccumulator(Matrix& mat_)
        : EntryAccumulatorFather<T,A,n,m>(mat_)
      {}

      template<class T1, class T2>
      void operator()(const T1& t1, const T2& t2, [[maybe_unused]] size_type i)
      {
        assert(this->col.index()==i);
        addMatMultTransposeMat(*this->col,t1,t2);
      }
    };


    template<int transpose>
    struct SizeSelector
    {};

    template<>
    struct SizeSelector<0>
    {
      template<class M1, class M2>
      static std::tuple<typename M1::size_type, typename M2::size_type>
      size(const M1& m1, const M2& m2)
      {
        return std::make_tuple(m1.N(), m2.M());
      }
    };

    template<>
    struct SizeSelector<1>
    {
      template<class M1, class M2>
      static std::tuple<typename M1::size_type, typename M2::size_type>
      size(const M1& m1, const M2& m2)
      {
        return std::make_tuple(m1.M(), m2.M());
      }
    };


    template<>
    struct SizeSelector<2>
    {
      template<class M1, class M2>
      static std::tuple<typename M1::size_type, typename M2::size_type>
      size(const M1& m1, const M2& m2)
      {
        return std::make_tuple(m1.N(), m2.N());
      }
    };

    template<int transpose, class T, class A, class A1, class A2, int n1, int m1, int n2, int m2, int n3, int m3>
    void matMultMat(BCRSMatrix<FieldMatrix<T,n1,m1>,A>& res, const BCRSMatrix<FieldMatrix<T,n2,m2>,A1>& mat1,
                    const BCRSMatrix<FieldMatrix<T,n3,m3>,A2>& mat2)
    {
      // First step is to count the number of nonzeros
      typename BCRSMatrix<FieldMatrix<T,n1,m1>,A>::size_type rows, cols;
      std::tie(rows,cols)=SizeSelector<transpose>::size(mat1, mat2);
      MatrixInitializer<transpose,T,A,n1,m1> patternInit(res, rows);
      Timer timer;
      NonzeroPatternTraverser<transpose>::traverse(mat1,mat2,patternInit);
      res.setSize(rows, cols, patternInit.nonzeros());
      res.setBuildMode(BCRSMatrix<FieldMatrix<T,n1,m1>,A>::row_wise);

      //std::cout<<"Counting nonzeros took "<<timer.elapsed()<<std::endl;
      timer.reset();

      // Second step is to allocate the storage for the result and initialize the nonzero pattern
      patternInit.initPattern(mat1, mat2);

      //std::cout<<"Setting up sparsity pattern took "<<timer.elapsed()<<std::endl;
      timer.reset();
      // As a last step calculate the entries
      res = 0.0;
      EntryAccumulator<T,A,n1,m1, transpose> entriesAccu(res);
      NonzeroPatternTraverser<transpose>::traverse(mat1,mat2,entriesAccu);
      //std::cout<<"Calculating entries took "<<timer.elapsed()<<std::endl;
    }

  }

  /**
   * @brief Helper TMP to get the result type of a sparse matrix matrix multiplication (\f$C=A*B\f$)
   *
   * The type of matrix C will be stored as the associated type MatMultMatResult::type.
   * @tparam M1 The type of matrix A.
   * @tparam M2 The type of matrix B.
   */
  template<typename M1, typename M2>
  struct MatMultMatResult
  {};

  template<typename T, int n, int k, int m>
  struct MatMultMatResult<FieldMatrix<T,n,k>,FieldMatrix<T,k,m> >
  {
    typedef FieldMatrix<T,n,m> type;
  };

  template<typename T, typename A, typename A1, int n, int k, int m>
  struct MatMultMatResult<BCRSMatrix<FieldMatrix<T,n,k>,A >,BCRSMatrix<FieldMatrix<T,k,m>,A1 > >
  {
    typedef BCRSMatrix<typename MatMultMatResult<FieldMatrix<T,n,k>,FieldMatrix<T,k,m> >::type,
        std::allocator<typename MatMultMatResult<FieldMatrix<T,n,k>,FieldMatrix<T,k,m> >::type> > type;
  };


  /**
   * @brief Helper TMP to get the result type of a sparse matrix matrix multiplication (\f$C=A*B\f$)
   *
   * The type of matrix C will be stored as the associated type MatMultMatResult::type.
   * @tparam M1 The type of matrix A.
   * @tparam M2 The type of matrix B.
   */
  template<typename M1, typename M2>
  struct TransposedMatMultMatResult
  {};

  template<typename T, int n, int k, int m>
  struct TransposedMatMultMatResult<FieldMatrix<T,k,n>,FieldMatrix<T,k,m> >
  {
    typedef FieldMatrix<T,n,m> type;
  };

  template<typename T, typename A, typename A1, int n, int k, int m>
  struct TransposedMatMultMatResult<BCRSMatrix<FieldMatrix<T,k,n>,A >,BCRSMatrix<FieldMatrix<T,k,m>,A1 > >
  {
    typedef BCRSMatrix<typename MatMultMatResult<FieldMatrix<T,n,k>,FieldMatrix<T,k,m> >::type,
        std::allocator<typename MatMultMatResult<FieldMatrix<T,n,k>,FieldMatrix<T,k,m> >::type> > type;
  };


  /**
   * @brief Calculate product of a sparse matrix with a transposed sparse matrices (\f$C=A*B^T\f$).
   *
   * @param res Matrix for the result of the computation.
   * @param mat Matrix A.
   * @param matt Matrix B, which will be transposed before the multiplication.
   * @param tryHard <i>ignored</i>
   */
  template<class T, class A, class A1, class A2, int n, int m, int k>
  void matMultTransposeMat(BCRSMatrix<FieldMatrix<T,n,k>,A>& res, const BCRSMatrix<FieldMatrix<T,n,m>,A1>& mat,
                           const BCRSMatrix<FieldMatrix<T,k,m>,A2>& matt, [[maybe_unused]] bool tryHard=false)
  {
    matMultMat<2>(res,mat, matt);
  }

  /**
   * @brief Calculate product of two sparse matrices (\f$C=A*B\f$).
   *
   * @param res Matrix for the result of the computation.
   * @param mat Matrix A.
   * @param matt Matrix B.
   * @param tryHard <i>ignored</i>
   */
  template<class T, class A, class A1, class A2, int n, int m, int k>
  void matMultMat(BCRSMatrix<FieldMatrix<T,n,m>,A>& res, const BCRSMatrix<FieldMatrix<T,n,k>,A1>& mat,
                  const BCRSMatrix<FieldMatrix<T,k,m>,A2>& matt, bool tryHard=false)
  {
    matMultMat<0>(res,mat, matt);
  }

  /**
   * @brief Calculate product of a transposed sparse matrix with another sparse matrices (\f$C=A^T*B\f$).
   *
   * @param res Matrix for the result of the computation.
   * @param mat Matrix A, which will be transposed before the multiplication.
   * @param matt Matrix B.
   * @param tryHard <i>ignored</i>
   */
  template<class T, class A, class A1, class A2, int n, int m, int k>
  void transposeMatMultMat(BCRSMatrix<FieldMatrix<T,n,m>,A>& res, const BCRSMatrix<FieldMatrix<T,k,n>,A1>& mat,
                           const BCRSMatrix<FieldMatrix<T,k,m>,A2>& matt, [[maybe_unused]] bool tryHard=false)
  {
    matMultMat<1>(res,mat, matt);
  }

}
#endif
