// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTLIO_HH
#define DUNE_ISTLIO_HH

#include <cmath>
#include <complex>
#include <limits>
#include <ios>
#include <iomanip>
#include <fstream>
#include <string>

#include "istlexception.hh"
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include "bcrsmatrix.hh"


namespace Dune {

  /**
              @addtogroup ISTL_SPMV
              @{
   */


  /** \file

     \brief Some generic functions for pretty printing vectors and matrices
   */
  //==== pretty printing of vectors

  // recursively print all the blocks
  template<class V>
  void recursive_printvector (std::ostream& s, const V& v, std::string rowtext, int& counter,
                              int columns, int width, int precision)
  {
    for (typename V::ConstIterator i=v.begin(); i!=v.end(); ++i)
      recursive_printvector(s,*i,rowtext,counter,columns,width,precision);
  }

  // specialization for FieldVector
  template<class K, int n>
  void recursive_printvector (std::ostream& s, const FieldVector<K,n>& v, std::string rowtext, int& counter,
                              int columns, int width, int precision)
  {
    // we now can print n numbers
    for (int i=0; i<n; i++)
    {
      if (counter%columns==0)
      {
        s << rowtext;                 // start a new row
        s << " ";                     // space in front of each entry
        s.width(4);                   // set width for counter
        s << counter;                 // number of first entry in a line
      }
      s << " ";                   // space in front of each entry
      s.width(width);             // set width for each entry anew
      s << v[i];                  // yeah, the number !
      counter++;                  // increment the counter
      if (counter%columns==0)
        s << std::endl;           // start a new line
    }
  }


  template<class V>
  void printvector (std::ostream& s, const V& v, std::string title, std::string rowtext,
                    int columns=1, int width=10, int precision=2)
  {
    // count the numbers printed to make columns
    int counter=0;

    // remember old flags
    std::ios_base::fmtflags oldflags = s.flags();

    // set the output format
    s.setf(std::ios_base::scientific, std::ios_base::floatfield);
    int oldprec = s.precision();
    s.precision(precision);

    // print title
    s << title << " [blocks=" << v.N() << ",dimension=" << v.dim() << "]" << std::endl;

    // print data from all blocks
    recursive_printvector(s,v,rowtext,counter,columns,width,precision);

    // check if new line is required
    if (counter%columns!=0)
      s << std::endl;

    // reset the output format
    s.flags(oldflags);
    s.precision(oldprec);
  }


  //==== pretty printing of matrices


  //! print a row of zeros for a non-existing block
  inline void fill_row (std::ostream& s, int m, int width, int precision)
  {
    for (int j=0; j<m; j++)
    {
      s << " ";                   // space in front of each entry
      s.width(width);             // set width for each entry anew
      s << ".";                   // yeah, the number !
    }
  }

  template<typename B, typename A>
  class BCRSMatrix;

  template<typename K, int n, int m>
  class FieldMatrix;

  template<typename M>
  struct MatrixDimension
  {};


  template<typename B, typename TA>
  struct MatrixDimension<BCRSMatrix<B,TA> >
  {
    typedef BCRSMatrix<B,TA> Matrix;
    typedef typename Matrix::block_type block_type;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim (const Matrix& A, size_type i)
    {
      const B* row = A.r[i].getptr();
      if(row)
        return MatrixDimension<block_type>::rowdim(*row);
      else
        return 0;
    }

    static size_type coldim (const Matrix& A, size_type c)
    {
      // find an entry in column j
      if (A.nnz>0)
      {
        for (size_type k=0; k<A.nnz; k++) {
          if (A.j[k]==c) {
            return MatrixDimension<block_type>::coldim(A.a[k]);
          }
        }
      }
      else
      {
        for (size_type i=0; i<A.N(); i++)
        {
          size_type* j = A.r[i].getindexptr();
          B*   a = A.r[i].getptr();
          for (size_type k=0; k<A.r[i].getsize(); k++)
            if (j[k]==c) {
              return MatrixDimension<block_type>::coldim(a[k]);
            }
        }
      }

      // not found
      return 0;
    }

    static size_type rowdim (const Matrix& A){
      size_type nn=0;
      for (size_type i=0; i<A.N(); i++)
        nn += rowdim(A,i);
      return nn;
    }

    static size_type coldim (const Matrix& A){
      typedef typename Matrix::ConstRowIterator ConstRowIterator;
      typedef typename Matrix::ConstColIterator ConstColIterator;

      // The following code has a complexity of nnz, and
      // typically a very small constant.
      //
      std::vector<size_type> coldims(A.M(),std::numeric_limits<size_type>::max());

      for (ConstRowIterator row=A.begin(); row!=A.end(); ++row)
        for (ConstColIterator col=row->begin(); col!=row->end(); ++col)
          // only compute blocksizes we don't already have
          if (coldims[col.index()]==std::numeric_limits<size_type>::max())
            coldims[col.index()] = MatrixDimension<block_type>::coldim(*col);

      size_type sum = 0;
      for (typename std::vector<size_type>::iterator it=coldims.begin(); it!=coldims.end(); ++it)
        // skip rows for which no coldim could be determined
        if ((*it)>=0)
          sum += *it;

      return sum;
    }
  };


  template<typename B, int n, int m, typename TA>
  struct MatrixDimension<BCRSMatrix<FieldMatrix<B,n,m> ,TA> >
  {
    typedef BCRSMatrix<FieldMatrix<B,n,m> ,TA> Matrix;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim (const Matrix& A, size_type i)
    {
      return n;
    }

    static size_type coldim (const Matrix& A, size_type c)
    {
      return m;
    }

    static size_type rowdim (const Matrix& A){
      return A.N()*n;
    }

    static size_type coldim (const Matrix& A){
      return A.M()*m;
    }
  };

  template<typename K, int n, int m>
  struct MatrixDimension<FieldMatrix<K,n,m> >
  {
    typedef FieldMatrix<K,n,m> Matrix;
    typedef typename Matrix::size_type size_type;

    static size_type rowdim(const Matrix& A, size_type r)
    {
      return 1;
    }

    static size_type coldim(const Matrix& A, size_type r)
    {
      return 1;
    }

    static size_type rowdim(const Matrix& A)
    {
      return n;
    }

    static size_type coldim(const Matrix& A)
    {
      return m;
    }
  };

  //! print one row of a matrix
  template<class M>
  void print_row (std::ostream& s, const M& A, typename M::size_type I,
                  typename M::size_type J, typename M::size_type therow,
                  int width, int precision)
  {
    typename M::size_type i0=I;
    for (typename M::size_type i=0; i<A.N(); i++)
    {
      if (therow>=i0 && therow<i0+MatrixDimension<M>::rowdim(A,i))
      {
        // the row is in this block row !
        typename M::size_type j0=J;
        for (typename M::size_type j=0; j<A.M(); j++)
        {
          // find this block
          typename M::ConstColIterator it = A[i].find(j);

          // print row or filler
          if (it!=A[i].end())
            print_row(s,*it,i0,j0,therow,width,precision);
          else
            fill_row(s,MatrixDimension<M>::coldim(A,j),width,precision);

          // advance columns
          j0 += MatrixDimension<M>::coldim(A,j);
        }
      }
      // advance rows
      i0 += MatrixDimension<M>::rowdim(A,i);
    }
  }

  //! print one row of a matrix, specialization for FieldMatrix
  template<class K, int n, int m>
  void print_row (std::ostream& s, const FieldMatrix<K,n,m>& A,
                  typename FieldMatrix<K,n,m>::size_type I, typename FieldMatrix<K,n,m>::size_type J,
                  typename FieldMatrix<K,n,m>::size_type therow, int width, int precision)
  {
    typedef typename FieldMatrix<K,n,m>::size_type size_type;

    for (size_type i=0; i<n; i++)
      if (I+i==therow)
        for (int j=0; j<m; j++)
        {
          s << " ";                         // space in front of each entry
          s.width(width);                   // set width for each entry anew
          s << A[i][j];                     // yeah, the number !
        }
  }

  //! print one row of a matrix, specialization for FieldMatrix<K,1,1>
  template<class K>
  void print_row (std::ostream& s, const FieldMatrix<K,1,1>& A, typename FieldMatrix<K,1,1>::size_type I,
                  typename FieldMatrix<K,1,1>::size_type J, typename FieldMatrix<K,1,1>::size_type therow,
                  int width, int precision)
  {
    if (I==therow)
    {
      s << " ";                   // space in front of each entry
      s.width(width);             // set width for each entry anew
      s << static_cast<K>(A);                   // yeah, the number !
    }
  }

  /** \brief Prints a generic block matrix
      \bug Empty rows and columns are omitted by this method.  (FlySpray #7)
   */
  template<class M>
  void printmatrix (std::ostream& s, const M& A, std::string title, std::string rowtext,
                    int width=10, int precision=2)
  {

    // remember old flags
    std::ios_base::fmtflags oldflags = s.flags();

    // set the output format
    s.setf(std::ios_base::scientific, std::ios_base::floatfield);
    int oldprec = s.precision();
    s.precision(precision);

    // print title
    s << title
      << " [n=" << A.N()
      << ",m=" << A.M()
      << ",rowdim=" << MatrixDimension<M>::rowdim(A)
      << ",coldim=" << MatrixDimension<M>::coldim(A)
      << "]" << std::endl;

    // print all rows
    for (typename M::size_type i=0; i<MatrixDimension<M>::rowdim(A); i++)
    {
      s << rowtext;            // start a new row
      s << " ";                // space in front of each entry
      s.width(4);              // set width for counter
      s << i;                  // number of first entry in a line
      print_row(s,A,0,0,i,width,precision);           // generic print
      s << std::endl;          // start a new line
    }

    // reset the output format
    s.flags(oldflags);
    s.precision(oldprec);
  }

  /**
   * @brief Prints a BCRSMatrix with fixed sized blocks.
   *
   * Only the nonzero entries will be printed as matrix blocks
   * together with their
   * corresponding column index and all others will be omitted.
   *
   * This might be preferable over printmatrix in the case of big
   * sparse matrices with nonscalar blocks.
   *
   * @param s The ostream to print to.
   * @param mat The matrix to print.
   * @param title The title for the matrix.
   * @param rowtext The text to prepend to each print out of a matrix row.
   * @param width The number of nonzero blocks to print in one line.
   * @param precision The precision to use when printing the numbers.
   */
  template<class B, int n, int m, class A>
  void printSparseMatrix(std::ostream& s,
                         const BCRSMatrix<FieldMatrix<B,n,m>,A>& mat,
                         std::string title, std::string rowtext,
                         int width=3, int precision=2)
  {
    typedef BCRSMatrix<FieldMatrix<B,n,m>,A> Matrix;
    // remember old flags
    std::ios_base::fmtflags oldflags = s.flags();
    // set the output format
    s.setf(std::ios_base::scientific, std::ios_base::floatfield);
    int oldprec = s.precision();
    s.precision(precision);
    // print title
    s << title
      << " [n=" << mat.N()
      << ",m=" << mat.M()
      << ",rowdim=" << MatrixDimension<Matrix>::rowdim(mat)
      << ",coldim=" << MatrixDimension<Matrix>::coldim(mat)
      << "]" << std::endl;

    typedef typename Matrix::ConstRowIterator Row;

    for(Row row=mat.begin(); row != mat.end(); ++row) {
      int skipcols=0;
      bool reachedEnd=false;

      while(!reachedEnd) {
        for(int innerrow=0; innerrow<n; ++innerrow) {
          int count=0;
          typedef typename Matrix::ConstColIterator Col;
          Col col=row->begin();
          for(; col != row->end(); ++col,++count) {
            if(count<skipcols)
              continue;
            if(count>=skipcols+width)
              break;
            if(innerrow==0) {
              if(count==skipcols) {
                s << rowtext;  // start a new row
                s << " ";      // space in front of each entry
                s.width(4);    // set width for counter
                s << row.index()<<": ";        // number of first entry in a line
              }
              s.width(4);
              s<<col.index()<<": |";
            }else{
              if(count==skipcols) {
                for(int i=0; i < rowtext.length(); i++)
                  s<<" ";
                s<<"       ";
              }
              s<<"      |";
            }
            for(int innercol=0; innercol < m; ++innercol) {
              s.width(9);
              s<<(*col)[innerrow][innercol]<<" ";
            }

            s<<"|";
          }
          if(innerrow==n-1 && col==row->end())
            reachedEnd=true;
          else
            s<<std::endl;
        }
        skipcols+=width;
        s<<std::endl;
      }
      s<<std::endl;
    }

    // reset the output format
    s.flags(oldflags);
    s.precision(oldprec);
  }

  namespace
  {
    template<typename T>
    struct MatlabPODWriter
    {
      static std::ostream& write(const T& t,  std::ostream& s)
      {
        s<<t;
        return s;
      }
    };
    template<typename T>
    struct MatlabPODWriter<std::complex<T> >
    {
      static std::ostream& write(const std::complex<T>& t,  std::ostream& s)
      {
        s<<t.real()<<" "<<t.imag();
        return s;
      }
    };
  }
  /** \brief Helper method for the writeMatrixToMatlab routine.

     This specialization for FieldMatrices ends the recursion
   */
  template <class FieldType, int rows, int cols>
  void writeMatrixToMatlabHelper(const FieldMatrix<FieldType,rows,cols>& matrix,
                                 int rowOffset, int colOffset,
                                 std::ostream& s)
  {
    for (int i=0; i<rows; i++)
      for (int j=0; j<cols; j++) {
        //+1 for Matlab numbering
        s << rowOffset + i + 1 << " " << colOffset + j + 1 << " ";
        MatlabPODWriter<FieldType>::write(matrix[i][j], s)<< std::endl;
      }
  }

  template <class MatrixType>
  void writeMatrixToMatlabHelper(const MatrixType& matrix,
                                 int externalRowOffset, int externalColOffset,
                                 std::ostream& s)
  {
    // Precompute the accumulated sizes of the columns
    std::vector<typename MatrixType::size_type> colOffset(matrix.M());
    if (colOffset.size() > 0)
      colOffset[0] = 0;

    for (typename MatrixType::size_type i=0; i<matrix.M()-1; i++)
      colOffset[i+1] = colOffset[i] + MatrixDimension<MatrixType>::coldim(matrix,i);

    typename MatrixType::size_type rowOffset = 0;

    // Loop over all matrix rows
    for (typename MatrixType::size_type rowIdx=0; rowIdx<matrix.N(); rowIdx++) {

      const typename MatrixType::row_type& row = matrix[rowIdx];

      typename MatrixType::row_type::ConstIterator cIt   = row.begin();
      typename MatrixType::row_type::ConstIterator cEndIt = row.end();

      // Loop over all columns in this row
      for (; cIt!=cEndIt; ++cIt)
        writeMatrixToMatlabHelper(*cIt,
                                  externalRowOffset+rowOffset,
                                  externalColOffset + colOffset[cIt.index()],
                                  s);

      rowOffset += MatrixDimension<MatrixType>::rowdim(matrix, rowIdx);
    }

  }

  /** \brief Writes sparse matrix in a Matlab-readable format
   *
   * This routine writes the argument BCRSMatrix to a file with the name given by
   * the filename argument.  The file format is ASCII, with no header, and three
   * data columns.  Each row describes a scalar matrix entry and consists of the
   * matrix row and column numbers (both counted starting from 1), and the matrix
   * entry.  Such a file can be read from Matlab using the command
   * \verbatim
       new_mat = spconvert(load('filename'));
     \endverbatim
   */
  template <class MatrixType>
  void writeMatrixToMatlab(const MatrixType& matrix,
                           const std::string& filename)
  {
    std::ofstream outStream(filename.c_str());

    writeMatrixToMatlabHelper(matrix, 0, 0, outStream);
  }

  /** @} end documentation */

} // end namespace

#endif
