// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTLIO_HH
#define DUNE_ISTLIO_HH

#include <cmath>
#include <complex>
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

  //! print one row of a matrix
  template<class M>
  void print_row (std::ostream& s, const M& A, typename M::size_type I,
                  typename M::size_type J, typename M::size_type therow,
                  int width, int precision)
  {
    typename M::size_type i0=I;
    for (typename M::size_type i=0; i<A.N(); i++)
    {
      if (therow>=i0 && therow<i0+A.rowdim(i))
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
            fill_row(s,A.coldim(j),width,precision);

          // advance columns
          j0 += A.coldim(j);
        }
      }
      // advance rows
      i0 += A.rowdim(i);
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
      << ",rowdim=" << A.rowdim()
      << ",coldim=" << A.coldim()
      << "]" << std::endl;

    // print all rows
    for (typename M::size_type i=0; i<A.rowdim(); i++)
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
      << ",rowdim=" << mat.rowdim()
      << ",coldim=" << mat.coldim()
      << "]" << std::endl;

    typedef typename BCRSMatrix<FieldMatrix<B,n,m>,A>::ConstRowIterator Row;

    for(Row row=mat.begin(); row != mat.end(); ++row) {
      int skipcols=0;
      bool reachedEnd=false;

      while(!reachedEnd) {
        for(int innerrow=0; innerrow<n; ++innerrow) {
          int count=0;
          typedef typename BCRSMatrix<FieldMatrix<B,n,m>,A>::ConstColIterator Col;
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
    std::vector<unsigned int> colOffset(matrix.M());
    if (colOffset.size() > 0)
      colOffset[0] = 0;

    for (int i=0; i<matrix.M()-1; i++)
      colOffset[i+1] = colOffset[i] + matrix.coldim(i);

    int rowOffset = 0;

    // Loop over all matrix rows
    for (int rowIdx=0; rowIdx<matrix.N(); rowIdx++) {

      const typename MatrixType::row_type& row = matrix[rowIdx];

      typename MatrixType::row_type::ConstIterator cIt   = row.begin();
      typename MatrixType::row_type::ConstIterator cEndIt = row.end();

      // Loop over all columns in this row
      for (; cIt!=cEndIt; ++cIt)
        writeMatrixToMatlabHelper(*cIt,
                                  externalRowOffset+rowOffset,
                                  externalColOffset + colOffset[cIt.index()],
                                  s);

      rowOffset += matrix.rowdim(rowIdx);
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
