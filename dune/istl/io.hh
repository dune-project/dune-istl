// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_IO_HH
#define DUNE_ISTL_IO_HH

#include <cmath>
#include <complex>
#include <limits>
#include <ios>
#include <iomanip>
#include <fstream>
#include <string>

#include "matrixutils.hh"
#include "istlexception.hh"
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/reservedvector.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/blocklevel.hh>

namespace Dune {

  /**
     @addtogroup ISTL_IO
     @{
   */


  /** \file

      \brief Some generic functions for pretty printing vectors and matrices
   */

  ////////////////////////////////////////////////////////////////////////
  //
  // pretty printing of vectors
  //

  /**
   * \brief Recursively print a vector
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   */
  template<class V>
  void recursive_printvector (std::ostream& s, const V& v, std::string rowtext,
                              int& counter, int columns, int width)
  {
    if constexpr (IsNumber<V>())
    {
      // Print one number
      if (counter%columns==0)
      {
        s << rowtext; // start a new row
        s << " ";     // space in front of each entry
        s.width(4);   // set width for counter
        s << counter; // number of first entry in a line
      }
      s << " ";         // space in front of each entry
      s.width(width);   // set width for each entry anew
      s << v;        // yeah, the number !
      counter++;        // increment the counter
      if (counter%columns==0)
        s << std::endl; // start a new line
    }
    else
    {
      // Recursively print a vector
      for (const auto& entry : v)
        recursive_printvector(s,entry,rowtext,counter,columns,width);
    }
  }


  /**
   * \brief Print an ISTL vector
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   */
  template<class V>
  void printvector (std::ostream& s, const V& v, std::string title,
                    std::string rowtext, int columns=1, int width=10,
                    int precision=2)
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
    s << title << " [blocks=" << v.N() << ",dimension=" << v.dim() << "]"
      << std::endl;

    // print data from all blocks
    recursive_printvector(s,v,rowtext,counter,columns,width);

    // check if new line is required
    if (counter%columns!=0)
      s << std::endl;

    // reset the output format
    s.flags(oldflags);
    s.precision(oldprec);
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // pretty printing of matrices
  //

  /**
   * \brief Print a row of zeros for a non-existing block
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   */
  inline void fill_row (std::ostream& s, int m, int width, [[maybe_unused]] int precision)
  {
    for (int j=0; j<m; j++)
    {
      s << " ";         // space in front of each entry
      s.width(width);   // set width for each entry anew
      s << ".";         // yeah, the number !
    }
  }

  /**
   * \brief Print one row of a matrix, specialization for number types
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   */
  template<class K>
  void print_row (std::ostream& s, const K& value,
                  [[maybe_unused]] typename FieldMatrix<K,1,1>::size_type I,
                  [[maybe_unused]] typename FieldMatrix<K,1,1>::size_type J,
                  [[maybe_unused]] typename FieldMatrix<K,1,1>::size_type therow,
                  int width,
                  [[maybe_unused]] int precision,
                  typename std::enable_if_t<Dune::IsNumber<K>::value>* sfinae = nullptr)
  {
    s << " ";         // space in front of each entry
    s.width(width);   // set width for each entry anew
    s << value;
  }

  /**
   * \brief Print one row of a matrix
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   */
  template<class M>
  void print_row (std::ostream& s, const M& A, typename M::size_type I,
                  typename M::size_type J, typename M::size_type therow,
                  int width, int precision,
                  typename std::enable_if_t<!Dune::IsNumber<M>::value>* sfinae = nullptr)
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

  /**
   * \brief Print a generic block matrix
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   * \bug Empty rows and columns are omitted by this method.  (FlySpray #7)
   */
  template<class M>
  void printmatrix (std::ostream& s, const M& A, std::string title,
                    std::string rowtext, int width=10, int precision=2)
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
      s << rowtext;  // start a new row
      s << " ";      // space in front of each entry
      s.width(4);    // set width for counter
      s << i;        // number of first entry in a line
      print_row(s,A,0,0,i,width,precision); // generic print
      s << std::endl; // start a new line
    }

    // reset the output format
    s.flags(oldflags);
    s.precision(oldprec);
  }

  /**
   * \brief Prints a BCRSMatrix with fixed sized blocks.
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
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
                s << rowtext;           // start a new row
                s << " ";               // space in front of each entry
                s.width(4);             // set width for counter
                s << row.index()<<": "; // number of first entry in a line
              }
              s.width(4);
              s<<col.index()<<": |";
            } else {
              if(count==skipcols) {
                for(typename std::string::size_type i=0; i < rowtext.length(); i++)
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
            reachedEnd = true;
          else
            s << std::endl;
        }
        skipcols += width;
        s << std::endl;
      }
      s << std::endl;
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
        s << t;
        return s;
      }
    };
    template<typename T>
    struct MatlabPODWriter<std::complex<T> >
    {
      static std::ostream& write(const std::complex<T>& t,  std::ostream& s)
      {
        s << t.real() << " " << t.imag();
        return s;
      }
    };
  } // anonymous namespace

  /**
   * \brief Helper method for the writeMatrixToMatlab routine.
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   *
   * This specialization for numbers ends the recursion
   */
  template <class FieldType>
  void writeMatrixToMatlabHelper(const FieldType& value,
                                 int rowOffset, int colOffset,
                                 std::ostream& s,
                                 typename std::enable_if_t<Dune::IsNumber<FieldType>::value>* sfinae = nullptr)
  {
    //+1 for Matlab numbering
    s << rowOffset + 1 << " " << colOffset + 1 << " ";
    MatlabPODWriter<FieldType>::write(value, s)<< std::endl;
  }

  /**
   * \brief Helper method for the writeMatrixToMatlab routine.
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   */
  template <class MatrixType>
  void writeMatrixToMatlabHelper(const MatrixType& matrix,
                                 int externalRowOffset, int externalColOffset,
                                 std::ostream& s,
                                 typename std::enable_if_t<!Dune::IsNumber<MatrixType>::value>* sfinae = nullptr)
  {
    // Precompute the accumulated sizes of the columns
    std::vector<typename MatrixType::size_type> colOffset(matrix.M());
    if (colOffset.size() > 0)
      colOffset[0] = 0;

    for (typename MatrixType::size_type i=0; i<matrix.M()-1; i++)
      colOffset[i+1] = colOffset[i] +
                       MatrixDimension<MatrixType>::coldim(matrix,i);

    typename MatrixType::size_type rowOffset = 0;

    // Loop over all matrix rows
    for (typename MatrixType::size_type rowIdx=0; rowIdx<matrix.N(); rowIdx++)
    {
      auto cIt    = matrix[rowIdx].begin();
      auto cEndIt = matrix[rowIdx].end();

      // Loop over all columns in this row
      for (; cIt!=cEndIt; ++cIt)
        writeMatrixToMatlabHelper(*cIt,
                                  externalRowOffset+rowOffset,
                                  externalColOffset + colOffset[cIt.index()],
                                  s);

      rowOffset += MatrixDimension<MatrixType>::rowdim(matrix, rowIdx);
    }

  }

  /**
   * \brief Writes sparse matrix in a Matlab-readable format
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   * This routine writes the argument BCRSMatrix to a file with the name given
   * by the filename argument.  The file format is ASCII, with no header, and
   * three data columns.  Each row describes a scalar matrix entry and
   * consists of the matrix row and column numbers (both counted starting from
   * 1), and the matrix entry.  Such a file can be read from Matlab using the
   * command
   * \code
   * new_mat = spconvert(load('filename'));
   * \endcode
   * @param matrix reference to matrix
   * @param filename
   * @param outputPrecision (number of digits) which is used to write the output file
   */
  template <class MatrixType>
  void writeMatrixToMatlab(const MatrixType& matrix,
                           const std::string& filename, int outputPrecision = 18)
  {
    std::ofstream outStream(filename.c_str());
    int oldPrecision = outStream.precision();
    outStream.precision(outputPrecision);

    writeMatrixToMatlabHelper(matrix, 0, 0, outStream);
    outStream.precision(oldPrecision);
  }

  // Recursively write vector entries to a stream
  template<class V>
  void writeVectorToMatlabHelper (const V& v, std::ostream& stream)
  {
    if constexpr (IsNumber<V>()) {
      stream << v << std::endl;
    } else {
      for (const auto& entry : v)
        writeVectorToMatlabHelper(entry, stream);
    }
  }

  /**
   * \brief Writes vectors in a Matlab-readable format
   *
   * \code
   * #include <dune/istl/io.hh>
   * \endcode
   * This routine writes the argument block vector to a file with the name given
   * by the filename argument. The file format is ASCII, with no header, and
   * a single data column. Such a file can be read from Matlab using the
   * command
   * \code
   * new_vec = load('filename');
   * \endcode
   * \param vector reference to vector to be printed to output file
   * \param filename filename of output file
   * \param outputPrecision (number of digits) which is used to write the output file
   */
  template <class VectorType>
  void writeVectorToMatlab(const VectorType& vector,
                           const std::string& filename, int outputPrecision = 18)
  {
    std::ofstream outStream(filename.c_str());
    int oldPrecision = outStream.precision();
    outStream.precision(outputPrecision);

    writeVectorToMatlabHelper(vector, outStream);
    outStream.precision(oldPrecision);
  }

  namespace Impl {

    //! Null stream. Any value streamed to this object is discarded
    struct NullStream {
      template <class Any>
      friend NullStream &operator<<(NullStream &dev0, Any &&) {
        return dev0;
      }
    };

    //! Writes an SVG header and scales offests to fit output options
    // svg shall be closed with a group and an svg. i.e. "</g></svg>"
    template <class Stream, class SVGMatrixOptions>
    void writeSVGMatrixHeader(Stream &out, const SVGMatrixOptions &opts,
                              std::pair<std::size_t, size_t> offsets) {
      auto [col_offset, row_offset] = offsets;
      double width = opts.width;
      double height = opts.height;
      // if empty, we try to figure out a sensible value of width and height
      if (opts.width == 0 and opts.height == 0)
        width = height = 500;
      if (opts.width == 0)
        width = opts.height * (double(col_offset) / row_offset);
      if (opts.height == 0)
        height = opts.width * (double(row_offset) / col_offset);

      // scale group w.r.t final offsets
      double scale_width = width / col_offset;
      double scale_height = height / row_offset;

      // write the header text
      out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << std::ceil(width)
          << "' height='" << std::ceil(height) << "' version='1.1'>\n"
          << "<style>\n"
          << opts.style << "</style>\n"
          << "<g transform='scale(" << scale_width << " " << scale_height
          << ")'>\n";
    }

    //! Writes (recursively) the svg content for an sparse matrix
    template <class Mat, class Stream, class SVGMatrixOptions,
              class RowPrefix, class ColPrefix>
    std::pair<std::size_t, size_t>
    writeSVGMatrix(const Mat &mat, Stream &out, SVGMatrixOptions opts,
                  RowPrefix row_prefix, ColPrefix col_prefix) {
      // get values to fill the offests
      const auto& block_size = opts.block_size;
      const auto& interspace = opts.interspace;

      const std::size_t rows = mat.N();
      const std::size_t cols = mat.M();

      // disable header write for recursive calls
      const bool write_header = opts.write_header;
      opts.write_header = false;

      // counter of offsets for every block
      std::size_t row_offset = interspace;
      std::size_t col_offset = interspace;

      // lambda helper: for-each value
      auto for_each_entry = [&mat](const auto &call_back) {
        for (auto row_it = mat.begin(); row_it != mat.end(); ++row_it) {
          for (auto col_it = row_it->begin(); col_it != row_it->end(); ++col_it) {
            call_back(row_it.index(), col_it.index(), *col_it);
          }
        }
      };

      // accumulate content in another stream so that we write in correct order
      std::stringstream ss;

      // we need to append current row and col values to the prefixes
      row_prefix.push_back(0);
      col_prefix.push_back(0);

      // do we need to write nested matrix blocks?
      if constexpr (Dune::blockLevel<typename Mat::block_type>() == 0) {
        // simple case: write svg block content to stream for each value
        for_each_entry([&](const auto &row, const auto &col, const auto &val) {
          std::size_t x_off = interspace + col * (interspace + block_size);
          std::size_t y_off = interspace + row * (interspace + block_size);
          row_prefix.back() = row;
          col_prefix.back() = col;
          opts.writeSVGBlock(ss, row_prefix, col_prefix, val,
                            {x_off, y_off, block_size, block_size});
        });
        col_offset += cols * (block_size + interspace);
        row_offset += rows * (block_size + interspace);
      } else {
        // before we write anything, we need to calculate the
        // offset for every {row,col} index
        const auto null_offset = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> col_offsets(cols + 1, null_offset);
        std::vector<std::size_t> row_offsets(rows + 1, null_offset);
        for_each_entry([&](const auto &row, const auto &col, const auto &val) {
          NullStream dev0;
          // get size of sub-block
          auto sub_size =
              writeSVGMatrix(val, dev0, opts, row_prefix, col_prefix);

          // if we didn't see col size before
          if (col_offsets[col + 1] == null_offset) // write it in the offset vector
            col_offsets[col + 1] = sub_size.first;

          // repeat proces for row sizes
          if (row_offsets[row + 1] == null_offset)
            row_offsets[row + 1] = sub_size.second;
        });

        // if some rows/cols were not visited, make an educated guess with the minimum offset
        auto min_row_offset = *std::min_element(begin(row_offsets), end(row_offsets));
        std::replace(begin(row_offsets), end(row_offsets), null_offset, min_row_offset);
        auto min_col_offset = *std::min_element(begin(col_offsets), end(col_offsets));
        std::replace(begin(col_offsets), end(col_offsets), null_offset, min_col_offset);

        // we have sizes for every block: to get offsets we make a partial sum
        col_offsets[0] = interspace;
        row_offsets[0] = interspace;
        for (std::size_t i = 1; i < col_offsets.size(); i++)
          col_offsets[i] += col_offsets[i - 1] + interspace;
        for (std::size_t i = 1; i < row_offsets.size(); i++)
          row_offsets[i] += row_offsets[i - 1] + interspace;

        for_each_entry([&](const auto &row, const auto &col, const auto &val) {
          // calculate svg view from offsets
          std::size_t width =
              col_offsets[col + 1] - col_offsets[col] - interspace;
          std::size_t height =
              row_offsets[row + 1] - row_offsets[row] - interspace;
          row_prefix.back() = row;
          col_prefix.back() = col;
          // content of the sub-block has origin at {0,0}: shift it to the correct place
          ss << "<svg x='" << col_offsets[col] << "' y='" << row_offsets[row]
            << "' width='" << width << "' height='" << height << "'>\n";
          // write a nested svg with the contents of the sub-block
          writeSVGMatrix(val, ss, opts, row_prefix, col_prefix);
          ss << "</svg>\n";
        });
        col_offset = col_offsets.back();
        row_offset = row_offsets.back();
      }

      // write content in order!
      // (i) if required, first header
      if (write_header)
        writeSVGMatrixHeader(out, opts, {col_offset, row_offset});

      col_prefix.pop_back();
      row_prefix.pop_back();
      // (ii) an svg block for this level
      opts.writeSVGBlock(out, row_prefix, col_prefix, mat,
                        {0, 0, col_offset, row_offset});
      // (iii) the content of the matrix
      out << ss.str();
      // (iv) if required, close the header
      if (write_header)
        out << "</g>\n</svg>\n";

      // return the total required for this block
      return {col_offset, row_offset};
    }
  } // namespace Impl


  /**
   * @brief Default options class to write SVG matrices
   *
   * This object is intended customize the output of the SVG writer for matrices
   * \ref writeSVGMatrix.
   */
  struct DefaultSVGMatrixOptions {
    //! size (pixels) of the deepst block/value of the matrix
    std::size_t block_size = 10;
    //! size (pixels) of the interspace between blocks
    std::size_t interspace = 5;
    //! Final width size (pixels) of the SVG header. If 0, size is automatic.
    std::size_t width = 500;
    //! Final height size (pixels) of the SVG header. If 0, size is automatic.
    std::size_t height = 0;
    //! Whether to write the SVG header
    bool write_header = true;
    //! CSS style block to write in header
    std::string style = " .matrix-block {\n"
                        "   fill: cornflowerblue;\n"
                        "   fill-opacity: 0.4;\n"
                        "   stroke-width: 2;\n"
                        "   stroke: black;\n"
                        "   stroke-opacity: 0.5;\n"
                        " }\n"
                        " .matrix-block:hover {\n"
                        "   fill: lightcoral;\n"
                        "   fill-opacity: 0.4;\n"
                        "   stroke-opacity: 1;\n"
                        " }\n";

    /**
     * @brief Color fill for default options
     *
     * Example:
     * @code{.cc}
     * opts.color_fill = [max,min](const double& val){
     *   auto percentage = (val-min)/(max-min)*100;
     *   return "hsl(348, " + std::to_string(precentage) + "%, 41%)";};
     * };
     * @endcode
     */
    std::function<std::string(const double&)> color_fill;

    /**
     * @brief Helper function that returns an style class for a given prefix
     * @note This function is only a helper to the default writeSVGBlock and is
     * not required for custom options classes.
     */
    template <class RowPrefix, class ColPrefix>
    std::string blockStyleClass(const RowPrefix &row_prefix,
                                const ColPrefix &col_prefix) const {
      // here, you can potentially give a different style to each block
      return "matrix-block";
    }

    //! (Helper) Whether to write a title on the rectangle value
    bool write_block_title = true;

    /**
     * @brief Helper function writes a title for a given block and prefix
     * @note This function is only a helper to the default writeSVGBlock and is
     * not required for custom options classes.
     */
    template <class Stream, class RowPrefix, class ColPrefix, class Block>
    void writeBlockTitle(Stream& out, const RowPrefix &row_prefix,
                                const ColPrefix &col_prefix,
                                const Block &block) const {
      if (this->write_block_title) {
        out << "<title>";
        assert(row_prefix.size() == col_prefix.size());
        for (std::size_t i = 0; i < row_prefix.size(); ++i)
          out << "[" << row_prefix[i] << ", "<< col_prefix[i] << "]";
        if constexpr (Dune::blockLevel<Block>() == 0)
          out << ": " << block;
        out << "</title>\n";
      }
    }

    /**
     * @brief  Write an SVG object for a given block/value in the matrix
     * @details This function is called for every matrix block; that includes
     *          root and intermediate nested blocks of matrices. SVG blocks are
     *          written from outer to inner matrix blocks, this means that in
     *          case of overlaps in the SVG objects, the more nested blocks take
     *          precedence.
     * @warning If the SVG bounding box is not respected, the content may be
     *          ommited in the final SVG view.
     * @note This function signature is required for any custom options class.
     *
     * @tparam Stream       An ostream type (possibly a NullStream!)
     * @tparam RowPrefix    A ReservedVector
     * @tparam ColPrefix    A ReservedVector
     * @tparam Block        The type of the current block
     * @param out           An stream to send SVG object to
     * @param row_prefix    A multindex of indices to access current row
     * @param col_prefix    A multindex of indices to access current column
     * @param block         The current matrix/sub-block/value
     * @param svg_box       SVG object bounding box (position and sizes).
     */
    template <class Stream, class RowPrefix, class ColPrefix, class Block>
    void writeSVGBlock(Stream &out,
                        const RowPrefix &row_prefix,
                        const ColPrefix &col_prefix, const Block block,
                        const std::array<std::size_t, 4> &svg_box) const {
      // get bounding box values
      auto &[x_off, y_off, width, height] = svg_box;
      // get style class
      std::string block_class = this->blockStyleClass(row_prefix, col_prefix);
      // write a rectangle on the bounding box
      out << "<rect class='" << block_class << "' x='" << x_off << "' y='"
          << y_off << "' width='" << width << "' height='" << height << "'";
      if constexpr (Dune::blockLevel<Block>() == 0 and std::is_convertible<Block,double>{})
        if (color_fill)
          out << " style='fill-opacity: 1;fill:" << color_fill(double(block)) << "'";

      out << ">\n";
      // give the rectangle a title (in html this shows info about the block)
      this->writeBlockTitle(out,row_prefix, col_prefix, block);
      // close rectangle
      out << "</rect>\n";
    }
  };

  /**
   * @brief Writes the visualization of matrix in the SVG format
   * @details The default visualization writes a rectangle for every block
   *          bounding box. This is enough to visualize patterns. If you need a
   *          more advance SVG object in each matrix block (e.g. color scale, or
   *          write the value in text), just provide a custom SVGOptions that
   *          fullfils the DefaultSVGMatrixOptions interface.
   *
   * @tparam Mat          Matrix type to write
   * @tparam SVGOptions   Options object type (see DefaultSVGMatrixOptions)
   * @param mat           The matrix to write
   * @param out           A output stream to write SVG to
   * @param opts          SVG Options object
   */
  template <class Mat, class SVGOptions = DefaultSVGMatrixOptions>
  void writeSVGMatrix(const Mat &mat, std::ostream &out,
                      SVGOptions opts = {}) {
    // We need a vector that can fit all the multi-indices for rows and colums
    using IndexPrefix = Dune::ReservedVector<std::size_t, blockLevel<Mat>()>;
    // Call overload for Mat type
    Impl::writeSVGMatrix(mat, out, opts, IndexPrefix{}, IndexPrefix{});
  }

  /** @} end documentation */

} // namespace Dune

#endif
