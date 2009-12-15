// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_MATRIXMARKET_HH
#define DUNE_MATRIXMARKET_HH

#include <ostream>
#include <limits>
#include "bcrsmatrix.hh"
#include <dune/common/fmatrix.hh>
#include <dune/common/tuples.hh>

namespace Dune
{
  namespace
  {
    /**
     * @brief Helper metaprogram to get
     * the matrix market string representation
     * of the numeric type.
     *
     * Member function mm_numeric_type::c_str()
     * returns the string representation of
     * the type.
     */
    template<class T>
    struct mm_numeric_type;

    template<>
    struct mm_numeric_type<int>
    {
      static std::string str()
      {
        return "integer";
      }
    };

    template<>
    struct mm_numeric_type<double>
    {
      static std::string str()
      {
        return "real";
      }
    };

    template<>
    struct mm_numeric_type<float>
    {
      static std::string str()
      {
        return "real";
      }
    };

    template<>
    struct mm_numeric_type<std::complex<double> >
    {
      static std::string str()
      {
        return "complex";
      }
    };

    template<>
    struct mm_numeric_type<std::complex<float> >
    {
      static std::string str()
      {
        return "complex";
      }
    };

    /**
     * @brief Meta program to write the correct Matrix Market
     * header.
     *
     * Member function mm_header::operator() writes the header.
     *
     * @tparam M The matrix type.
     */
    template<class M>
    struct mm_header_printer;

    template<typename T, typename A, size_t i, size_t j>
    struct mm_header_printer<BCRSMatrix<FieldMatrix<T,i,j>,A> >
    {
      static void print(std::ostream& os)
      {
        os<<"%%MatrixMarket matrix coordinate ";
        os<<mm_numeric_type<T>::str()<<" general"<<std::endl;
      }
    };

    template<typename T, size_t i, size_t j>
    struct mm_header_printer<FieldMatrix<T,i,j> >
    {
      static void print(std::ostream& os)
      {
        os<<"%%MatrixMarket matrix array ";
        os<<mm_numeric_type<T>::str()<<" general"<<std::endl;
      }
    };

    /**
     * @brief Metaprogram for writing the ISTL block
     * structure header.
     *
     * Member function mm_block_structure_header::operaor() writes
     * the corresponding header to the specified ostream.
     * @tparam The type of the matrix to generate the header for.
     */
    template<class M>
    struct mm_block_structure_header;

    template<typename T, typename A, size_t i, size_t j>
    struct mm_block_structure_header<BCRSMatrix<FieldMatrix<T,i,j>,A> >
    {
      typedef BCRSMatrix<FieldMatrix<T,i,j>,A> M;

      static void print(std::ostream& os, const M& m)
      {
        os<<"% ISTL_STRUCT blocked ";
        os<<i<<" "<<j<<std::endl;
      }
    };

    template<typename T, size_t i, size_t j>
    struct mm_block_structure_header<FieldMatrix<T,i,j> >
    {
      typedef FieldMatrix<T,i,j> M;

      static void print(std::ostream& os, const M& m)
      {}
    };


    enum LineType { MM_HEADER, MM_ISTLSTRUCT, DATA };
    enum { MM_MAX_LINE_LENGTH=1025 };

    enum MM_TYPE { coordinate_type, array_type, unknown_type };

    enum MM_CTYPE { integer_type, double_type, complex_type, pattern, unknown_ctype };

    enum MM_STRUCTURE { general, symmetric, skew_symmetric, hermitian, unknown_structure  };

    struct MMHeader
    {
      MMHeader()
        : type(coordinate_type), ctype(double_type), structure(general)
      {}
      MM_TYPE type;
      MM_CTYPE ctype;
      MM_STRUCTURE structure;
    };

    bool lineFeed(std::istream& file)
    {
      char c=file.peek();
      // ignore whitespace
      while(c==' ')
      {
        file.get();
        c=file.peek();
      }

      if(c=='\n') {
        /* eat the line feed */
        file.get();
        return true;
      }
      return false;
    }

    void skipComments(std::istream& file)
    {
      lineFeed(file);
      char c=file.peek();
      // ignore comment lines
      while(c=='%')
      {
        /* disgard the rest of the line */
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        c=file.peek();
      }
    }


    bool readMatrixMarketBanner(std::istream& file, MMHeader& mmHeader)
    {
      std::string buffer;
      char c;
      c=file.peek();
      mmHeader=MMHeader();
      if(c!='%')
        return false;

      /* read the banner */
      file >> buffer;
      if(buffer!="%%MatrixMarket") {
        /* disgard the rest of the line */
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        return false;
      }

      if(lineFeed(file))
        /* premature end of line */
        return false;

      /* read the matrix_type */
      file >> buffer;

      if(buffer != "matrix")
      {
        /* disgard the rest of the line */
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        return false;
      }

      if(lineFeed(file))
        /* premature end of line */
        return false;

      /* The type of the matrix */
      file >> buffer;

      if(buffer.empty())
        return false;

      std::transform(buffer.begin(), buffer.end(), buffer.begin(),
                     tolower);

      switch(buffer[0])
      {
      case 'a' :
        /* sanity check */
        if(buffer != "array")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.type=array_type;
        break;
      case 'c' :
        /* sanity check */
        if(buffer != "coordinate")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.type=coordinate_type;
        break;
      default :
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        return false;
      }

      if(lineFeed(file))
        /* premature end of line */
        return false;

      /* The numeric type used. */
      file >> buffer;

      if(buffer.empty())
        return false;

      std::transform(buffer.begin(), buffer.end(), buffer.begin(),
                     tolower);
      switch(buffer[0])
      {
      case 'i' :
        /* sanity check */
        if(buffer != "integer")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.ctype=integer_type;
        break;
      case 'r' :
        /* sanity check */
        if(buffer != "real")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.ctype=double_type;
        break;
      case 'c' :
        /* sanity check */
        if(buffer != "complex")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.ctype=complex_type;
        break;
      case 'p' :
        /* sanity check */
        if(buffer != "pattern")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.ctype=pattern;
        break;
      default :
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        return false;
      }

      if(lineFeed(file))
        return false;

      file >> buffer;

      std::transform(buffer.begin(), buffer.end(), buffer.begin(),
                     tolower);
      switch(buffer[0])
      {
      case 'g' :
        /* sanity check */
        if(buffer != "general")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.structure=general;
        break;
      case 'h' :
        /* sanity check */
        if(buffer != "hermitian")
        {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
        mmHeader.structure=hermitian;
        break;
      case 's' :
        if(buffer.size()==1) {
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }

        switch(buffer[1])
        {
        case 'y' :
          /* sanity check */
          if(buffer != "symmetric")
          {
            file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
            return false;
          }
          mmHeader.structure=symmetric;
          break;
        case 'k' :
          /* sanity check */
          if(buffer != "skew-symmetric")
          {
            file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
            return false;
          }
          mmHeader.structure=skew_symmetric;
          break;
        default :
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          return false;
        }
      default :
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        return false;
      }
      file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
      c=file.peek();
      return true;

    }

    void readNextLine(std::istream& file, std::ostringstream& line, LineType& type)
    {
      char c;
      std::size_t index=0;

      //empty lines will be disgarded and we will simply read the next line
      while(index==0&&!file.eof())
      {
        // strip spaces
        while(!file.eof() && (c=file.get())==' ') ;

        //read the rest of the line until comment
        while(!file.eof() && (c=file.get())=='\n') {
          switch(c)
          {
          case '%' :
            /* disgard the rest of the line */
            file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
          }
        }
      }

      //      buffer[index]='\0';
    }

    template<std::size_t brows, std::size_t bcols>
    Dune::tuple<std::size_t, std::size_t, std::size_t>
    calculateNNZ(std::size_t rows, std::size_t cols, std::size_t entries, const MMHeader& header)
    {
      std::size_t blockrows=rows/brows;
      std::size_t blockcols=cols/bcols;
      std::size_t blocksize=brows*bcols;
      std::size_t blockentries=0;

      switch(header.structure)
      {
      case general :
        blockentries = entries/blocksize; break;
      case skew_symmetric :
        blockentries = 2*entries/blocksize; break;
      case symmetric :
        blockentries = (2*entries-rows)/blocksize; break;
      case hermitian :
        blockentries = (2*entries-rows)/blocksize; break;
      default :
        throw Dune::NotImplemented();
      }
      return Dune::make_tuple(blockrows, blockcols, blockentries);
    }

    /*
     *   @brief Storage class for the column index and the numeric value.
     *
     * \tparam T Either a NumericWrapper of the numeric type or PatternDummy
     *     for MatrixMarket pattern case.
     */
    template<typename T>
    struct IndexData : public T
    {
      std::size_t index;
    };


    /**
     * @brief a wrapper class of numeric values.
     *
     * Use for template specialization to catch the pattern
     * type MatrixMarket matrices. Uses Empty base class optimization
     * in the pattern case.
     *
     * @tparam T Either a NumericWrapper of the numeric type or PatternDummy
     *     for MatrixMarket pattern case.
     */
    template<typename T>
    struct NumericWrapper
    {
      T number;
      operator T&()
      {
        return number;
      }
    };

    /**
     * @brief Utility class for marking the pattern type of the MatrixMarket matrices.
     */
    struct PatternDummy
    {};

    template<>
    struct NumericWrapper<PatternDummy>
    {};

    template<typename T>
    std::istream& operator>>(std::istream& is, NumericWrapper<T>& num)
    {
      return is>>num.number;
    }

    std::istream& operator>>(std::istream& is, NumericWrapper<PatternDummy>& num)
    {
      return is;
    }

    /**
     * @brief LessThan operator.
     *
     * It simply compares the index.
     */
    template<typename T>
    bool operator<(const IndexData<T>& i1, const IndexData<T>& i2)
    {
      return i1.index<i2.index;
    };

    /**
     * @brief Read IndexData from a stream.
     * @param is The input stream we read.
     * @param data Where to store the read data.
     */
    template<typename T>
    std::istream& operator>>(std::istream& is, IndexData<T>& data)
    {
      is>>data.index;
      /* MatrixMarket indices are one based. Decrement for C++ */
      --data.index;
      return is>>data.number;
    }

    /**
     * @brief Functor to the data values of the matrix.
     *
     * This is specialized for PatternDummy. The specialization does not
     * set anything.
     */
    template<typename D, int brows, int bcols>
    struct MatrixValuesSetter
    {
      /**
       * @brief Sets the matrixvalues.
       * @param row The row data as read from file.
       * @param matrix The matrix whose data we set.
       */
      template<typename M>
      void operator()(const std::vector<std::set<IndexData<D> > >& rows,
                      M& matrix)
      {
        for(typename M::RowIterator iter=matrix.begin();
            iter!= matrix.end(); ++iter)
        {
          for(typename M::size_type brow=iter.index()*brows,
              browend=iter.index()*brows+brows;
              brow<browend; ++brow)
          {
            typedef typename std::set<IndexData<D> >::const_iterator Siter;
            for(Siter siter=rows[brow].begin(), send=rows[brow].end();
                siter != send; ++siter)
              (*iter)[siter->index/bcols][brow%brows][siter->index%bcols]=siter->number;
          }
        }
      }
    };

    template<int brows, int bcols>
    struct MatrixValuesSetter<PatternDummy,brows,bcols>
    {
      template<typename M>
      void operator()(const std::vector<std::set<IndexData<PatternDummy> > >& rows,
                      M& matrix)
      {}
    };

    template<typename T, typename A, int brows, int bcols, typename D>
    void readSparseEntries(Dune::BCRSMatrix<Dune::FieldMatrix<T,brows,bcols>,A>& matrix,
                           std::istream& file, std::size_t entries,
                           const MMHeader& mmHeader, D)
    {
      typedef Dune::BCRSMatrix<Dune::FieldMatrix<T,brows,bcols>,A> Matrix;
      std::vector<std::set<IndexData<D> > > rows(matrix.N()*brows);

      for(entries; entries>0; --entries) {
        std::size_t row;
        IndexData<D> data;
        skipComments(file);
        file>>row;
        --row; // Index was 1 based.
        assert(row/bcols<matrix.N());
        file>>data;
        assert(data.index/bcols<matrix.M());
        rows[row].insert(data);
      }

      // TODO extend to capture the nongeneral cases.
      if(mmHeader.structure!= general)
        DUNE_THROW(Dune::NotImplemented, "Only general is supported right now!");

      // Setup the matrix sparsity pattern
      int nnz=0;
      for(typename Matrix::CreateIterator iter=matrix.createbegin();
          iter!= matrix.createend(); ++iter)
      {
        for(std::size_t brow=iter.index()*brows, browend=iter.index()*brows+brows;
            brow<browend; ++brow)
        {
          typedef typename std::set<IndexData<D> >::const_iterator Siter;
          for(Siter siter=rows[brow].begin(), send=rows[brow].end();
              siter != send; ++siter, ++nnz)
            iter.insert(siter->index/bcols);
        }
      }

      //Set the matrix values
      matrix=0;

      MatrixValuesSetter<D,brows,bcols> Setter;

      Setter(rows, matrix);
    }
  } // end anonymous namespace

  class MatrixMarketFormatError : public Dune::Exception
  {};

  /**
   * @brief Reads a sparse matrix from a matrix market file.
   * @param matrix The matrix to store the data in.
   * @param istr The input stream to read the data from.
   * @warning Not all formats are supported!
   */
  template<typename T, typename A, int brows, int bcols>
  void readMatrixMarket(Dune::BCRSMatrix<Dune::FieldMatrix<T,brows,bcols>,A>& matrix,
                        std::istream& istr)
  {

    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,brows,bcols> > Matrix;

    MMHeader header;
    if(!readMatrixMarketBanner(istr, header))
      std::cerr << "First line was not a correct Matrix Market banner. Using default:\n"
                << "%%MatrixMarket matrix coordinate real general"<<std::endl;

    skipComments(istr);

    std::size_t rows, cols, entries;

    if(lineFeed(istr))
      throw MatrixMarketFormatError();

    istr >> rows;

    if(lineFeed(istr))
      throw MatrixMarketFormatError();
    istr >> cols;

    if(lineFeed(istr))
      throw MatrixMarketFormatError();

    istr >>entries;

    std::size_t nnz, blockrows, blockcols;

    Dune::tie(blockrows, blockcols, nnz) = calculateNNZ<brows, bcols>(rows, cols, entries, header);

    istr.ignore(std::numeric_limits<std::streamsize>::max(),'\n');


    matrix.setSize(blockrows, blockcols);
    matrix.setBuildMode(Dune::BCRSMatrix<Dune::FieldMatrix<T,brows,bcols>,A>::row_wise);

    if(header.type==array_type)
      DUNE_THROW(Dune::NotImplemented, "currently not supported!");

    NumericWrapper<double> d;

    readSparseEntries(matrix, istr, entries, header,d);
  }
  template<typename M>
  void printMatrixMarket(M& matrix,
                         std::ostream ostr)
  {
    mm_header_printer<M>::print(ostr, matrix);
    mm_block_structure_header<M>::print(ostr,matrix);
  }

}
#endif
