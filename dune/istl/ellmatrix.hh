
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ELLMATRIX_HH
#define DUNE_ISTL_ELLMATRIX_HH

#include <cmath>
#include <complex>
#include <memory>
#include <limits>

#include <dune/common/static_assert.hh>
#include <dune/common/promotiontraits.hh>
#include <dune/common/dotproduct.hh>

#include "istlexception.hh"
#include "basearray.hh"

/*! \file

   \brief  This file implements a vector space as a tensor product of
   a given vector space. The number of components can be given at
   run-time.
 */

namespace Dune {
  namespace ISTL {

    // F_: DT-Field, A_: Allocator
    template<typename F_, typename A_>
    class ELLMatrix
    {
      struct Layout
      {
        shared_ptr<LayoutData> data;
        Layout();
        Layout(size_type rows, size_type cols, size_type nonzeroes, size_type chunk_size, size_type * chunk_length, size_type * col_index)
      };
      struct LayoutData
      {
      };

      public:
        typedef F_ Field;
        typedef F_::DataType DataType;
        typedef A_ Allocator;
        typedef A_ allocator_type;
        typedef A_::size_type size_type;
        typedef ... iterator;
        typedef ... const_iterator;

      private:
        Layout _layout;
        DataType * values;

      public:
        ELLMatrix() :
          _rows(0),
          _cols(0),
          _nonzeroes(0)
        {
          //TODO
        }

        ELLMatrix(ELLLayout layout)
        {
          //TODO
        }

        ELLMatrix(const ELLMatrix & other) :
          _size(other._size)
        {
          //TODO
        }

        ELLMatrix(ELLMatrix && other) :
          _size(other._size)
        {
          //TODO
        }

        ELLLayout layout()
        {
        }

        size_type rows() const
        {
          return _rows;
        }

        size_type cols() const
        {
          return _cols;
        }

        size_type nonzeroes() const
        {
          return _nonzeroes;
        }

        ELLMatrix & operator= (const ELLMatrix & other)
        {
        }

        ELLMatrix & operator= (ELLMatrix && other)
        {
        }

        Row operator[] (size_type i)
        {
          return Row(i)
        }

        const Row operator[] (size_type i) const
        {
          return Row[i]
        }

        Field operator() (size_type row, size_type col)
        {
        }

        const Field operator() (size_type row, size_type col) const
        {
        }

        const Field element(size_type row, size_type col) const
        {
          // no exception, but _zero_element
        }

        write(row_indices, col_indices, values)
        {
        }

        read(row_indices, col_indices, values)
        {
        }

        accumulate(row_indices, col_indices, values)
        {
        }

        write(row_offset, row_size, col_offset, col_size, count, values);
        read(row_offset, row_size, col_offset, col_size, count, values);
        accumulate(row_offset, row_size, col_offset, col_size, count, values);

        iterator begin()
        const_iterator begin() const
        iterator end()
        const_iterator end() const

        row_iterator row_begin()
        const_row_iterator begin() const
        row_iterator end()
        const_row_iterator end()

        // http://www.dune-project.org/doc/doxygen/html/classDune_1_1BCRSMatrix.html
        scale
        normen
        axpy
        mv
        umv
        mmv
        usmv
        mtv
        umtv
        mmtv
        usmtv
        umhv
        mmhv
        usmhv


        setLayout(Layout layout)
        {
          if ! _layout.unset() throw exception
            _layout = layout;
          malloc data
        }

    };


  } // end namespace istl
} // end namespace dune

#endif
