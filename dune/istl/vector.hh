// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_VECTOR_HH
#define DUNE_ISTL_VECTOR_HH

#if 0

#include <cmath>
#include <complex>
#include <memory>
#include <limits>

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

    template<typename F_, typename A_>
    class Vector
    {
      private:
        size_type _size;

      public:
        typedef F_ Field;
        typedef F_::DataType DataType;
        typedef A_ Allocator;
        typedef A_ allocator_type;
        typedef A_::size_type size_type;
        typedef ... iterator;
        typedef ... const_iterator;

        Vector() :
          _size(0)
        {
          //TODO
        }

        Vector(size_type size) :
          _size(size)
        {
          //TODO
        }

        Vector(const Vector & other) :
          _size(other._size)
        {
          //TODO
        }

        Vector(Vector && other) :
          _size(other._size)
        {
          //TODO
        }

        size_type size() const
        {
          return _size;
        }

        Vector & operator= (const Vector & other)
        {
        }

        Vector & operator= (Vector && other)
        {
        }

        Field operator[] (size_type i)
        {
          return _data[i]
        }

        const Field operator[] (size_type i) const
        {
          return _data[i]
        }

        write(indices, values)
        {
        }

        read(indices, values)
        {
        }

        accumulate(indices, values)
        {
        }

        write(offset, count, values);
        read(offset, count, values);
        accumulate(offset, count, values);

        iterator begin()
        const_iterator begin() const
        iterator end()
        const_iterator end() const

        scale
        axpy
        dot
        normen

        setSize(size_type size)
        {
          if _size != 0 throw exception
          alloc(size)
        }

    };


  } // end namespace istl
} // end namespace dune

#endif // 0

#endif // DUNE_ISTL_VECTOR_HH
