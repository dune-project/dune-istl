// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BLOCKVECTOR_HOST_BLOCK_HH
#define DUNE_ISTL_BLOCKVECTOR_HOST_BLOCK_HH

#include <limits>
#include <type_traits>
#include <algorithm>

#include <dune/common/static_assert.hh>
#include <dune/common/typetraits.hh>

#include <dune/common/iteratorfacades.hh>

namespace Dune {
  namespace ISTL {
    namespace blockvector {


    template<typename V, typename T>
    struct Block
    {

      typedef T value_type;
      typedef typename V::size_type size_type;

      static const size_type simd_block_size = V::simd_block_size;

      typedef value_type* iterator;
      typedef typename std::add_const<value_type>::type* const_iterator;


      value_type& operator[](size_type i)
      {
        return _data[i];
      }

      const value_type& operator[](size_type i) const
      {
        return _data[i];
      }

      size_type size() const
      {
        return _size;
      }

      iterator begin()
      {
        return _data;
      }

      iterator end()
      {
        return _data + _size;
      }

      const_iterator begin() const
      {
        return _data;
      }

      const_iterator end() const
      {
        return _data + _size;
      }

      Block(value_type* data, size_type size)
        : _data(data)
        , _size(size)
      {}

    private:

      value_type* _data;
      size_type _size;

    };


    template<typename V, typename T>
    struct BlockIterator
      : public RandomAccessIteratorFacade<BlockIterator<V,T>,
                                          Dune::ISTL::blockvector::Block<V,T>,
                                          Dune::ISTL::blockvector::Block<V,T>,
                                          typename V::Allocator::difference_type
                                          >
    {

      friend class RandomAccessIteratorFacade<
        BlockIterator,
        Dune::ISTL::blockvector::Block<V,T>,
        Dune::ISTL::blockvector::Block<V,T>,
        typename V::Allocator::difference_type
        >;

      typedef V Vector;

    public:

      typedef typename V::size_type size_type;
      typedef Dune::ISTL::blockvector::Block<V,T> value_type;
      typedef value_type Block;
      typedef value_type reference;
      typedef typename V::Allocator::difference_type difference_type;

      static const size_type simd_block_size = V::simd_block_size;

      size_type index() const
      {
        return _index;
      }

    private:
      // keep this public for now, access from the facade is a mess!
    public:

      reference dereference() const
      {
        return Block(_data + _index * _block_size,_block_size);
      }

      reference elementAt(size_type n) const
      {
        return Block(_data + _index * _block_size,_block_size);
      }

      bool equals(const BlockIterator& other) const
      {
        return _index == other._index;
      }

      void increment()
      {
        ++_index;
      }

      void decrement()
      {
        --_index;
      }

      void advance(difference_type n)
      {
        _index += n;
      }

      difference_type distanceTo(const BlockIterator& other) const
      {
        return _index - other._index;
      }

      BlockIterator(T* data, size_type index, size_type block_size)
        : _data(data)
        , _index(index)
        , _block_size(block_size)
      {}

      T* _data;
      size_type _index;
      size_type _block_size;

    };

    } // end namespace blockvector
  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_BLOCKVECTOR_HOST_BLOCK_HH
