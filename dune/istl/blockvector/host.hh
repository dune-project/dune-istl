// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BLOCKVECTOR_HOST_HH
#define DUNE_ISTL_BLOCKVECTOR_HOST_HH

#include <cmath>
#include <cassert>
#include <memory>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <functional>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/cache_aligned_allocator.h>

#include <dune/common/static_assert.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/promotiontraits.hh>
#include <dune/common/dotproduct.hh>
#include <dune/common/memory/domain.hh>
#include <dune/common/memory/alignment.hh>
#include <dune/common/memory/traits.hh>
#include <dune/common/kernel/vec.hh>

#include <dune/common/threads/range.hh>

#include <dune/istl/forwarddeclarations.hh>
#include <dune/istl/blockvector/host/block.hh>

namespace Dune {
  namespace ISTL {


    template<typename F_, typename A_>
    class BlockVector<F_,A_,Memory::Domain::Host>
    {

    public:
      typedef F_ Field;
      typedef F_ DataType;
      typedef F_ value_type;
      typedef F_ field_type;

      typedef Memory::Domain::Host Domain;

      // F_::DataType DataType;
      typedef F_ DF;

      typedef typename A_::template rebind<Field>::other Allocator;
      typedef Allocator allocator_type;
      typedef typename A_::size_type size_type;

      typedef blockvector::Block<BlockVector,DF> Block;
      typedef blockvector::Block<BlockVector,const DF> ConstBlock;

      typedef blockvector::BlockIterator<BlockVector,DF> iterator;
      typedef blockvector::BlockIterator<BlockVector,const DF> const_iterator;

      static const size_type kernel_block_size = Allocator::block_size;
      static const size_type alignment = Allocator::alignment;
      static const size_type minimum_chunk_size = Allocator::minimum_chunk_size;

      typedef Threads::fixed_block_size_range<size_type> range_type;

      BlockVector()
        : _block_size(0)
        , _size(0)
        , _allocation_size(0)
        , _data(nullptr)
        , _chunk_size(minimum_chunk_size)
      {}

      explicit BlockVector(size_type size, size_type block_size)
        : _block_size(block_size)
        , _size(0)
        , _allocation_size(0)
        , _data(nullptr)
        , _chunk_size(minimum_chunk_size)
      {
        allocate(size);
      }

      BlockVector(const BlockVector & other)
        : _block_size(other._block_size)
        , _size(0)
        , _allocation_size(0)
        , _data(nullptr)
        , _chunk_size(other._chunk_size)
      {
        allocate(other._size,false);
        // TODO: check whether compiler optimizes to memcpy for trivially_copyable types
        std::uninitialized_copy(other._data,other._data + _size * _block_size,_data);
      }

      BlockVector(BlockVector && other)
        : _block_size(other._block_size)
        , _size(other._size)
        , _allocation_size(other._allocation_size)
        , _allocator(std::move(other._allocator))
        , _data(other._data)
        , _chunk_size(other._chunk_size)
      {
        other._data = nullptr;
        other._block_size = 0;
        other._size = 0;
        other._allocation_size = 0;
      }

      size_type blockSize() const
      {
        return _block_size;
      }

      size_type size() const
      {
        return _size;
      }

      void setChunkSize(size_type chunk_size)
      {
        if (chunk_size < minimum_chunk_size)
          DUNE_THROW(Exception,"chunk size too small (" << chunk_size << " < " << minimum_chunk_size << ")");
        _chunk_size = chunk_size;
      }

      size_type chunkSize() const
      {
        return _chunk_size;
      }

      BlockVector & operator= (const BlockVector& other)
      {
        if (_size == other._size && _block_size == other._block_size)
          {
            std::copy(other._data,other._data + _size * _block_size,_data);
            // don't copy allocator because we keep the old memory!
            _chunk_size = other._chunk_size;
            return *this;
          }
        else
          {
            if (_data)
              deallocate();
            _block_size = other._block_size;
            _allocator = other._allocator;
            if (other._size == 0)
              return *this;
            allocate(other._size,false);
            std::uninitialized_copy(other._data,other._data + _size * _block_size,_data);
            _chunk_size = other._chunk_size;
          }
        return *this;
      }

      BlockVector & operator= (BlockVector&& other)
      {
        if (_data)
          deallocate();
        _block_size = other._block_size;
        _size = other._size;
        _allocation_size = other._allocation_size;
        _allocator = std::move(other._allocator);
        _data = other._data;
        _chunk_size = other._chunk_size;
        other._data = nullptr;
        other._block_size = 0;
        other._size = 0;
        other._allocation_size = 0;
        return *this;
      }

      Block operator[] (size_type i)
      {
        return {_data + i * _block_size,_block_size};
      }

      ConstBlock operator[] (size_type i) const
      {
        return {_data + i * _block_size,_block_size};
      }

      Field& operator()(size_type block, size_type entry)
      {
        return _data[block * _block_size + entry];
      }

      const Field& operator()(size_type block, size_type entry) const
      {
        return _data[block * _block_size + entry];
      }

      /*
      template<typename Indices, typename Values>
      void read(const Indices& indices, Values& values) const
      {
        for (size_type i = 0, end = indices.size(); i != end; ++i)
           values[i] = _data[indices[i]];
      }

      template<typename Indices, typename Values>
      void write(const Indices& indices, const Values& values)
      {
        for (size_type i = 0, end = indices.size(); i != end; ++i)
          _data[indices[i]] = values[i];
      }

      template<typename Indices, typename Values>
      void accumulate(const Indices& indices, const Values& values)
      {
        for (size_type i = 0, end = indices.size(); i != end; ++i)
          _data[indices[i]] += values[i];
      }

      template<typename Values>
      void read(size_type offset, size_type count, Values& values) const
      {
        for (size_type i = 0, o = offset; i != count; ++i, ++o)
          values[i] = _data[o];
      }

      template<typename Values>
      void write(size_type offset, size_type count, const Values& values)
      {
        for (size_type i = 0, o = offset; i != count; ++i, ++o)
          _data[o] = values[i];
      }

      template<typename Values>
      void accumulate(size_type offset, size_type count, const Values& values)
      {
        for (size_type i = 0, o = offset; i != count; ++i, ++o)
          _data[o] += values[i];
      }
      */

      iterator begin()
      {
        return {_data,0,_block_size};
      }

      iterator end()
      {
        return {_data,_size,_block_size};
      }

      const_iterator begin() const
      {
        return {_data,0,_block_size};
      }

      const_iterator end() const
      {
        return {_data,_size,_block_size};
      }

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        BlockVector
        >::type&
      operator+=(const BlockVector<OF,OA,Domain>& other)
      {
        assert(_block_size == other._block_size);
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Kernel::vec::blocked::add<
              value_type,
              OF,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                other._data+r.begin() * _block_size,
                r.block_count() * _block_size);
          });
        return *this;
      }

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        BlockVector
        >::type&
      operator-=(const BlockVector<OF,OA,Domain>& other)
      {
        assert(_block_size == other._block_size);
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Kernel::vec::blocked::subtract<
              value_type,
              OF,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                other._data+r.begin() * _block_size,
                r.block_count() * _block_size);
          });
        return *this;
      }

      BlockVector& operator=(value_type b)
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Kernel::vec::blocked::assign_scalar<
              value_type,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                b,
                r.block_count() * _block_size);
          });
        return *this;
      }


      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        BlockVector
        >::type&
      operator*=(const BlockVector<OF,OA,Domain>& other)
      {
        assert(_block_size == other._block_size);
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Kernel::vec::blocked::mul<
              value_type,
              OF,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                other._data+r.begin() * _block_size,
                r.block_count() * _block_size);
          });
        return *this;
      }

      BlockVector& operator*=(value_type b)
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Kernel::vec::blocked::scale<
              value_type,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                b,
                r.block_count() * _block_size);
          });
        return *this;
      }

      BlockVector& operator/=(value_type b)
      {
        return this->operator*=(value_type(1.0)/b);
      }

      BlockVector& operator+=(value_type b)
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Kernel::vec::blocked::shift<
              value_type,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                b,
                r.block_count() * _block_size);
          });
        return *this;
      }

      BlockVector& operator-=(value_type b)
      {
        return this->operator+=(-b);
      }


      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        BlockVector
        >::type&
      axpy(value_type a, const BlockVector<OF,OA,Domain>& other)
      {
        assert(_block_size == other._block_size);
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::vec::blocked::axpy<
              value_type,
              value_type,
              OF,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                a,
                other._data+r.begin() * _block_size,
                r.block_count() * _block_size);
          });
        return *this;
      }

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        typename PromotionTraits<value_type,OF>::PromotedType
        >::type
      dot(const BlockVector<OF,OA,Domain>& other) const
      {
        assert(_block_size == other._block_size);
        typedef typename PromotionTraits<value_type,OF>::PromotedType result_type;
        return tbb::parallel_reduce(
          iteration_range(),
          result_type(0),
          [&](const range_type& r, result_type result) -> result_type
          {
            return result + Dune::Kernel::vec::blocked::dot<
              value_type,
              OF,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                other._data+r.begin() * _block_size,
                r.block_count() * _block_size);
          },
          std::plus<result_type>());
      }

      value_type two_norm2() const
      {
        return tbb::parallel_reduce(
          iteration_range(),
          DF(0),
          [&](const range_type& r, DF result) -> DF
          {
            return result + Dune::Kernel::vec::blocked::two_norm2<
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                r.block_count() * _block_size);
          },
          std::plus<DF>());
      }

      value_type two_norm() const
      {
        return std::sqrt(two_norm2());
      }

      value_type one_norm() const
      {
        return tbb::parallel_reduce(
          iteration_range(),
          DF(0),
          [&](const range_type& r, DF result) -> DF
          {
            return result + Dune::Kernel::vec::blocked::one_norm<
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin() * _block_size,
                r.block_count() * _block_size);
          },
          std::plus<DF>());
      }

      value_type infinity_norm() const
      {
        return tbb::parallel_reduce(
          iteration_range(),
          DF(0),
          [&](const range_type& r, DF result) -> DF
          {
            return std::max(
              result,
              Dune::Kernel::vec::blocked::infinity_norm<
                value_type,
                size_type,
                alignment,
                kernel_block_size>(
                  _data+r.begin() * _block_size,
                  r.block_count() * _block_size)
              );
          },
          [](const value_type& a, const value_type& b) -> DF
          {
            return std::max(a,b);
          });
      }

      void setSize(size_type size)
      {
        if (_data)
          DUNE_THROW(NotImplemented,"not allowed");
        allocate(size);
      }


      ~BlockVector()
      {
        deallocate();
      }


      value_type* data() const
      {
        return _data;
      }

      Allocator& allocator() const
      {
        return const_cast<Allocator&>(_allocator);
      }

      void swap(BlockVector& other)
      {
        using std::swap;
        swap(_block_size,other._block_size);
        swap(_size,other._size);
        swap(_allocation_size,other._allocation_size);
        swap(_allocator,other._allocator);
        swap(_data,other._data);
        swap(_chunk_size,other._chunk_size);
      }

      friend void swap(BlockVector& a, BlockVector& b)
      {
        a.swap(b);
      }

      range_type iteration_range() const
      {
        return range_type(0,_size,kernel_block_size,(_chunk_size > minimum_chunk_size ? _chunk_size : minimum_chunk_size),minimum_chunk_size/kernel_block_size);
      }

      template<typename Archive>
      void archive(Archive& ar)
      {
        ar & _block_size;
        ar & _size;
        ar & _allocation_size;
        ar & _chunk_size;
        if (Archive::Traits::is_reading)
          allocate(_size,false);
        ar.bulk(_data,_allocation_size * _block_size);
      }

      bool operator==(const BlockVector& other) const
      {
        return
          _block_size == other._block_size &&
          _size == other._size &&
          _allocation_size == other._allocation_size &&
          std::equal(_data,_data + _allocation_size,other._data);
      }

      bool operator!=(const BlockVector& other) const
      {
        return !operator==(other);
      }


    private:

      void deallocate()
      {
        if (_data)
          {
            if (!std::is_trivial<value_type>::value)
              {
                tbb::parallel_for(
                  iteration_range(),
                  [&](const range_type& r)
                  {
                    DF* __restrict__  a = _data;
                    for (size_type i = r.begin() * _block_size, end = r.end() * _block_size; i != end; ++i)
                      _allocator.destroy(a+i);
                  });
              }
            _allocator.deallocate(_data,_allocation_size * _block_size);
            _data = nullptr;
            _block_size = 0;
            _size = 0;
            _allocation_size = 0;
          }
      }

      void allocate(size_type size, bool init = true)
      {
        if (_data)
          DUNE_THROW(NotImplemented,"do not do this");
        // add padding to minimum chunk size
        size_type allocation_size = size + minimum_chunk_size - 1 - ((size-1) % minimum_chunk_size);
        _data = _allocator.allocate(allocation_size * _block_size);
        if (!_data)
          DUNE_THROW(Exception,"could not allocate memory");
        _allocation_size = allocation_size;
        _size = size;
        if (init && !std::is_trivial<value_type>::value)
          {
            tbb::parallel_for(
              iteration_range(),
              [&](const range_type& r)
              {
                DF* __restrict__  a = _data;
                for (size_type i = r.begin() * _block_size, end = r.end() * _block_size; i != end; ++i)
                  _allocator.construct(a+i);
              });
          }
        // always initialize padded memory area
        for (size_type i = size * _block_size, end = allocation_size * _block_size; i != end; ++i)
          _allocator.construct(_data+i);
      }

      size_type _block_size;
      size_type _size;
      size_type _allocation_size;
      Allocator _allocator;
      value_type* _data;
      size_type _chunk_size;

    };

  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_BLOCKVECTOR_HOST_HH
