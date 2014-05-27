// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_VECTOR_HOST_HH
#define DUNE_ISTL_VECTOR_HOST_HH

#include <cmath>
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
#include <dune/common/memory/alignment.hh>
#include <dune/common/memory/traits.hh>
#include <dune/common/kernel/vec.hh>

#include <dune/common/threads/range.hh>

#include <dune/istl/forwarddeclarations.hh>

namespace Dune {
  namespace ISTL {

    template<typename F_, typename A_>
    class Vector<F_,A_,Memory::Domain::Host>
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
      typedef value_type* iterator;
      typedef const value_type* const_iterator;

      static const size_type kernel_block_size = Allocator::block_size;
      static const size_type alignment = Allocator::alignment;
      static const size_type minimum_chunk_size = Allocator::minimum_chunk_size;

      typedef Threads::fixed_block_size_range<size_type> range_type;

    public:

      Vector()
        : _size(0)
        , _allocation_size(0)
        , _data(nullptr)
        , _chunk_size(minimum_chunk_size)
      {}

      explicit Vector(size_type size)
        : _size(0)
        , _allocation_size(0)
        , _data(nullptr)
        , _chunk_size(minimum_chunk_size)
      {
        allocate(size);
      }

      Vector(const Vector & other)
        : _size(0)
        , _allocation_size(0)
        , _data(nullptr)
        , _chunk_size(other._chunk_size)
      {
        allocate(other._size,false);
        // TODO: check whether compiler optimizes to memcpy for trivially_copyable types
        std::uninitialized_copy(other._data,other._data+_size,_data);
      }

      Vector(Vector && other)
        : _size(other._size)
        , _allocation_size(other._allocation_size)
        , _allocator(std::move(other._allocator))
        , _data(other._data)
        , _chunk_size(other._chunk_size)
      {
        other._data = nullptr;
        other._size = 0;
        other._allocation_size = 0;
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

      Vector & operator= (const Vector& other)
      {
        if (_size == other._size)
          {
            std::copy(other._data,other._data+_size,_data);
            // don't copy allocator because we keep the old memory!
            _chunk_size = other._chunk_size;
            return *this;
          }
        else
          {
            if (_data)
              deallocate();
            _allocator = other._allocator;
            if (other._size == 0)
              return *this;
            allocate(other._size,false);
            std::uninitialized_copy(other._data,other._data+_size,_data);
            _chunk_size = other._chunk_size;
          }
        return *this;
      }

      Vector & operator= (Vector&& other)
      {
        if (_data)
          deallocate();
        _size = other._size;
        _allocation_size = other._allocation_size;
        _allocator = std::move(other._allocator);
        _data = other._data;
        _chunk_size = other._chunk_size;
        other._data = nullptr;
        other._size = 0;
        other._allocation_size = 0;
        return *this;
      }

      Field& operator[] (size_type i)
      {
        return _data[i];
      }

      const Field& operator[] (size_type i) const
      {
        return _data[i];
      }

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

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        Vector
        >::type&
      operator+=(const Vector<OF,OA,Domain>& other)
      {
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
                _data+r.begin(),
                other._data+r.begin(),
                r.block_count());
          });
        return *this;
      }

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        Vector
        >::type&
      operator-=(const Vector<OF,OA,Domain>& other)
      {
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
                _data+r.begin(),
                other._data+r.begin(),
                r.block_count());
          });
        return *this;
      }

      Vector& operator=(value_type b)
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
                _data+r.begin(),
                b,
                r.block_count());
          });
        return *this;
      }


      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        Vector
        >::type&
      operator*=(const Vector<OF,OA,Domain>& other)
      {
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
                _data+r.begin(),
                other._data+r.begin(),
                r.block_count());
          });
        return *this;
      }

      Vector& operator*=(value_type b)
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
                _data+r.begin(),
                b,
                r.block_count());
          });
        return *this;
      }

      Vector& operator/=(value_type b)
      {
        return this->operator*=(value_type(1.0)/b);
      }

      Vector& operator+=(value_type b)
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
                _data+r.begin(),
                b,
                r.block_count());
          });
        return *this;
      }

      Vector& operator-=(value_type b)
      {
        return this->operator+=(-b);
      }


      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        Vector
        >::type&
      axpy(value_type a, const Vector<OF,OA,Domain>& other)
      {
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
                _data+r.begin(),
                a,
                other._data+r.begin(),
                r.block_count());
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
      dot(const Vector<OF,OA,Domain>& other) const
      {
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
                _data+r.begin(),
                other._data+r.begin(),
                r.block_count());
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
                _data+r.begin(),
                r.block_count());
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
                _data+r.begin(),
                r.block_count());
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
                  _data+r.begin(),
                  r.block_count())
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


      ~Vector()
      {
        deallocate();
      }


      value_type* data() const
      {
        return _data;
      }

      Allocator& allocator()
      {
        return _allocator;
      }

      range_type iteration_range() const
      {
        return range_type(0,_size,kernel_block_size,(_chunk_size > minimum_chunk_size ? _chunk_size : minimum_chunk_size),minimum_chunk_size/kernel_block_size);
      }

      size_type blockSize() const
      {
        return 1;
      }

      template<typename Archive>
      void archive(Archive& ar)
      {
        ar & _size;
        ar & _allocation_size;
        ar & _chunk_size;
        if (Archive::Traits::is_reading)
          allocate(_size,false);
        ar.bulk(_data,_allocation_size);
      }

      bool operator==(const Vector& other) const
      {
        return
          _size == other._size &&
          _allocation_size == other._allocation_size &&
          std::equal(_data,_data + _allocation_size,other._data);
      }

      bool operator!=(const Vector& other) const
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
                    for (size_type i = r.begin(), end = r.end(); i != end; ++i)
                      _allocator.destroy(a+i);
                  });
              }
            _allocator.deallocate(_data,_allocation_size);
            _data = nullptr;
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
        _data = _allocator.allocate(allocation_size);
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
                for (size_type i = r.begin(), end = r.end(); i != end; ++i)
                  _allocator.construct(a+i);
              });
          }
        // always initialize padded memory area
        for (size_type i = size, end = allocation_size; i != end; ++i)
          _allocator.construct(_data+i);
      }

      size_type _size;
      size_type _allocation_size;
      Allocator _allocator;
      value_type* _data;
      size_type _chunk_size;

    };

  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_VECTOR_HOST_HH
