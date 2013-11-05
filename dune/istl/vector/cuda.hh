// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_VECTOR_CUDA_HH
#define DUNE_ISTL_VECTOR_CUDA_HH

#include <cmath>
#include <memory>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <functional>

#include <dune/common/static_assert.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/promotiontraits.hh>
#include <dune/common/dotproduct.hh>
#include <dune/common/memory/cuda_allocator.hh>
#include <dune/common/kernel/vec/cuda_kernels.hh>
#include <dune/istl/forwarddeclarations.hh>

#include <dune/common/memory/blocked_allocator.hh>
#include <dune/istl/vector/host.hh>


namespace Dune {
  namespace ISTL {

    template<typename F_, typename A_>
    class Vector<F_,A_,Memory::Domain::CUDA>
    {
      public:
      typedef F_ Field;
      typedef F_ DataType;
      typedef F_ value_type;
      typedef F_ field_type;

      typedef Memory::Domain::CUDA Domain;

      // typedef F_::DataType DataType;
      typedef F_ DT_;

      typedef typename A_::template rebind<F_>::other Allocator;
      typedef Allocator allocator_type;
      typedef typename A_::size_type size_type;
      typedef value_type* iterator;
      typedef const value_type* const_iterator;

      private:
      size_type _size;
      Allocator _allocator;
      value_type* _data;

      public:

      Vector()
        : _size(0)
        , _data(nullptr)
      {}

       explicit Vector(size_type size)
        : _size(0)
        , _data(nullptr)
      {
        allocate(size);
      }

      explicit Vector(size_type size, value_type val)
        : _size(0)
        , _data(nullptr)
      {
        allocate(size, false);
        Cuda::set(_data, val, size);
      }

      Vector(const Vector & other)
        : _size(0)
        , _data(nullptr)
      {
        allocate(other._size, false);
        Cuda::copy(_data, other._data, _size);
      }

      Vector(Vector && other)
        : _size(other._size)
        , _allocator(std::move(other._allocator))
        , _data(other._data)
      {
        other._data = nullptr;
        other._size = 0;
      }

      template <size_t blocksize_>
      Vector(const Vector<DT_, Dune::Memory::blocked_cache_aligned_allocator<F_,std::size_t, blocksize_> > & other)
        : _size(0)
        , _data(nullptr)
      {
        allocate(other.size(), false);
        for (size_t i(0) ; i < _size ; ++i)
          Cuda::set(_data + i, other[i]);
      }

      size_type size() const
      {
        return _size;
      }

      Vector & operator= (const Vector & other)
      {
        if (_size == other._size)
        {
          Cuda::copy(_data, other._data, _size);
        }
        else
        {
          if (_data)
            deallocate();
          _allocator = other._allocator;
          if (other._size == 0)
            return *this;
          allocate(other._size ,false);
          Cuda::copy(_data, other._data, _size);
        }
        return *this;
      }

      Vector & operator= (Vector && other)
      {
        if (_data)
          deallocate();
        _size = other._size;
        _allocator = std::move(other._allocator);
        _data = other._data;
        other._data = nullptr;
        other._size = 0;
      }

      template <size_t blocksize_>
      Vector & operator= (const Vector<DT_, Dune::Memory::blocked_cache_aligned_allocator<F_,std::size_t, blocksize_> > & other)
      {
        if (_size == other.size())
        {
          for (size_t i(0) ; i < _size ; ++i)
            Cuda::set(_data + i, other[i]);
        }
        else
        {
          if (_data)
            deallocate();
          if (other.size() == 0)
            return *this;
          allocate(other.size() ,false);
          for (size_t i(0) ; i < _size ; ++i)
            Cuda::set(_data + i, other[i]);
        }
        return *this;
      }

      template <size_t blocksize_>
      Vector<DT_, Dune::Memory::blocked_cache_aligned_allocator<F_,std::size_t, blocksize_> > & download_to(Vector<DT_, Dune::Memory::blocked_cache_aligned_allocator<F_,std::size_t, blocksize_> > & other)
       {
         if (_size != other.size())
           DUNE_THROW(Exception,"download: vector size missmatch!");

         DT_ * temp = new DT_[_size];
         Cuda::download(temp, _data, _size);
         for (size_type i(0) ; i != _size ; ++i)
           other[i] = temp[i];
         delete[] temp;

         return other;
       }

      /*Field operator[] (size_type i)
      {
        DUNE_THROW(Exception,"not implemented");
        return _data[i];
      }*/

      const DataType operator[] (size_type i) const
      {
        return Cuda::get(_data + i);
      }

      const DataType operator() (size_type i) const
      {
        return Cuda::get(_data + i);
      }

      void operator() (size_type i, DataType val)
      {
        Cuda::set(_data + i, val);
      }

      template<typename Indices>
      void read(const Indices& indices, DT_ * values) const
      {
        DT_ * temp = new DT_[_size];
        Cuda::download(temp, _data, _size);
        for (size_type i = 0, end = indices.size(); i != end; ++i)
           values[i] = temp[indices[i]];
        delete[] temp;
      }

      template<typename Indices, typename Values>
      void write(const Indices& indices, const Values& values)
      {
        DT_ * temp = new DT_[_size];
        Cuda::download(temp, _data, _size);
        for (size_type i = 0, end = indices.size(); i != end; ++i)
          temp[indices[i]] = values[i];
        Cuda::upload(_data, temp, _size);
        delete[] temp;
      }

      template<typename Indices, typename Values>
      void accumulate(const Indices& indices, const Values& values)
      {
        DT_ * temp = new DT_[_size];
        Cuda::download(temp, _data, _size);
        for (size_type i = 0, end = indices.size(); i != end; ++i)
          temp[indices[i]] += values[i];
        Cuda::upload(_data, temp, _size);
        delete[] temp;
      }

      template<typename Values>
      void read(size_type offset, size_type count, Values& values) const
      {
        DT_ * temp = new DT_[_size];
        Cuda::download(temp, _data, _size);
        for (size_type i = 0, o = offset; i != count; ++i, ++o)
          values[i] = temp[o];
        delete[] temp;
      }

      template<typename Values>
      void write(size_type offset, size_type count, const Values& values)
      {
        DT_ * temp = new DT_[_size];
        Cuda::download(temp, _data, _size);
        for (size_type i = 0, o = offset; i != count; ++i, ++o)
          temp[o] = values[i];
        Cuda::upload(_data, temp, _size);
        delete[] temp;
      }

      template<typename Values>
      void accumulate(size_type offset, size_type count, const Values& values)
      {
        DT_ * temp = new DT_[_size];
        Cuda::download(temp, _data, _size);
        for (size_type i = 0, o = offset; i != count; ++i, ++o)
          temp[o] += values[i];
        Cuda::upload(_data, temp, _size);
        delete[] temp;
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

      //TODO size checks
      Vector & operator+=(const Vector & b)
      {
        Cuda::sum(_data, _data, b.begin(), _size);
        return *this;
      }

      Vector & operator-=(const Vector & b)
      {
        Cuda::difference(_data, _data, b.begin(), _size);
        return *this;
      }

      Vector & operator*=(const Vector & b)
      {
        Cuda::element_product(_data, _data, b.begin(), _size);
        return *this;
      }

      Vector & operator/=(const Vector & b)
      {
        Cuda::element_division(_data, _data, b.begin(), _size);
        return *this;
      }

      Vector & operator+=(value_type b)
      {
        Cuda::sum_scalar(_data, _data, b, _size);
        return *this;
      }

      Vector & operator-=(value_type b)
      {
        Cuda::difference_scalar(_data, _data, b, _size);
        return *this;
      }

      Vector & operator*=(value_type b)
      {
        Cuda::product_scalar(_data, _data, b, _size);
        return *this;
      }

      Vector & operator/=(value_type b)
      {
        Cuda::division_scalar(_data, _data, b, _size);
        return *this;
      }

      Vector & axpy(value_type a, const Vector & b)
      {
        Cuda::axpy(_data, _data, a, b.begin(), _size);
        return *this;
      }

      value_type dot(const Vector & b) const
      {
        return Cuda::dot(_data, b.begin(), _size);
      }

      value_type two_norm2() const
      {
        DT_ r(Cuda::two_norm2(_data, _size));
        return r*r;
      }

      value_type two_norm() const
      {
        return Cuda::two_norm2(_data, _size);
      }

      value_type one_norm() const
      {
        return Cuda::one_norm(_data, _size);
      }

      value_type infinity_norm() const
      {
        return Cuda::infinity_norm(_data, _size);
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

      private:
      void deallocate()
      {
        if (_data)
        {
          if (!std::is_trivial<value_type>::value)
          {
            DUNE_THROW(NotImplemented, "do not use non trivial types with cuda!");
          }
          _allocator.deallocate(_data,_size);
          _data = nullptr;
          _size = 0;
        }
      }

      void allocate(size_type size, bool init = true)
      {
        if (_data)
          DUNE_THROW(NotImplemented,"do not reallocate memory");
        _data = _allocator.allocate(size);
        if (!_data)
          DUNE_THROW(Exception,"could not allocate memory");
        _size = size;
        if (init && !std::is_trivial<value_type>::value)
        {
          DUNE_THROW(NotImplemented, "do not use non trivial types with cuda!");
        }
      }
    };
  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_VECTOR_CUDA_HH
