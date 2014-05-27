// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ELLMATRIX_HOST_HH
#define DUNE_ISTL_ELLMATRIX_HOST_HH

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
#include <dune/common/memory/domain.hh>
#include <dune/common/memory/alignment.hh>
#include <dune/common/memory/traits.hh>
#include <dune/common/kernel/ell.hh>
#include <dune/common/kernel/vec.hh>

#include <dune/common/threads/range.hh>

#include <dune/istl/forwarddeclarations.hh>
#include <dune/istl/ellmatrix/layout.hh>
#include <dune/istl/ellmatrix/iterator.hh>

namespace Dune {
  namespace ISTL {

    template<typename F_, typename A_>
    class ELLMatrix<F_,A_,Memory::Domain::Host>
    {

      template<typename, typename>
      friend struct ellmatrix::RowIterator;

      template<typename, typename>
      friend struct ellmatrix::Row;

    public:
      typedef F_ Field;
      typedef F_ DataType;
      typedef F_ value_type;

      typedef Memory::Domain::Host Domain;

      // F_::DataType DataType;
      typedef F_ DF;

      typedef typename A_::template rebind<F_>::other Allocator;
      typedef typename A_::template rebind<typename Allocator::size_type>::other IndexAllocator;
      typedef Allocator allocator_type;
      typedef IndexAllocator index_allocator_type;
      typedef typename A_::size_type size_type;

      typedef ellmatrix::RowIterator<ELLMatrix,value_type> RowIterator;
      typedef ellmatrix::RowIterator<ELLMatrix,const value_type> ConstRowIterator;

      typedef ellmatrix::Row<ELLMatrix,value_type> Row;
      typedef ellmatrix::Row<ELLMatrix,const value_type> ConstRow;

      //typedef ellmatrix::Iterator<value_type> Iterator;
      //typedef ellmatrix::Iterator<value_type> iterator;

      //typedef ellmatrix::Iterator<const value_type> ConstIterator;
      //typedef ellmatrix::Iterator<const value_type> const_iterator;

      typedef ellmatrix::Layout<IndexAllocator> Layout;
      typedef ellmatrix::LayoutBuilder<IndexAllocator> LayoutBuilder;

      static const size_type kernel_block_size = Allocator::block_size;
      static const size_type alignment = Allocator::alignment;
      static const size_type minimum_chunk_size =
        (Allocator::minimum_chunk_size < IndexAllocator::minimum_chunk_size
         ? IndexAllocator::minimum_chunk_size
         : Allocator::minimum_chunk_size);

      static const size_type block_shift = Memory::block_size_log2<kernel_block_size,5>::value;
      static const size_type block_mask = (size_type(1) << block_shift) - 1;

      typedef Threads::fixed_block_size_range<size_type> range_type;

      ELLMatrix()
        : _data(nullptr)
        , _chunk_size(minimum_chunk_size)
        , _zero_element()
      {}

      ELLMatrix(Layout layout)
        : _layout(layout)
        , _data(nullptr)
        , _chunk_size(layout.chunkSize())
        , _nonzeros_chunk_size(layout.nonZerosChunkSize())
        , _zero_element()
      {
        allocate();
      }

      void setLayout(Layout layout)
      {
        _layout = layout;
        _chunk_size = layout.chunkSize();
        _nonzeros_chunk_size = layout.nonZerosChunkSize();
        allocate();
      }

      const Layout& layout() const
      {
        return _layout;
      }

      /*
      ELLMatrix(const ELLMatrix& other)
        : _size(0)
        , _allocation_size(0)
        , _data(nullptr)
        , _chunk_size(other._chunk_size)
      {
        allocate(other._size,false);
        // TODO: check whether compiler optimizes to memcpy for trivially_copyable types
        std::uninitialized_copy(other._data,other._data+_size,_data);
      }

      ELLMatrix(ELLMatrix&& other)
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
      */

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

      size_type rows() const
      {
        return _layout.rows();
      }

      size_type cols() const
      {
        return _layout.cols();
      }

      /*
      Vector & operator= (const Vector & other)
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

      Vector & operator= (Vector && other)
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
      }
      */

      std::pair<size_type,bool> findEntry(size_type i, size_type j) const
      {
        size_type block = i >> block_shift;
        size_type local_index = i & block_mask;
        size_type block_length = _layout.blockLength(block+1);
        const size_type* col_index = _layout.colIndex();
        size_type row_start = _layout.blockOffset(block) + local_index;
        size_type l = 0, r = _layout.rowLength(i)-1;
        while (l<r)
          {
            size_type q = (l+r)/2;
            if (j <= col_index[row_start + q * kernel_block_size])
              r = q;
            else
              l = q+1;
          }
        return {row_start + l * kernel_block_size,col_index[row_start + l * kernel_block_size] == j};
      }

      size_type getEntry(size_type i, size_type j) const
      {
        auto e = findEntry(i,j);
        if (!e.second)
          DUNE_THROW(ISTLError,"entry (" << i << "," << j << ") not in pattern");
        return e.first;
      }

      Field& operator() (size_type i, size_type j)
      {
        return _data[getEntry(i,j)];
      }

      const Field& operator() (size_type i, size_type j) const
      {
        return _data[getEntry(i,j)];
      }

      Row operator[](size_type i)
      {
        return {*this,i};
      }

      ConstRow operator[](size_type i) const
      {
        return {*this,i};
      }

      const Field& element(size_type i, size_type j) const
      {
        auto e = findEntry(i,j);
        return (e.second ? _data[e.first] : _zero_element);
      }

      template<typename RowIndices, typename ColIndices, typename Values>
      void read(const RowIndices& row_indices, const ColIndices& col_indices, Values& values) const
      {
        const size_type* col_index = _layout.colIndex();
        size_type cs = col_indices.size();
        for (size_type i = 0, rs = row_indices.size(); i < rs; ++i)
          {
            size_type ri = row_indices[i];
            size_type block = ri >> block_shift;
            size_type local_index = ri & block_mask;
            size_type row_start = _layout.blockOffset(block) + local_index;
            size_type row_length = _layout.rowLength(i);
            size_type gj = 0;
            size_type j = 0;
            for (; j < cs && gj < row_length; ++j)
              {
                size_type ci = col_indices[j];
                while (col_index[row_start + gj * kernel_block_size] < ci)
                  ++gj;
                assert(col_index[row_start + gj * kernel_block_size] == ci);
                values(i,j) = _data[row_start + gj * kernel_block_size];
              }
            assert(j == cs);
          }
      }

      template<typename RowIndices, typename ColIndices, typename Values>
      void write(const RowIndices& row_indices, const ColIndices& col_indices, const Values& values)
      {
        const size_type* col_index = _layout.colIndex();
        size_type cs = col_indices.size();
        for (size_type i = 0, rs = row_indices.size(); i < rs; ++i)
          {
            size_type ri = row_indices[i];
            size_type block = ri >> block_shift;
            size_type local_index = ri & block_mask;
            size_type row_start = _layout.blockOffset(block) + local_index;
            size_type row_length = _layout.rowLength(i);
            size_type gj = 0;
            size_type j = 0;
            for (; j < cs && gj < row_length; ++j)
              {
                size_type ci = col_indices[j];
                while (col_index[row_start + gj * kernel_block_size] < ci)
                  ++gj;
                assert(col_index[row_start + gj * kernel_block_size] == ci);
                _data[row_start + gj * kernel_block_size] = values(i,j);
              }
            assert(j == cs);
          }
      }

      template<typename RowIndices, typename ColIndices, typename Values>
      void accumulate(const RowIndices& row_indices, const ColIndices& col_indices, const Values& values)
      {
        const size_type* col_index = _layout.colIndex();
        size_type cs = col_indices.size();
        for (size_type i = 0, rs = row_indices.size(); i < rs; ++i)
          {
            size_type ri = row_indices[i];
            size_type block = ri >> block_shift;
            size_type local_index = ri & block_mask;
            size_type row_start = _layout.blockOffset(block) + local_index;
            size_type row_length = _layout.rowLength(i);
            size_type gj = 0;
            size_type j = 0;
            for (; j < cs && gj < row_length; ++j)
              {
                size_type ci = col_indices[j];
                while (col_index[row_start + gj * kernel_block_size] < ci)
                  ++gj;
                assert(col_index[row_start + gj * kernel_block_size] == ci);
                _data[row_start + gj * kernel_block_size] += values(i,j);
              }
            assert(j == cs);
          }
      }

      template<typename Values>
      void read(size_type row_offset, size_type row_size, size_type col_offset, size_type col_size, Values& values) const
      {
        const size_type* col_index = _layout.colIndex();
        for (size_type i = row_offset; i < row_offset + row_size; ++i)
          {
            size_type block = i >> block_shift;
            size_type local_index = i & block_mask;
            size_type row_start = _layout.blockOffset(block) + local_index;
            size_type row_length = _layout.rowLength(i);
            size_type gj = 0;
            size_type j = col_offset;
            for (; j < col_offset + col_size && gj < row_length; ++j)
              {
                while (col_index[row_start + gj * kernel_block_size] < j)
                  ++gj;
                assert(col_index[row_start + gj * kernel_block_size] == j);
                _data[row_start + gj * kernel_block_size] += values(i,j);
              }
            assert(j == col_offset + col_size);
          }
      }

      template<typename Values>
      void write(size_type row, size_type row_size, size_type col, size_type col_size, const Values& values)
      {
        const size_type* col_index = _layout.colIndex();
        size_type start_block = row >> block_shift;
        size_type start_local_index = row & block_mask;
        size_type end_block = (row + row_size) >> block_shift;
        size_type row_start = _layout.blockOffset(start_block) + start_local_index;

        // look for start column
        size_type start_col = 0;
        while (col_index[row_start + start_col * kernel_block_size] < col)
          ++start_col;
        assert(col_index[row_start + start_col * kernel_block_size] == col);

        // (partial) copy of lower part of first block
        for (size_type j = 0; j < col_size; ++j)
          for (size_type i = 0; i < kernel_block_size - start_local_index; ++i)
            _data[row_start + i + (start_col + j) * kernel_block_size] = values(i,j);

        // copy remaining blocks
        size_type row_offset = kernel_block_size - start_local_index;
        for (size_type b = start_block + 1; b < end_block; ++b, row_offset += kernel_block_size)
          {
            row_start = _layout.blockOffset(b);
            // special-case last block
            size_type end_row = (b < end_block - 1 ? kernel_block_size : row_size - row_offset);
            for (size_type j = 0; j < col_size; ++j)
              for (size_type i = row_offset; i < end_row; ++i)
                _data[row_start + i + (start_col + j) * kernel_block_size] = values(i,j);
          }
      }

      template<typename Values>
      void accumulate(size_type offset, size_type count, const Values& values)
      {
        for (size_type i = 0, o = offset; i != count; ++i, ++o)
          _data[o] += values[i];
      }

      void clearRow(size_type i)
      {
        Row r = (*this)[i];
        std::fill(r.begin(),r.end(),value_type(0));
      }

      /*
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
      */

      ELLMatrix& operator=(value_type b)
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
        ELLMatrix
        >::type&
      operator+=(const ELLMatrix<OF,OA,Domain>& other)
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
        ELLMatrix
        >::type&
      operator-=(const ELLMatrix<OF,OA,Domain>& other)
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            value_type* __restrict__  a = _data;
            typename Vector<OF,OA,Domain>::value_type* __restrict__  b = other._data;
            for (size_type i = r.begin(), end = r.end(); i != end; ++i)
              a[i] -= b[i];
          });
        return *this;
      }

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        ELLMatrix
        >::type&
      operator*=(const ELLMatrix<OF,OA,Domain>& other)
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

      ELLMatrix& operator*=(value_type b)
      {
        tbb::parallel_for(
          nonzeros_iteration_range(),
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

      ELLMatrix& operator/=(value_type b)
      {
        return this->operator*=(value_type(1.0)/b);
      }

      ELLMatrix& operator+=(value_type b)
      {
        tbb::parallel_for(
          nonzeros_iteration_range(),
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

      ELLMatrix& operator-=(value_type b)
      {
        return this->operator+=(-b);
      }

      template<typename XF, typename XA, typename YF, typename YA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          XA,
          YA
          >::value
        >::type
      umv(const Vector<XF,XA,Domain>& x, Vector<YF,YA,Domain>& y) const
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::ell::blocked::umv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin(),
                x.data(),
                _data+_layout.blockOffset(r.begin_block()),
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count());
          });
      }

      template<typename XF, typename XA, typename YF, typename YA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          XA,
          YA
          >::value
        >::type
      mv(const Vector<XF,XA,Domain>& x, Vector<YF,YA,Domain>& y) const
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::ell::blocked::mv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin(),
                x.data(),
                _data+_layout.blockOffset(r.begin_block()),
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count());
          });
      }

      template<typename XF, typename XA, typename YF, typename YA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          XA,
          YA
          >::value
        >::type
      mmv(const Vector<XF,XA,Domain>& x, Vector<YF,YA,Domain>& y) const
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::ell::blocked::mmv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin(),
                x.data(),
                _data+_layout.blockOffset(r.begin_block()),
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count());
          });
      }

      template<typename AF, typename XF, typename XA, typename YF, typename YA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          XA,
          YA
          >::value
        >::type
      usmv(const AF& alpha, const Vector<XF,XA,Domain>& x, Vector<YF,YA,Domain>& y) const
      {
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::ell::blocked::usmv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin(),
                x.data(),
                _data+_layout.blockOffset(r.begin_block()),
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count(),
                alpha);
          });
      }


      /*
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
            return Dune::Kernel::vec::blocked::dot<
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
            return Dune::Kernel::vec::blocked::two_norm2<
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
            return Dune::Kernel::vec::blocked::one_norm<
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
            return Dune::Kernel::vec::blocked::infinity_norm<
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                _data+r.begin(),
                r.block_count());
          },
          [](const value_type& a, const value_type& b) -> DF
          {
            return std::max(a,b);
          });
      }
      */

      void setSize(size_type size)
      {
        if (_data)
          DUNE_THROW(NotImplemented,"not allowed");
        allocate(size);
      }


      ~ELLMatrix()
      {
        deallocate();
      }


      range_type iteration_range() const
      {
        return range_type(0,_layout.rows(),kernel_block_size,(_chunk_size > minimum_chunk_size ? _chunk_size : minimum_chunk_size),minimum_chunk_size/kernel_block_size);
      }

      range_type nonzeros_iteration_range() const
      {
        return range_type(0,_layout.nonzeros(),kernel_block_size,(_chunk_size > minimum_chunk_size ? _chunk_size : minimum_chunk_size),minimum_chunk_size/kernel_block_size);
      }

      Allocator& allocator() const
      {
        return _allocator;
      }

      value_type* data() const
      {
        return _data;
      }

      template<typename Archive>
      void archive(Archive& ar)
      {
        ar & _layout;
        if (Archive::Traits::is_writing)
          {
            if (!_data)
              DUNE_THROW(Exception, "need valid data for writing");
          }
        else
          {
            allocate();
          }
        ar.bulk(_data,_layout.allocatedRows());
      }

    private:

      void deallocate()
      {
        if (_data)
          {
            if (!std::is_trivial<value_type>::value)
              {
                tbb::parallel_for(
                  nonzeros_iteration_range(),
                  [&](const range_type& r)
                  {
                    DF* __restrict__  a = _data;
                    for (size_type i = r.begin(), end = r.end(); i != end; ++i)
                      _allocator.destroy(a+i);
                  });
              }
            _allocator.deallocate(_data,_layout.nonzeros());
            _data = nullptr;
          }
      }

      void allocate()
      {
        if (_data)
          return;
        _data = _allocator.allocate(_layout.nonzeros());
        if (!_data)
          DUNE_THROW(Exception,"could not allocate memory");
        // fill padded rows with zeros
        for (size_type i = _layout.rows(), end = _layout.allocatedRows(); i < end; ++i)
          {
            Row row = (*this)[i];
            std::fill(row.begin(),row.end(),value_type(0));
          }
      }

      Layout _layout;
      value_type* _data;
      size_type _chunk_size;
      size_type _nonzeros_chunk_size;
      value_type _zero_element;
      Allocator _allocator;

    };

  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_ELLMATRIX_HOST_HH
