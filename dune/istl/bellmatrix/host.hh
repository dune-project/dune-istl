// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BELLMATRIX_HOST_HH
#define DUNE_ISTL_BELLMATRIX_HOST_HH

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
#include <dune/common/kernel/bell.hh>
#include <dune/common/kernel/vec.hh>

#include <dune/common/threads/range.hh>

#include <dune/istl/forwarddeclarations.hh>
#include <dune/istl/ellmatrix/layout.hh>
#include <dune/istl/ellmatrix/iterator.hh>

namespace Dune {
  namespace ISTL {

    template<typename F_, typename A_>
    class BELLMatrix<F_,A_,Memory::Domain::Host>
    {

      /*
      template<typename, typename>
      friend struct bellmatrix::RowIterator;

      template<typename, typename>
      friend struct bellmatrix::Row;
      */
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

      //typedef ellmatrix::RowIterator<ELLMatrix,value_type> RowIterator;
      //typedef ellmatrix::RowIterator<ELLMatrix,const value_type> ConstRowIterator;

      //typedef ellmatrix::Row<ELLMatrix,value_type> Row;
      //typedef ellmatrix::Row<ELLMatrix,const value_type> ConstRow;

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

      static const size_type kernel_block_shift = Memory::block_size_log2<kernel_block_size,5>::value;
      static const size_type kernel_block_mask = (size_type(1) << kernel_block_shift) - 1;

      typedef Threads::fixed_block_size_range<size_type> range_type;

    public:

      explicit BELLMatrix(size_type block_rows = 1, size_type block_cols = 1)
        : _data(nullptr)
        , _block_rows(block_rows)
        , _block_cols(block_cols)
        , _chunk_size(minimum_chunk_size)
        , _zero_element()
      {}

      explicit BELLMatrix(Layout layout, size_type block_rows = 1, size_type block_cols = 1)
        : _layout(layout)
        , _data(nullptr)
        , _block_rows(block_rows)
        , _block_cols(block_cols)
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

      void setBlockSize(size_type block_rows, size_type block_cols)
      {
        _block_rows = block_rows;
        _block_cols = block_cols;
      }

      size_type blockRows() const
      {
        return _block_rows;
      }

      size_type blockCols() const
      {
        return _block_cols;
      }

      void setLayout(Layout layout, size_type block_rows, size_type block_cols)
      {
        _layout = layout;
        _block_rows = block_rows;
        _block_cols = block_cols;
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
        size_type block = i >> kernel_block_shift;
        size_type local_index = i & kernel_block_mask;
        const size_type* col_index = _layout.colIndex();
        size_type block_offset = _layout.blockOffset(block);
        size_type row_start =  block_offset + local_index;
        size_type l = 0, r = _layout.rowLength(i)-1;
        while (l<r)
          {
            size_type q = (l+r)/2;
            if (j <= col_index[row_start + q * kernel_block_size])
              r = q;
            else
              l = q+1;
          }
        return {local_index + (block_offset + l * kernel_block_size) * _block_rows * _block_cols,col_index[row_start + l * kernel_block_size] == j};
      }

      size_type getEntry(size_type i, size_type j) const
      {
        auto e = findEntry(i,j);
        if (!e.second)
          DUNE_THROW(ISTLError,"entry (" << i << "," << j << ") not in pattern");
        return e.first;
      }

      Field& operator() (size_type i, size_type j, size_type ii, size_type jj)
      {
        assert(getEntry(i,j) + jj * kernel_block_size + ii * kernel_block_size * _block_cols < _layout.nonzeros() * _block_cols * _block_rows);
        return _data[getEntry(i,j) + jj * kernel_block_size + ii * kernel_block_size * _block_cols];
      }

      const Field& operator() (size_type i, size_type j, size_type ii, size_type jj) const
      {
        return _data[getEntry(i,j) + jj * kernel_block_size + ii * kernel_block_size * _block_cols];
      }

      /*
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
      */

      /*
      template<typename RowIndices, typename ColIndices, typename Values>
      void read(const RowIndices& row_indices, const ColIndices& col_indices, Values& values) const
      {
        const size_type* col_index = _layout.colIndex();
        size_type cs = col_indices.size();
        for (size_type i = 0, rs = row_indices.size(); i < rs; ++i)
          {
            size_type ri = row_indices[i];
            size_type block = ri >> kernel_block_shift;
            size_type local_index = ri & kernel_block_mask;
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
            size_type block = ri >> kernel_block_shift;
            size_type local_index = ri & kernel_block_mask;
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
            size_type block = ri >> kernel_block_shift;
            size_type local_index = ri & kernel_block_mask;
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
            size_type block = i >> kernel_block_shift;
            size_type local_index = i & kernel_block_mask;
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
        size_type start_block = row >> kernel_block_shift;
        size_type start_local_index = row & kernel_block_mask;
        size_type end_block = (row + row_size) >> kernel_block_shift;
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
      */

      void clearRow(size_type block, size_type entry)
      {
        //Row r = (*this)[i];
        //std::fill(r.begin(),r.end(),value_type(0));
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

      BELLMatrix& operator=(value_type b)
      {
        size_type block_size = _block_rows * _block_cols;
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
                _data+r.begin() * block_size,
                b,
                r.block_count() * block_size);
          });
        return *this;
      }


      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        BELLMatrix
        >::type&
      operator+=(const BELLMatrix<OF,OA,Domain>& other)
      {
        size_type block_size = _block_rows * _block_cols;
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
                _data+r.begin() * block_size,
                other._data+r.begin() * block_size,
                r.block_count() * block_size);
          });
        return *this;
      }

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        BELLMatrix
        >::type&
      operator-=(const BELLMatrix<OF,OA,Domain>& other)
      {
        size_type block_size = _block_rows * _block_cols;
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
                _data+r.begin() * block_size,
                other._data+r.begin() * block_size,
                r.block_count() * block_size);
          });
        return *this;
      }

      template<typename OF, typename OA>
      typename enable_if<
        Memory::allocators_are_interoperable<
          allocator_type,
          OA
          >::value,
        BELLMatrix
        >::type&
      operator*=(const BELLMatrix<OF,OA,Domain>& other)
      {
        size_type block_size = _block_rows * _block_cols;
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
                _data+r.begin() * block_size,
                other._data+r.begin() * block_size,
                r.block_count() * block_size);
          });
        return *this;
      }

      BELLMatrix& operator*=(value_type b)
      {
        size_type block_size = _block_rows * _block_cols;
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
                _data+r.begin() * block_size,
                b,
                r.block_count() * block_size);
          });
        return *this;
      }

      BELLMatrix& operator/=(value_type b)
      {
        return this->operator*=(value_type(1.0)/b);
      }

      BELLMatrix& operator+=(value_type b)
      {
        size_type block_size = _block_rows * _block_cols;
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
                _data+r.begin() * block_size,
                b,
                r.block_count() * block_size);
          });
        return *this;
      }

      BELLMatrix& operator-=(value_type b)
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
      umv(const BlockVector<XF,XA,Domain>& x, BlockVector<YF,YA,Domain>& y) const
      {
        size_type block_size = _block_rows * _block_cols;
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::bell::blocked::umv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin() * _block_rows,
                x.data(),
                _data+_layout.blockOffset(r.begin_block()) * block_size,
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count(),
                _block_rows,
                _block_cols);
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
      mmv(const BlockVector<XF,XA,Domain>& x, BlockVector<YF,YA,Domain>& y) const
      {
        size_type block_size = _block_rows * _block_cols;
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::bell::blocked::mmv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin() * _block_rows,
                  x.data(),
                _data+_layout.blockOffset(r.begin_block()) * block_size,
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count(),
                _block_rows,
                _block_cols);
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
      usmv(const AF& alpha, const BlockVector<XF,XA,Domain>& x, BlockVector<YF,YA,Domain>& y) const
      {
        size_type block_size = _block_rows * _block_cols;
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::bell::blocked::usmv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin() * _block_rows,
                x.data(),
                _data+_layout.blockOffset(r.begin_block()) * block_size,
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count(),
                _block_rows,
                _block_cols,
                alpha);
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
      mv(const BlockVector<XF,XA,Domain>& x, BlockVector<YF,YA,Domain>& y) const
      {
        size_type block_size = _block_rows * _block_cols;
        tbb::parallel_for(
          iteration_range(),
          [&](const range_type& r)
          {
            Dune::Kernel::bell::blocked::mv<
              YF,
              XF,
              value_type,
              size_type,
              alignment,
              kernel_block_size>(
                y.data()+r.begin() * _block_rows,
                x.data(),
                _data+_layout.blockOffset(r.begin_block()) * block_size,
                _layout.colIndex()+_layout.blockOffset(r.begin_block()),
                _layout.blockOffset()+r.begin_block(),
                r.block_count(),
                _block_rows,
                _block_cols);
          });
      }

      void dumpCOOMatrix(std::ostream& os) const
      {
        for (size_type block = 0; block < _layout.blocks(); ++block)
          {
            size_type block_offset = _layout.blockOffset(block);
            for (size_type ii = 0; ii < kernel_block_size; ++ii)
              {
                size_type block_row = block * kernel_block_size + ii;
                size_type row_length = _layout.rowLength(block_row);
                for (size_type jj = 0; jj < row_length; ++ jj)
                  {
                    size_type block_col = _layout.colIndex(block_offset + ii + jj * kernel_block_size);
                    for (int i = 0; i < _block_rows; ++i)
                      for (int j = 0; j < _block_cols; ++j)
                        os
                          << (block_row * _block_rows + i) << " "
                          << (block_col * _block_cols + j) << " "
                          << _data[block_offset * _block_rows * _block_cols + ii + jj * kernel_block_size * _block_rows * _block_cols + i * _block_cols * kernel_block_size + j * kernel_block_size]
                          << std::endl;
                  }
              }
          }
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

      ~BELLMatrix()
      {
        deallocate();
      }

      Allocator& allocator() const
      {
        return const_cast<Allocator&>(_allocator);
      }

      range_type iteration_range(size_type start, size_type end) const
      {
        return range_type(start,end,kernel_block_size,(_chunk_size > minimum_chunk_size ? _chunk_size : minimum_chunk_size),minimum_chunk_size/kernel_block_size);
      }

      range_type iteration_range(size_type start) const
      {
        return iteration_range(start,_layout.rows());
      }

      range_type iteration_range() const
      {
        return iteration_range(0,_layout.rows());
      }

      range_type nonzeros_iteration_range() const
      {
        return range_type(0,_layout.nonzeros(),kernel_block_size,(_chunk_size > minimum_chunk_size ? _chunk_size : minimum_chunk_size),minimum_chunk_size/kernel_block_size);
      }

      value_type* data()
      {
        return _data;
      }

      const value_type* data() const
      {
        return _data;
      }

      template<typename Archive>
      void archive(Archive& ar)
      {
        ar & _layout;
        ar & _block_rows;
        ar & _block_cols;
        ar & _chunk_size;
        ar & _nonzeros_chunk_size;
        ar & _zero_element;
        if (Archive::Traits::is_reading)
          allocate();
        ar.bulk(_data,_layout.nonzeros() * _block_rows * _block_cols);
      }

      bool operator==(const BELLMatrix& other) const
      {
        return
          _block_rows == other._block_rows &&
          _block_cols == other._block_cols &&
          _layout == other._layout &&
          std::equal(_data,_data + _layout.nonzeros() * _block_rows * _block_cols, other._data);
      }

      bool operator!=(const BELLMatrix& other) const
      {
        return !operator==(other);
      }

    private:

      void deallocate()
      {
        if (_data)
          {
            /*
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
            */
            const size_type block_size = _block_rows * _block_cols;
            _allocator.deallocate(_data,_layout.nonzeros() * block_size);
            _data = nullptr;
          }
      }

      void allocate()
      {
        if (_data)
          return;
        const size_type block_size = _block_rows * _block_cols;
        _data = _allocator.allocate(_layout.nonzeros() * block_size);
        if (!_data)
          DUNE_THROW(Exception,"could not allocate memory");
        // fill padded rows with zeros
        /*
        for (size_type i = _layout.rows(), end = _layout.allocatedRows(); i < end; ++i)
          {
            Row row = (*this)[i];
            std::fill(row.begin(),row.end(),value_type(0));
          }
        */
      }

      Layout _layout;
      value_type* _data;
      size_type _block_rows;
      size_type _block_cols;
      size_type _chunk_size;
      size_type _nonzeros_chunk_size;
      value_type _zero_element;
      Allocator _allocator;

    };

  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_BELLMATRIX_HOST_HH
