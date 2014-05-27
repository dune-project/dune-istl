// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ELLMATRIX_LAYOUT_HH
#define DUNE_ISTL_ELLMATRIX_LAYOUT_HH

#include <cmath>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <functional>

#include <dune/common/static_assert.hh>
#include <dune/common/memory/traits.hh>
#include <dune/common/kernel/ell.hh>

#include <dune/common/threads/range.hh>

#include <dune/istl/istlexception.hh>

namespace Dune {
  namespace ISTL {
    namespace ellmatrix {

      class IllegalStateError
        : public ISTLError
      {};

      template<typename A_>
      struct Data
      {

        typedef A_ Allocator;
        typedef A_ allocator_type;

        typedef typename Allocator::size_type size_type;

        dune_static_assert((is_same<typename Allocator::value_type,size_type>::value),
                           "Layout data allocator must be for allocator_type::size_type");

        // don't change this order, it makes sure that everything before _allocated_rows fits
        // into a single cache line even if sizeof(size_type) == 8.
        size_type _rows;
        size_type _cols;

        size_type* _col_index;
        size_type* _block_offset;
        size_type* _row_length;

        size_type _nonzeros;
        size_type _blocks;

        size_type _chunk_size;
        size_type _non_zeros_chunk_size;
        size_type _allocated_rows;
        allocator_type _allocator;

        template<typename Archive>
        void archive(Archive& ar)
        {
          if (Archive::Traits::is_writing && !(_col_index && _block_offset && _row_length))
            DUNE_THROW(Exception,"Cannot write incomplete archive");
          ar & _rows;
          ar & _cols;
          ar & _nonzeros;
          ar & _blocks;
          ar & _chunk_size;
          ar & _non_zeros_chunk_size;
          ar & _allocated_rows;
          if (Archive::Traits::is_reading)
            {
              allocateRows();
              allocateCols();
            }
          ar.bulk(_col_index,_nonzeros);
          ar.bulk(_block_offset,_blocks+1);
          ar.bulk(_row_length,_allocated_rows);
        }

        Data()
          : _rows(0)
          , _cols(0)
          , _col_index(nullptr)
          , _block_offset(nullptr)
          , _row_length(nullptr)
          , _nonzeros(0)
          , _blocks(0)
          , _chunk_size(Allocator::minimum_chunk_size)
          , _non_zeros_chunk_size(Allocator::minimum_chunk_size)
          , _allocated_rows(0)
        {}

        bool operator==(const Data& other) const
        {
          return
            _rows == other._rows &&
            _cols == other._cols &&
            _nonzeros == other._nonzeros &&
            _blocks == other._blocks &&
            _allocated_rows == other._allocated_rows &&
            std::equal(_col_index,_col_index + _nonzeros,other._col_index) &&
            std::equal(_block_offset,_block_offset + _blocks + 1,other._block_offset) &&
            std::equal(_row_length,_row_length + _allocated_rows,other._row_length);
        }

        bool operator!=(const Data& other) const
        {
          return !operator==(other);
        }

        // Data should never be copied
        Data(const Data&) = delete;
        Data& operator=(const Data&) = delete;

        void allocateRows()
        {
          assert(!_row_length);
          assert(!_block_offset);
          _row_length = _allocator.allocate(_allocated_rows);
          // make sure rows added for padding are empty
          std::fill(_row_length + _rows,_row_length + _allocated_rows,size_type(0));
          _block_offset = _allocator.allocate(_blocks + 1);
        }

        void allocateCols()
        {
          assert(!_col_index);
          _col_index = _allocator.allocate(_nonzeros);
        }

        ~Data()
        {
          if (_col_index)
            {
              _allocator.deallocate(_col_index,_nonzeros);
              _col_index = nullptr;
            }
          if (_block_offset)
            {
              _allocator.deallocate(_block_offset,_blocks + 1);
              _block_offset =  nullptr;
            }
          if (_row_length)
            {
              _allocator.deallocate(_row_length,_allocated_rows);
              _row_length = nullptr;
            }
        }

      };

    template<typename A_>
    class LayoutBuilder;


    template<typename A_>
    class Layout
    {

    private:

      typedef ellmatrix::Data<A_> Data;

      friend class LayoutBuilder<A_>;

    public:

      typedef A_ Allocator;
      typedef Allocator allocator_type;
      typedef typename Allocator::size_type size_type;
      static const size_type block_size = Allocator::block_size;
      static const size_type block_shift = Memory::block_size_log2<block_size>::value;
      static const size_type block_mask = Allocator::block_size - 1;

      template<typename Archive>
      void archive(Archive& ar)
      {
        if (Archive::Traits::is_reading)
          _data = make_shared<Data>();
        ar & *_data;
      }

      size_type minChunkSize() const
      {
        return Allocator::minimum_chunk_size;
      }

      size_type rows() const
      {
        return _data->_rows;
      }

      size_type cols() const
      {
        return _data->_cols;
      }

      size_type blocks() const
      {
        return _data->_blocks;
      }

      size_type allocatedRows() const
      {
        return _data->_allocated_rows;
      }

      size_type nonzeros() const
      {
        return _data->_nonzeros;
      }

      const size_type* colIndex() const
      {
        return _data->_col_index;
      }

      size_type colIndex(size_type i) const
      {
        return _data->_col_index[i];
      }

      const size_type* blockOffset() const
      {
        return _data->_block_offset;
      }

      size_type blockOffset(size_type i) const
      {
        return _data->_block_offset[i];
      }

      size_type blockLength(size_type i) const
      {
        return (_data->_block_offset[i+1] - _data->_block_offset[i]) >> block_shift;
      }

      const size_type* rowLength() const
      {
        return _data->_row_length;
      }

      size_type rowLength(size_type i) const
      {
        return _data->_row_length[i];
      }

      bool operator==(const Layout& other) const
      {
        return _data == other._data || *_data == *(other._data);
      }

      bool operator!=(const Layout& other) const
      {
        return _data != other._data || *_data != *(other._data);
      }

      size_type chunkSize() const
      {
        return _data->_chunk_size;
      }

      size_type nonZerosChunkSize() const
      {
        return _data->_non_zeros_chunk_size;
      }

      Layout()
      {}

      Allocator& allocator() const
      {
        return _data->_allocator;
      }

    private:

      Layout(shared_ptr<Data> data)
        : _data(data)
      {}

      shared_ptr<Data> _data;

    };


    template<typename A_>
    class LayoutBuilder
    {

    public:

      typedef A_ Allocator;
      typedef A_ allocator_type;

      typedef Dune::ISTL::ellmatrix::Layout<A_> Layout;

      typedef typename Allocator::size_type size_type;

      static const size_type block_size = Allocator::block_size;
      static const size_type block_shift = Memory::block_size_log2<block_size>::value;
      static const size_type block_mask = block_size - 1;

      static const size_type minimum_chunk_size = Allocator::minimum_chunk_size;

    private:

      typedef ellmatrix::Data<A_> Data;

    public:

      size_type minChunkSize() const
      {
        return Allocator::minimum_chunk_size;
      }

      size_type chunkSize() const
      {
        return _data->_chunk_size;
      }

      void setChunkSize(size_type chunk_size)
      {
        //TODO: add sanity check!
        _data->_chunk_size = chunk_size;
      }

      size_type nonZerosChunkSize() const
      {
        return _data->_non_zeroschunk_size;
      }

      void setNonZerosChunkSize(size_type non_zeros_chunk_size)
      {
        //TODO: add sanity check!
        _data->_non_zeros_chunk_size = non_zeros_chunk_size;
      }

      size_type rows() const
      {
        return _data->_rows;
      }

      size_type blocks() const
      {
        return _data->_blocks;
      }

      void setRows(size_type rows) const
      {
        if (_rows_allocated)
          DUNE_THROW(IllegalStateError,"Cannot change number of rows, data structures already allocated.");
        _data->_rows = rows;
        // add padding to minimum chunk size
        _data->_allocated_rows = rows + minimum_chunk_size - 1 - ((rows - 1) % minimum_chunk_size);
      }

      size_type cols() const
      {
        return _data->_cols;
      }

      void setCols(size_type cols) const
      {
        if (_cols_allocated)
          DUNE_THROW(IllegalStateError,"Cannot change number of columns, data structures already allocated.");
        _data->_cols = cols;
      }

      size_type allocatedRows() const
      {
        return _data->_allocated_rows;
      }

      size_type nonzeros() const
      {
        return _data->_nonzeros;
      }

      size_type* colIndex()
      {
        assert(_data->_col_index);
        return _data->_col_index;
      }

      size_type colIndex(size_type i) const
      {
        assert(_data->_col_index);
        return _data->_col_index[i];
      }

      size_type* blockOffset()
      {
        assert(_data->_block_offset);
        return _data->_block_offset;
      }

      size_type blockOffset(size_type i) const
      {
        assert(_data->_block_offset);
        return _data->_block_offset[i];
      }

      size_type blockLength(size_type i) const
      {
        assert(_data->_block_offset);
        return (_data->_block_offset[i+1] - _data->_block_offset[i]) >> block_shift;
      }

      size_type* rowLength()
      {
        assert(_data->_row_length);
        return _data->_row_length;
      }

      size_type rowLength(size_type i) const
      {
        assert(_data->_row_length);
        return _data->_row_length[i];
      }

      void setSize(size_type rows, size_type cols)
      {
        if (_rows_allocated)
          DUNE_THROW(IllegalStateError,"Cannot change size, data structures already allocated.");
        _data->_rows = rows;
        _data->_cols = cols;
        _data->_blocks = rows / block_size + ((rows % minimum_chunk_size) > 0) * (minimum_chunk_size / block_size);
        // add padding to minimum chunk size
        _data->_allocated_rows = _data->_blocks * block_size;
      }

      void allocateRows()
      {
        if (_rows_allocated)
          DUNE_THROW(IllegalStateError,"Cannot allocate row data, already allocated");
        _data->allocateRows();
        _rows_allocated = true;
      }

      void allocateCols()
      {
        if (!_rows_allocated)
          DUNE_THROW(IllegalStateError,"Cannot allocate column data, allocate row data first");
        if (_cols_allocated)
          DUNE_THROW(IllegalStateError,"Cannot allocate column data, already allocated");

        _data->_block_offset[0] = 0;
        size_type block_offset = 0;
        for (size_type b = 0, i = 0, blocks = _data->_blocks; b < blocks; ++b, i += block_size)
          {
            size_type block_length = *std::max_element(rowLength()+i,rowLength()+i+block_size);
            size_type block_nonzeros = block_length * block_size;
            _data->_block_offset[b+1] = (block_offset += block_nonzeros);
          }
        _data->_nonzeros = _data->_block_offset[_data->_blocks];
        _data->allocateCols();
        _cols_allocated = true;
      }

      Layout layout()
      {
        if (!_cols_allocated)
          DUNE_THROW(IllegalStateError,"Layout is not completely built.");
        return {_data};
      }

      LayoutBuilder()
        : _data(make_shared<Data>())
        , _rows_allocated(false)
        , _cols_allocated(false)
      {}

      // LayoutBuilder cannot be copied
      LayoutBuilder(const LayoutBuilder&) = delete;
      LayoutBuilder& operator=(const LayoutBuilder&) = delete;

    private:

      shared_ptr<Data> _data;
      bool _rows_allocated;
      bool _cols_allocated;

    };

    } // namespace ellmatrix
  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_ELLMATRIX_LAYOUT_HH
