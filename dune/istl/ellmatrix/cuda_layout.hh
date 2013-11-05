// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ELLMATRIX_CUDA_LAYOUT_HH
#define DUNE_ISTL_ELLMATRIX_CUDA_LAYOUT_HH

#include <cmath>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <vector>
#include <set>

#include <dune/common/static_assert.hh>
#include <dune/common/memory/traits.hh>
//#include <dune/common/kernel/ell.hh>

#include <dune/istl/istlexception.hh>
#include <dune/istl/ellmatrix/layout.hh>
#include <dune/common/memory/cuda_allocator.hh>

namespace Dune {
  namespace ISTL {
    namespace ellmatrix {

      template<typename A_>
      struct CudaData
      {

        typedef A_ Allocator;
        typedef A_ allocator_type;

        typedef typename Allocator::size_type size_type;

        dune_static_assert((is_same<typename Allocator::value_type,size_type>::value),
                           "Layout data allocator must be for allocator_type::size_type");

        size_type _rows;
        size_type _cols;
        // total non zero element count
        size_type _nonzeros;
        // total data/column array size allocated
        size_type _allocated_size;

        // C: Row count per chunk
        size_type _rows_per_chunk;
        // total chunk count
        size_type _chunks;
        // sigma: scope of descending sorted rows, must be a multiple of C (rows_per_chunk)
        size_type _sorting_scope;

        // column indices for each non zero value
        size_type * _col;
        // starting offset of each row-chunk
        size_type * _cs;
        // width of each chunk; i.e. length of the longes row in the chunk
        size_type * _cl;

        allocator_type _allocator;

        CudaData()
          : _rows(0)
          , _cols(0)
          , _nonzeros(0)
          , _allocated_size(0)
          , _rows_per_chunk(1)
          , _sorting_scope(1)
        {}

        // CudaData should never be copied
        CudaData(const CudaData&) = delete;
        CudaData& operator=(const CudaData&) = delete;

        ~CudaData()
        {
        }

      };

    template<typename A_>
    class CudaLayout
    {

    private:

      typedef ellmatrix::CudaData<A_> Data;

    public:

      typedef A_ Allocator;
      typedef Allocator allocator_type;
      typedef typename Allocator::size_type size_type;

      CudaLayout(size_type * row, size_type * col, size_type rows, size_type cols, size_type nonzeros, size_type rows_per_chunk, size_type sorting_scope)
        : _data(new CudaData<A_>())
      {
        _data->_rows = rows;
        _data->_cols = cols;
        _data->_nonzeros = nonzeros;
        _data->_rows_per_chunk = rows_per_chunk;
        _data->_sorting_scope = sorting_scope;
        _data->_chunks = (size_t)ceil(float(rows) / float(rows_per_chunk));

        _data->_cs = _data->_allocator.allocate(_data->_chunks);
        _data->_cl = _data->_allocator.allocate(_data->_chunks);

        size_type * tcs = new size_type[_data->_chunks];
        size_type * tcl = new size_type[_data->_chunks];

        // hold all column indices, sorted for each row (including padded rows)
        std::vector<std::set<size_t> > row_idx;
        for (size_type i(0) ; i < rows_per_chunk * _data->_chunks ; ++i)
        {
          std::set<size_t> t;
          row_idx.push_back(t);
        }
        for (size_type i(0) ; i < nonzeros ; ++i)
        {
          row_idx.at(row[i]).insert(col[i]);
        }

        // calculate max chunk size and chunk starting offsets
        tcs[0] = 0;
        for (size_type chunk(0) ; chunk < _data->_chunks ; ++chunk)
        {
          size_type row_start(chunk * rows_per_chunk);
          size_type row_end(row_start + rows_per_chunk);

          size_type max_cl(0);
          for (size_type i(row_start) ; i < row_end ; ++i)
            max_cl=std::max(max_cl, row_idx.at(i).size());
          tcl[chunk] = max_cl;
          if (chunk != _data->_chunks - 1)
            tcs[chunk+1] = tcs[chunk] + (max_cl * rows_per_chunk);
        }


        // calculate column and nonzero-value array size
        _data->_allocated_size = 0;
        for (size_type i(0) ; i < _data->_chunks ; ++i)
          _data->_allocated_size += tcl[i] * rows_per_chunk;

        size_type * tcol = new size_type[_data->_allocated_size];
        memset(tcol, 0, _data->_allocated_size * sizeof(size_t));

        //fill column index array
        for (size_type chunk(0) ; chunk < _data->_chunks ; ++chunk)
        {
          // starting global row in chunk
          size_type row_start(chunk * rows_per_chunk);

          // the column to be filled in chunk
          for (size_type col_insert(0) ; col_insert < tcl[chunk] ; ++col_insert)
          {
            // the current row (relative to row_start) to be filled
            for (size_type row_insert(0) ; row_insert < rows_per_chunk ; ++row_insert)
            {
              // search for col_insert'th column index in current row
              auto it(row_idx.at(row_start + row_insert).begin());
              for (size_type i(0) ; i < col_insert && it != row_idx.at(row_start + row_insert).end() ; ++i, ++it) ;
              // if not reached end of row, insert column index
              if (it != row_idx.at(row_start + row_insert).end())
              {
                // index in global column array
                size_type idx (tcs[chunk] + col_insert * rows_per_chunk + row_insert);
                tcol[idx] = *it;
              }
            }
          }
        }

        _data->_col = _data->_allocator.allocate(_data->_allocated_size);
        //upload data
        Cuda::upload(_data->_cs, tcs, _data->_chunks);
        Cuda::upload(_data->_cl, tcl, _data->_chunks);
        Cuda::upload(_data->_col, tcol, _data->_allocated_size);

        delete[] tcs;
        delete[] tcl;
        delete[] tcol;

      }

      template <size_t blocksize_>
      CudaLayout(const Layout<Dune::Memory::blocked_cache_aligned_allocator<typename Allocator::value_type,std::size_t, blocksize_> > & host_layout)
        : _data(new CudaData<A_>())
      {
        _data->_rows = host_layout.rows();
        _data->_cols = host_layout.cols();
        _data->_nonzeros = host_layout.nonzeros();
        _data->_rows_per_chunk = host_layout.chunkSize();
        /// \todo remove hardcoded numbers
        _data->_sorting_scope = 1;
        _data->_chunks = host_layout.blocks();
        _data->_allocated_size = host_layout.nonzeros();

        _data->_cs = _data->_allocator.allocate(_data->_chunks);
        _data->_cl = _data->_allocator.allocate(_data->_chunks);
        _data->_col = _data->_allocator.allocate(_data->_allocated_size);

        size_type * tcs = new size_type[_data->_chunks];
        for (size_type i(0) ; i < _data->_chunks ; ++i)
          tcs[i] = host_layout.blockOffset()[i];

        size_type * tcl = new size_type[_data->_chunks];
        for (size_type i(0), j(0) ; i < _data->_chunks ; ++i, j+=_data->_rows_per_chunk)
          tcl[i] = host_layout.rowLength()[j];

        size_type * tcol = new size_type[_data->_allocated_size];
        for (size_type i(0) ; i < _data->_allocated_size ; ++i)
          tcs[i] = host_layout.colIndex()[i];

        //upload data
        Cuda::upload(_data->_cs, tcs, _data->_chunks);
        Cuda::upload(_data->_cl, tcl, _data->_chunks);
        Cuda::upload(_data->_col, tcol, _data->_allocated_size);

        delete[] tcs;
        delete[] tcl;
        delete[] tcol;
      }

      void print() const
      {
        std::cout<<"Rows: "<<_data->_rows<<std::endl;
        std::cout<<"Cols: "<<_data->_cols<<std::endl;
        std::cout<<"NonZeros: "<<_data->_nonzeros<<std::endl;
        std::cout<<"AllocatedSize: "<<_data->_allocated_size<<std::endl;
        std::cout<<"RowsPerChunk: "<<_data->_rows_per_chunk<<std::endl;
        std::cout<<"SortingScope: "<<_data->_sorting_scope<<std::endl;
        std::cout<<"Chunks: "<<_data->_chunks<<std::endl;

        size_type * temp = new size_type[_data->_allocated_size];
        Cuda::download(temp, _data->_col, _data->_allocated_size);
        std::cout<<"Col: ";
        for (size_type i(0) ; i < _data->_allocated_size ; ++i)
          std::cout<<temp[i]<<" ";
        std::cout<<std::endl;
        delete[] temp;

        temp = new size_type[_data->_chunks];
        Cuda::download(temp, _data->_cs, _data->_chunks);
        std::cout<<"cs: ";
        for (size_type i(0) ; i < _data->_chunks ; ++i)
          std::cout<<temp[i]<<" ";
        std::cout<<std::endl;
        delete[] temp;

        temp = new size_type[_data->_chunks];
        Cuda::download(temp, _data->_cl, _data->_chunks);
        std::cout<<"cl: ";
        for (size_type i(0) ; i < _data->_chunks ; ++i)
          std::cout<<temp[i]<<" ";
        std::cout<<std::endl;
        delete[] temp;
      }


      size_type rows() const
      {
        return _data->_rows;
      }

      size_type cols() const
      {
        return _data->_cols;
      }

      size_type nonzeros() const
      {
        return _data->_nonzeros;
      }

      size_type allocated_size() const
      {
        return _data->_allocated_size;
      }

      size_type chunks() const
      {
        return _data->_chunks;
      }

      size_type rows_per_chunk() const
      {
        return _data->_rows_per_chunk;
      }

      size_type * cs() const
      {
        return _data->_cs;
      }

      size_type * cl() const
      {
        return _data->_cl;
      }

      size_type * col() const
      {
        return _data->_col;
      }

      bool operator==(const CudaLayout& other) const
      {
        return _data == other._data;
      }

      bool operator!=(const CudaLayout& other) const
      {
        return _data != other._data;
      }

      Allocator& allocator() const
      {
        return _data->_allocator;
      }

    private:

      CudaLayout(std::shared_ptr<Data> data)
        : _data(data)
      {}

      std::shared_ptr<Data> _data;

    };
    } // namespace ellmatrix
  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_ELLMATRIX_CUDA_LAYOUT_HH
