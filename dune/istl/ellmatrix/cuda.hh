// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ELLMATRIX_CUDA_HH
#define DUNE_ISTL_ELLMATRIX_CUDA_HH

#include <cmath>
#include <memory>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <map>


#include <dune/common/static_assert.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/promotiontraits.hh>
#include <dune/common/dotproduct.hh>
#include <dune/common/memory/domain.hh>
#include <dune/common/memory/alignment.hh>
#include <dune/common/memory/traits.hh>
#include <dune/common/memory/cuda_allocator.hh>
#include <dune/common/kernel/vec/cuda_kernels.hh>
#include <dune/common/kernel/ell/cuda_kernels.hh>

#include <dune/istl/forwarddeclarations.hh>
#include <dune/istl/ellmatrix/cuda_layout.hh>

namespace Dune {
  namespace ISTL {

    template<typename F_, typename A_>
    class ELLMatrix<F_,A_,Memory::Domain::CUDA>
    {
    public:
      typedef F_ Field;
      typedef F_ DataType;
      typedef F_ DT_;
      typedef F_ value_type;

      typedef Memory::Domain::CUDA Domain;

      // F_::DataType DataType;
      typedef F_ DF;

      typedef typename A_::template rebind<F_>::other Allocator;
      typedef typename A_::template rebind<typename Allocator::size_type>::other IndexAllocator;
      typedef Allocator allocator_type;
      typedef IndexAllocator index_allocator_type;
      typedef typename A_::size_type size_type;

      typedef ellmatrix::CudaLayout<IndexAllocator> Layout;

    private:
      Layout _layout;
      DT_* _data;
      DT_ _zero_element;
      Allocator _allocator;
      size_t _cuda_blocksize;



    public:

      /*ELLMatrix()
        : _data(nullptr)
        , _zero_element(0)
      {}*/

      ELLMatrix(Layout layout)
        : _layout(layout)
        , _data(nullptr)
        , _zero_element(0)
        , _cuda_blocksize(128)
      {
        allocate();
      }

      ELLMatrix(DT_ * val, size_type * row, size_type * col, size_type nonzeros, size_type rows, size_type cols, size_type rows_per_chunk, size_type sorting_scope)
        : _layout(row, col, rows, cols, nonzeros, rows_per_chunk, sorting_scope)
        , _data(nullptr)
        , _zero_element(0)
        , _cuda_blocksize(128)
      {
        allocate();
        DT_ * tdata = new DT_[_layout.allocated_size()];
        memset(tdata, 0, _layout.allocated_size() * sizeof(DT_));

        size_t * tcs = new size_t[_layout.chunks() + 1];
        Cuda::download(tcs, _layout.cs(), _layout.chunks() + 1);
        size_t * tcl = new size_t[_layout.chunks()];
        Cuda::download(tcl, _layout.cl(), _layout.chunks());

        // hold all non zero values, sorted for each row (with respect to column number) (including padded rows)
        std::vector<std::map<size_t, DT_> > row_idx;
        for (size_t i(0) ; i < rows_per_chunk * _layout.chunks() ; ++i)
        {
          std::map<size_t, DT_> t;
          row_idx.push_back(t);
        }
        for (size_t i(0) ; i < nonzeros ; ++i)
        {
          row_idx.at(row[i]).insert(std::pair<size_t, DT_>(col[i], val[i]));
        }

        //fill column index array
        for (size_t chunk(0) ; chunk < _layout.chunks() ; ++chunk)
        {
          // starting global row in chunk
          size_t row_start(chunk * rows_per_chunk);

          // the column to be filled in current chunk
          for (size_t col_insert(0) ; col_insert < tcl[chunk] ; ++col_insert)
          {
            // the current row (relative to row_start) to be filled
            for (size_t row_insert(0) ; row_insert < rows_per_chunk ; ++row_insert)
            {
              // search for col_insert'th non zero value in current row
              auto it(row_idx.at(row_start + row_insert).begin());
              for (size_t i(0) ; i < col_insert && it != row_idx.at(row_start + row_insert).end() ; ++i, ++it) ;
              // if not reached end of row, insert non zero value
              if (it != row_idx.at(row_start + row_insert).end())
              {
                // index in global nonzero array
                size_t idx (tcs[chunk] + col_insert * rows_per_chunk + row_insert);
                tdata[idx] = it->second;
              }
            }
          }
        }

        Cuda::upload(_data, tdata, _layout.allocated_size());

        delete[] tcs;
        delete[] tcl;
        delete[] tdata;
      }

      void print() const
      {
        _layout.print();
        DT_ * temp = new DT_[_layout.allocated_size()];
        Cuda::download(temp, _data, _layout.allocated_size());
        std::cout<<"Val: ";
        for (size_t i(0) ; i < _layout.allocated_size() ; ++i)
          std::cout<<temp[i]<<" ";
        std::cout<<std::endl;
        delete[] temp;

        for (size_t row(0) ; row < _layout.rows() ; ++row)
        {
          for (size_t col(0) ; col < _layout.cols() ; ++col)
          {
            std::cout<<this->element(row, col)<<" ";
          }
          std::cout<<std::endl;
        }
      }

      void setLayout(Layout layout)
      {
        _layout = layout;
        allocate();
      }

      const Layout& layout() const
      {
        return _layout;
      }

      DT_ * data ()
      {
        return _data;
      }

      DT_ * data () const
      {
        return _data;
      }

      ELLMatrix(const ELLMatrix& other)
        : _layout(other._layout)
        , _data(nullptr)
        , _zero_element(other._zero_element)
        , _cuda_blocksize(other._cuda_blocksize)
      {
        allocate();
        Cuda::copy(_data, other._data, _layout.allocated_size());
      }

      ELLMatrix(ELLMatrix&& other)
        : _allocator(std::move(other._allocator))
        , _layout(other.layout)
        , _data(other._data)
        , _zero_element(other._zero_element)
        , _cuda_blocksize(other._cuda_blocksize)
      {
        other._data = nullptr;
      }

      ELLMatrix & operator= (const ELLMatrix & other)
      {
        if (_layout.allocated_size() == other._layout.allocated_size())
        {
          _layout = other._layout;
          Cuda::copy(_data, other._data, other._layout.allocated_size());
        }
        else
        {
          _layout = other._layout;
          if (_data)
            deallocate();
          _allocator = other._allocator;
          allocate();
          Cuda::copy(_data, other._data, other._layout.allocated_size());
        }
        _zero_element = other._zero_element;

        return *this;
      }

      ELLMatrix & operator= (ELLMatrix && other)
      {
        if (_data)
          deallocate();
        _layout = other._layout;
        _zero_element = other._zero_element;
        _allocator = std::move(other._allocator);
        _data = other._data;
        other._data = nullptr;
      }

      template<typename A2_>
      ELLMatrix(ELLMatrix<F_, A2_, Memory::Domain::Host> & other)
        : _layout(other.layout())
        , _data(nullptr)
        , _zero_element(0)
        , _cuda_blocksize(128)
      {
        allocate();
        Cuda::upload(_data, other.data(), _layout.allocated_size());
      }

      DT_ operator() (size_t row, size_t col) const
      {
        size_t * tcs = new size_t[_layout.chunks() + 1];
        Cuda::download(tcs, _layout.cs(), _layout.chunks() + 1);
        size_t * tcol = new size_t[_layout.allocated_size()];
        Cuda::download(tcol, _layout.col(), _layout.allocated_size());

        // chunk to look at
        size_t chunk(row / _layout.rows_per_chunk());
        // row offset into current chunk
        size_t local_row(row % _layout.rows_per_chunk());
        // starting index of our row, leftmost column
        size_t pcol(tcs[chunk] + local_row);
        // end index (into global array) of our chunk
        size_t chunk_end(tcs[chunk+1]);
        DT_ result(0);
        // walk down the current row until we find the searched col or are behind it
        for ( ; pcol < chunk_end ; pcol += _layout.rows_per_chunk() )
        {
          if (tcol[pcol] < col)
            continue;
          else if (tcol[pcol] == col)
          {
            result = Cuda::get(_data + pcol);
            if (fabs(result - _zero_element) < 1e-10)
              result = _zero_element;
            break;
          }
          else if (tcol[pcol] > col)
          {
            result = _zero_element;
            break;
          }
        }
        delete[] tcs;
        delete[] tcol;
        return result;
      }

      DT_ element(size_t row, size_t col) const
      {
        return (*this)(row, col);
      }

      size_type rows() const
      {
        return _layout.rows();
      }

      size_type cols() const
      {
        return _layout.cols();
      }

      size_type size() const
      {
        return cols() * rows();
      }

      size_type nonzeros() const
      {
        return _layout.nonzeros();
      }

      size_t cuda_blocksize() const
      {
        return _cuda_blocksize;
      }

      void set_cuda_blocksize(size_t cbs)
      {
        _cuda_blocksize = cbs;
      }

      void mv(const Vector<F_, A_, Memory::Domain::CUDA> & x, Vector<F_, A_, Memory::Domain::CUDA> & y) const
      {
        Cuda::mv(x.begin(), y.begin(), _data, _layout.cs(), _layout.col(), _layout.rows(), _layout.rows_per_chunk(), _layout.chunks(), _layout.allocated_size(), _cuda_blocksize);
      }

      void umv(const Vector<F_, A_, Memory::Domain::CUDA> & x, Vector<F_, A_, Memory::Domain::CUDA> & y) const
      {
        Cuda::umv(x.begin(), y.begin(), _data, _layout.cs(), _layout.col(), _layout.rows(), _layout.rows_per_chunk(), _layout.chunks(), _layout.allocated_size(), _cuda_blocksize);
      }

      void mmv(const Vector<F_, A_, Memory::Domain::CUDA> & x, Vector<F_, A_, Memory::Domain::CUDA> & y) const
      {
        Cuda::mmv(x.begin(), y.begin(), _data, _layout.cs(), _layout.col(), _layout.rows(), _layout.rows_per_chunk(), _layout.chunks(), _layout.allocated_size(), _cuda_blocksize);
      }

      void usmv(const DT_ alpha, const Vector<F_, A_, Memory::Domain::CUDA> & x, Vector<F_, A_, Memory::Domain::CUDA> & y) const
      {
        Cuda::usmv(alpha, x.begin(), y.begin(), _data, _layout.cs(), _layout.col(), _layout.rows(), _layout.rows_per_chunk(), _layout.chunks(), _layout.allocated_size(), _cuda_blocksize);
      }

      ~ELLMatrix()
      {
        deallocate();
      }


    private:

      void deallocate()
      {
        if (_data)
          {
            if (!std::is_trivial<DT_>::value)
              {
                DUNE_THROW(NotImplemented, "do not use non trivial types with cuda!");
              }
            _allocator.deallocate(_data,_layout.nonzeros());
            _data = nullptr;
          }
      }

      void allocate()
      {
        if (_data)
          DUNE_THROW(NotImplemented,"do not reallocate memory");
        _data = _allocator.allocate(_layout.allocated_size());
        if (!_data)
          DUNE_THROW(Exception,"could not allocate memory");
      }

    };

  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_ELLMATRIX_CUDA_HH
