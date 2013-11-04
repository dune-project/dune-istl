// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_ELLMATRIX_ITERATOR_HH
#define DUNE_ISTL_ELLMATRIX_ITERATOR_HH

#include <limits>
#include <type_traits>
#include <algorithm>

#include <dune/common/static_assert.hh>
#include <dune/common/typetraits.hh>

namespace Dune {
  namespace ISTL {
    namespace ellmatrix {

      /*
    template<typename T>
    struct Iterator
    {

      void operator++()
      {
        if (_i == _rows)
          DUNE_THROW(Exception,"kaboom");
        if (_col != _col_end)
          {
            _col += block_size;
            _val += block_size;
            return;
          }
        if ((_i & block_mask) != 0)
          {
            ++_i;
            ++_col_begin;
            _col_end = _col_begin + _layout->rowLength()[_i];
            ++_val_begin;
            _col = _col_begin;
            _val = _val_begin;
          }
        else
          {
            ++_i;
            _col_begin = _col_index[_layout->blockOffset(_i >> block_shift)];
            _col_end = _col_begin + _layout->rowLength()[_i];
            _col = _col_begin;
            _val_begin = _data[_layout->blockOffset(_i >> block_shift)];
            _val = _val_begin;
          }
      }

      size_type row() const
      {
        return _i;
      }

      size_type col() const
      {
        return *_col;
      }

      size_type block() const
      {
        return _i >> block_shift;
      }

      bool operator==(const Iterator& other) const
      {
        return (_i == other._i && _col == other._col);
      }

      bool operator!=(const Iterator& other) const
      {
        return !(_i == other._i && _col == other._col);
      }

      Field operator*() const
      {
        return *_val;
      }

      Field* operator->() const
      {
        return _val;
      }

    private:

      size_type i;
      const size_type _rows;
      size_type* _col;
      value_type* _val;
      size_type* _col_begin;
      size_type* _col_end;
      size_type* _col_index;
      value_type* _val_begin;
      value_type* _data;

      const Layout* _layout;

    };
      */


    template<typename M, typename T>
    struct Row;


    template<typename M, typename T>
    class ColIterator
      : public RandomAccessIteratorFacade<ColIterator<M,T>,
                                          T,
                                          typename std::add_lvalue_reference<T>::type,
                                          typename M::Allocator::difference_type
                                          >
    {

    private:

      template<typename, typename>
      friend struct Row;

      friend class RandomAccessIteratorFacade<
        ColIterator,
        T,
        typename std::add_lvalue_reference<T>::type,
        typename M::Allocator::difference_type
        >;

    public:

      typedef typename M::size_type size_type;
      typedef typename remove_const<T>::type value_type;
      typedef typename std::add_lvalue_reference<T>::type reference;
      typedef typename M::Allocator::difference_type difference_type;

      static const size_type kernel_block_size = M::kernel_block_size;
      static const size_type block_shift = M::block_shift;
      static const size_type block_mask = M::block_mask;

    private:
      // keep this public for now, access from the facade is a mess!
    public:

      reference dereference() const
      {
        return *_val;
      }

      reference elementAt(difference_type n) const
      {
        return _val[n];
      }

      bool equals(const ColIterator& other) const
      {
        return _col == other._col;
      }

      void increment()
      {
        _val += kernel_block_size;
        _col += kernel_block_size;
      }

      void decrement()
      {
        _val -= kernel_block_size;
        _col -= kernel_block_size;
      }

      void advance(difference_type n)
      {
        _val += n * kernel_block_size;
        _col += n * kernel_block_size;
      }

      difference_type distanceTo(const ColIterator& other) const
      {
        return (_col - other._col) >> block_shift;
      }

    public:

      difference_type col() const
      {
        return (_col - _col_begin) >> block_shift;
      }

    private:

      ColIterator(const size_type* col_begin, const size_type* col, value_type* val)
        : _col(col)
        , _val(val)
        , _col_begin(col_begin)
      {}

      const size_type* _col;
      value_type* _val;
      const size_type* _col_begin;

    };



    template<typename M, typename T>
    struct Row
    {
      typedef T value_type;
      typedef typename M::size_type size_type;

      typedef ColIterator<M,T> iterator;
      typedef ColIterator<M,typename std::add_const<T>::type> const_iterator;

      static const size_type kernel_block_size = M::kernel_block_size;
      static const size_type block_shift = M::block_shift;
      static const size_type block_mask = M::block_mask;

      size_type row() const
      {
        return _row;
      }

      size_type block() const
      {
        return _row >> block_shift;
      }

      size_type size() const
      {
        return (_col_end - _col_begin) >> block_shift;
      }

      iterator begin()
      {
        return {_col_begin,_col_begin,_val_begin};
      }

      iterator end()
      {
        return {_col_begin,_col_end,_val_begin + (_col_end - _col_begin)};
      }

      const_iterator begin() const
      {
        return {_col_begin,_col_begin,_val_begin};
      }

      const_iterator end() const
      {
        return {_col_begin,_col_end,_val_begin + (_col_end - _col_begin)};
      }

      size_type nonzeros() const
      {
        return (_col_end - _col_begin) >> block_shift;
      }

      Row(const M& matrix, size_type i)
        : _row(i)
        , _cols(matrix.cols())
        , _col_begin(matrix.layout().colIndex() + matrix.layout().blockOffset(i >> block_shift) + (i & block_mask))
        , _col_end(_col_begin + (matrix.layout().rowLength(i) << block_shift))
        , _val_begin(matrix._data + matrix.layout().blockOffset(i >> block_shift) + (i & block_mask))
      {}

    private:

      size_type _row;
      size_type _cols;
      const size_type* _col_begin;
      const size_type* _col_end;
      value_type* _val_begin;

    };


    template<typename M, typename T>
    struct RowIterator
      : public RandomAccessIteratorFacade<RowIterator<M,T>,
                                          Dune::ISTL::ellmatrix::Row<M,T>,
                                          Dune::ISTL::ellmatrix::Row<M,T>,
                                          typename M::Allocator::difference_type
                                          >
    {

      friend class RandomAccessIteratorFacade<
        RowIterator,
        Dune::ISTL::ellmatrix::Row<M,T>,
        Dune::ISTL::ellmatrix::Row<M,T>,
        typename M::Allocator::difference_type
        >;

      typedef M Matrix;

    public:

      typedef typename M::size_type size_type;
      typedef Dune::ISTL::ellmatrix::Row<M,T> value_type;
      typedef value_type Row;
      typedef value_type reference;
      typedef typename M::Allocator::difference_type difference_type;

      static const size_type kernel_block_size = M::kernel_block_size;
      static const size_type block_shift = M::block_shift;
      static const size_type block_mask = M::block_mask;

      size_type row() const
      {
        return _row;
      }

    private:
      // keep this public for now, access from the facade is a mess!
    public:

      reference dereference() const
      {
        return Row(*_matrix,_row);
      }

      reference elementAt(size_type n) const
      {
        return Row(*_matrix,_row + n);
      }

      bool equals(const RowIterator& other) const
      {
        return _row == other._row;
      }

      void increment()
      {
        ++_row;
      }

      void decrement()
      {
        --_row;
      }

      void advance(difference_type n)
      {
        _row += n;
      }

      difference_type distanceTo(const RowIterator& other) const
      {
        return _row - other._row;
      }

      size_type _row;
      Matrix* _matrix;

    };

    } // end namespace ellmatrix
  } // end namespace ISTL
} // end namespace Dune

#endif // DUNE_ISTL_ELLMATRIX_ITERATOR_HH
