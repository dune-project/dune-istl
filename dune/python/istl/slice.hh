// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_SLICE_HH
#define DUNE_PYTHON_ISTL_SLICE_HH

#include <cassert>

#include <type_traits>

#include <dune/common/iteratorfacades.hh>

namespace Dune
{

  namespace Python
  {

    // ArraySlice
    // ----------

    template< class A >
    struct ArraySlice
    {
      typedef typename A::member_type member_type;
      typedef typename A::size_type size_type;

      typedef decltype( std::declval< A & >()[ 0 ] ) reference;
      typedef decltype( std::declval< const A & >()[ 0 ] ) const_reference;

    private:
      template< class T >
      struct IteratorImpl
        : public RandomAccessIteratorFacade< IteratorImpl< T >, T >
      {
        friend class IteratorImpl< std::add_const_t< T > >;
        friend class IteratorImpl< std::remove_const_t< T > >;

        friend class RandomAccessIteratorFacade< IteratorImpl< std::add_const_t< T > >, std::add_const_t< T > >;
        friend class RandomAccessIteratorFacade< IteratorImpl< std::remove_const_t< T > >, std::remove_const_t< T > >;

        IteratorImpl () noexcept = default;
        IteratorImpl ( T &array, size_type start, size_type step ) noexcept : array_( &array ), index_( start ), step_( step ) {}
        IteratorImpl ( const IteratorImpl< std::remove_const_t< T > > &other ) noexcept : array_( other.array_ ), index_( other.index_ ), step_( other.step_ ) {}

      private:
        bool equals ( const IteratorImpl &other ) const noexcept { return (index_ == other.index_); }

        std::ptrdiff_t distanceTo ( const IteratorImpl &other ) const noexcept { return (other.index_ - index_) / step_; }

        reference elementAt ( std::ptrdiff_t i ) const noexcept( noexcept( std::declval< T & >()[ 0 ] ) ) { return (*array_)[ index_ ]; }

        void increment () noexcept { index_ += step_; }
        void decrement () noexcept { index_ -= step_; }
        void advance ( std::ptrdiff_t i ) noexcept { index_ += i*step_; }

        T *array_ = nullptr;
        size_type index_ = 0, step_ = 0;
      };

    public:
      typedef IteratorImpl< const A > const_iterator;
      typedef IteratorImpl< A > iterator;

      ArraySlice ( A &array, size_type start, size_type step, size_type size ) noexcept
        : array_( array ), start_( start ), step_( step ), size_( size )
      {}

      reference operator[] ( size_type i ) noexcept( noexcept( std::declval< A & >()[ 0 ] ) ) { return array_[ start_ + i*step_ ]; }
      const_reference operator[] ( size_type i ) const noexcept( noexcept( std::declval< const A & >()[ 0 ] ) ) { return array_[ start_ + i*step_ ]; }

      const_iterator begin () const noexcept { return const_iterator( array_, start_, step_ ); }
      iterator begin () noexcept { return iterator( array_, start_, step_); }

      const_iterator end () const noexcept { return const_iterator( array_, start_ + step_*size_, step_ ); }
      iterator end () noexcept { return iterator( array_, start_ + step_*size_, step_ ); }

      const_iterator beforeEnd () const noexcept { return const_iterator( array_, start_ + step_*(size_-1), step_ ); }
      iterator beforeEnd () noexcept { return iterator( array_, start_ + step_*(size_-1), step_ ); }

      const_iterator beforeBegin () const noexcept { return const_iterator( array_, start_ - step_, step_ ); }
      iterator beforeBegin () noexcept { return iterator( array_, start_ - step_, step_ ); }

      const_iterator find ( size_type i ) const noexcept { return iterator( array_, start_ + step_*std::min( i, size_ ), step_ ); }
      iterator find ( size_type i ) noexcept { return iterator( array_, start_ + step_*std::min( i, size_ ), step_ ); }

      size_type size () const noexcept { return size_; }

    private:
      A &array_;
      size_type start_, step_, size_;
    };

  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_SLICE_HH
