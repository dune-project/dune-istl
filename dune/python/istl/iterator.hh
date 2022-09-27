// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_ITERATOR_HH
#define DUNE_PYTHON_ISTL_ITERATOR_HH

#include <tuple>

#include <dune/common/visibility.hh>

#include <dune/python/pybind11/extensions.h>
#include <dune/python/pybind11/pybind11.h>

namespace Dune
{

  namespace Python
  {

    namespace detail
    {

      template< class  T>
      struct DUNE_PRIVATE ISTLEnumerateState
      {
        T *obj;
        pybind11::object pyObj;
      };

      template< class T >
      struct DUNE_PRIVATE ISTLIteratorState
      {
        typename T::iterator it, end;
      };

      template< class T >
      struct DUNE_PRIVATE ISTLReverseIteratorState
      {
        typename T::iterator it, end;
      };

      template< class T >
      struct DUNE_PRIVATE ISTLEnumerateIteratorState
      {
        typename T::iterator it, end;
      };

      template< class T >
      struct DUNE_PRIVATE ISTLReverseEnumerateIteratorState
      {
        typename T::iterator it, end;
      };



      // registerISTLIterator
      // --------------------

      template< class T, class... options >
      inline static void registerISTLIterators ( pybind11::class_< T, options... > cls )
      {
        if( !pybind11::already_registered< ISTLIteratorState< T > >() )
        {
          pybind11::class_< ISTLIteratorState< T > > cls( pybind11::handle(), "iterator", pybind11::module_local() );
          cls.def( "__iter__", [] ( pybind11::object self ) { return self; } );
          cls.def( "__next__", [] ( ISTLIteratorState< T > &state ) -> decltype( *state.it ) {
              if( state.it == state.end )
                throw pybind11::stop_iteration();
              decltype( *state.it ) result = *state.it;
              ++state.it;
              return result;
            }, pybind11::keep_alive< 0, 1 >() );
        }

        cls.def( "__iter__", [] ( T &self ) { return ISTLIteratorState< T >{ self.begin(), self.end() }; }, pybind11::keep_alive< 0, 1 >() );

        if( !pybind11::already_registered< ISTLReverseIteratorState< T > >() )
        {
          pybind11::class_< ISTLReverseIteratorState< T > > cls( pybind11::handle(), "iterator", pybind11::module_local() );
          cls.def( "__iter__", [] ( pybind11::object self ) { return self; } );
          cls.def( "__next__", [] ( ISTLReverseIteratorState< T > &state ) -> decltype( *state.it ) {
              if( state.it == state.end )
                throw pybind11::stop_iteration();
              decltype( *state.it ) result = *state.it;
              --state.it;
              return result;
            }, pybind11::keep_alive< 0, 1 >() );
        }

        cls.def( "__reversed__", [] ( T &self ) { return ISTLReverseIteratorState< T >{ self.beforeEnd(), self.beforeBegin() }; }, pybind11::keep_alive< 0, 1 >() );

        if( !pybind11::already_registered< ISTLEnumerateState< T > >() )
        {
          if( !pybind11::already_registered< ISTLEnumerateIteratorState< T > >() )
          {
            pybind11::class_< ISTLEnumerateIteratorState< T > > cls( pybind11::handle(), "iterator", pybind11::module_local() );
            cls.def( "__iter__", [] ( pybind11::object self ) { return self; } );
            cls.def( "__next__", [] ( pybind11::object self ) {
                auto &state = pybind11::cast< ISTLEnumerateIteratorState< T > & >( self );
                if( state.it == state.end )
                  throw pybind11::stop_iteration();
                std::tuple< decltype( state.it.index() ), pybind11::object > result( state.it.index(), pybind11::cast( *state.it ) );
                pybind11::detail::keep_alive_impl( self, std::get< 1 >( result ) );
                ++state.it;
                return result;
              } );
          }

          if( !pybind11::already_registered< ISTLReverseEnumerateIteratorState< T > >() )
          {
            pybind11::class_< ISTLReverseEnumerateIteratorState< T > > cls( pybind11::handle(), "iterator", pybind11::module_local() );
            cls.def( "__iter__", [] ( pybind11::object self ) { return self; } );
            cls.def( "__next__", [] ( pybind11::object self ) {
                auto &state = pybind11::cast< ISTLReverseEnumerateIteratorState< T > & >( self );
                if( state.it == state.end )
                  throw pybind11::stop_iteration();
                std::tuple< decltype( state.it.index() ), pybind11::object > result( state.it.index(), pybind11::cast( *state.it ) );
                pybind11::detail::keep_alive_impl( self, std::get< 1 >( result ) );
                --state.it;
                return result;
              } );
          }

          pybind11::class_< ISTLEnumerateState< T > > cls( pybind11::handle(), "enumerate", pybind11::module_local() );
          cls.def( "__iter__", [] ( ISTLEnumerateState< T > &self ) {
              return ISTLEnumerateIteratorState< T >{ self.obj->begin(), self.obj->end() };
            }, pybind11::keep_alive< 0, 1 >() );
          cls.def( "__reversed__", [] ( ISTLEnumerateState< T > &self ) {
              return ISTLReverseEnumerateIteratorState< T >{ self.obj->beforeEnd(), self.obj->beforeBegin() };
            }, pybind11::keep_alive< 0, 1 >() );
        }

        cls.def_property_readonly( "enumerate", [] ( pybind11::object self ) {
            return ISTLEnumerateState< T >{ &pybind11::cast< T & >( self ), self };
          } );
      }

    } // namespace detail

  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_ITERATOR_HH
