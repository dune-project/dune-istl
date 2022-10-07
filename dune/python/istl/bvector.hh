// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_BVECTOR_HH
#define DUNE_PYTHON_ISTL_BVECTOR_HH

#include <cstddef>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <dune/common/typeutilities.hh>

//this added otherwise insert class wasn't possible on line ~190
#include <dune/python/common/typeregistry.hh>
#include <dune/python/common/fvecmatregistry.hh>
#include <dune/python/common/string.hh>
#include <dune/python/common/vector.hh>
#include <dune/python/istl/iterator.hh>
#include <dune/python/pybind11/operators.h>
#include <dune/python/pybind11/pybind11.h>

#include <dune/istl/bvector.hh>
#include <dune/istl/blocklevel.hh>

namespace Dune
{

  namespace Python
  {

    namespace detail
    {

      template< class K, int n >
      inline static void copy ( const char *ptr, const ssize_t *shape, const ssize_t *strides, Dune::FieldVector< K, n > &v )
      {
        if( *shape != static_cast< ssize_t >( n ) )
          throw pybind11::value_error( "Invalid buffer size: " + std::to_string( *shape ) + " (should be: " + std::to_string( n ) + ")." );

        for( ssize_t i = 0; i < static_cast< ssize_t >( n ); ++i )
          v[ i ] = *reinterpret_cast< const K * >( ptr + i*(*strides) );
      }


      template< class B, class A >
      inline static void copy ( const char *ptr, const ssize_t *shape, const ssize_t *strides, Dune::BlockVector< B, A > &v )
      {
        v.resize( *shape );
        for( ssize_t i = 0; i < *shape; ++i )
          copy( ptr + i*(*strides), shape+1, strides+1, v[ i ] );
      }


      template< class BlockVector >
      inline static void copy ( pybind11::buffer buffer, BlockVector &v )
      {
        typedef typename BlockVector::field_type field_type;

        pybind11::buffer_info info = buffer.request();

        if( info.format != pybind11::format_descriptor< field_type >::format() )
          throw pybind11::value_error( "Incompatible buffer format." );
        if( info.ndim != blockLevel<BlockVector>() )
          throw pybind11::value_error( "Block vectors can only be initialized from one-dimensional buffers." );

        copy( static_cast< const char * >( info.ptr ), info.shape.data(), info.strides.data(), v );
      }



      // blockVectorGetItem
      // ------------------

      template< class BlockVector >
      inline static pybind11::object blockVectorGetItem ( const pybind11::object &vObj, BlockVector &v, typename BlockVector::size_type index )
      {
        auto pos = v.find( index );
        if( pos == v.end() )
          throw pybind11::index_error( "Index " + std::to_string( index ) + " does not exist in block vector." );
        pybind11::object result = pybind11::cast( *pos, pybind11::return_value_policy::reference );
        pybind11::detail::keep_alive_impl( result, vObj );
        return result;
      }

    } // namespace detail



    // to_string
    // ---------

    template< class X >
    inline static auto to_string ( const X &x )
      -> std::enable_if_t< std::is_base_of< Imp::block_vector_unmanaged< typename X::block_type, typename X::allocator_type >, X >::value, std::string >
    {
      return "(" + join( ", ", [] ( auto &&x ) { return to_string( x ); }, x.begin(), x.end() ) + ")";
    }



    // registserBlockVector
    // --------------------

    template< class BlockVector, class... options >
    inline void registerBlockVector ( pybind11::class_< BlockVector, options... > cls )
    {
      typedef typename BlockVector::field_type field_type;
      typedef typename BlockVector::block_type block_type;
      typedef typename BlockVector::size_type size_type;

      registerFieldVecMat<block_type>::apply();

      using pybind11::operator""_a;

      cls.def( "assign", [] ( BlockVector &self, const BlockVector &x ) { self = x; }, "x"_a );

      cls.def( "copy", [] ( const BlockVector &self ) { return new BlockVector( self ); } );

      cls.def( "__getitem__", [] ( const pybind11::object &self, size_type index ) {
          return detail::blockVectorGetItem( self, pybind11::cast< BlockVector & >( self ), index );
        } );
      cls.def( "__getitem__", [] ( const pybind11::object &self, pybind11::iterable index ) {
          BlockVector &v = pybind11::cast< BlockVector & >( self );
          pybind11::tuple refs( pybind11::len( index ) );
          std::size_t j = 0;
          for( pybind11::handle i : index )
            refs[ j++ ] = detail::blockVectorGetItem( self, v, pybind11::cast< size_type >( i ) );
          return refs;
        } );

      cls.def( "__setitem__", [] ( BlockVector &self, size_type index, block_type value ) {
          auto pos = self.find( index );
          if( pos != self.end() )
            *pos = value;
          else
            throw pybind11::index_error();
        } );
      cls.def( "__setitem__", [] ( BlockVector &self, pybind11::slice index, pybind11::iterable value ) {
          std::size_t start, stop, step, length;
          index.compute( self.N(), &start, &stop, &step, &length );
          for( auto v : value )
          {
            if( start >= stop )
              throw pybind11::value_error( "too many values passed" );
            auto pos = self.find( start );
            if( pos != self.end() )
              *pos = pybind11::cast< block_type >( v );
            else
              throw pybind11::index_error();
            start += step;
          }
          if( start < stop )
            throw pybind11::value_error( "too few values passed" );
        } );

      cls.def( "__len__", [] ( const BlockVector &self ) { return self.N(); } );

      cls.def( pybind11::self += pybind11::self );
      cls.def( pybind11::self -= pybind11::self );

      detail::registerOneTensorInterface( cls );
      detail::registerISTLIterators( cls );

      cls.def( "__imul__", [] ( BlockVector &self, field_type x ) -> BlockVector & { self *= x; return self; } );
      cls.def( "__idiv__", [] ( BlockVector &self, field_type x ) -> BlockVector & { self /= x; return self; } );
      cls.def( "__itruediv__", [] ( BlockVector &self, field_type x ) -> BlockVector & { self /= x; return self; } );

      cls.def( "__add__", [] ( const BlockVector &self, const BlockVector &x ) { BlockVector *copy = new BlockVector( self ); *copy += x; return copy; } );
      cls.def( "__sub__", [] ( const BlockVector &self, const BlockVector &x ) { BlockVector *copy = new BlockVector( self ); *copy -= x; return copy; } );

      cls.def( "__div__", [] ( const BlockVector &self, field_type x ) { BlockVector *copy = new BlockVector( self ); *copy /= x; return copy; } );
      cls.def( "__truediv__", [] ( const BlockVector &self, field_type x ) { BlockVector *copy = new BlockVector( self ); *copy /= x; return copy; } );
      cls.def( "__mul__", [] ( const BlockVector &self, field_type x ) { BlockVector *copy = new BlockVector( self ); *copy *= x; return copy; } );
      cls.def( "__rmul__", [] ( const BlockVector &self, field_type x ) { BlockVector *copy = new BlockVector( self ); *copy *= x; return copy; } );
    }



    // registserBlockVector
    // --------------------

    //for the new bindings and arbitrary block size haven't
    //the generator acutally takes the scope into account which is why we do nothing with it here
    //so when doing a dune.istl blockvector it doesn't actually define any of the rest ofthe bindings
    template< class BlockVector, class ... options >
    void registerBlockVector ( pybind11::handle scope, pybind11::class_<BlockVector, options ... > cls )
    {
      typedef typename BlockVector::size_type size_type;
      using pybind11::operator""_a;

      registerBlockVector( cls );

      cls.def( pybind11::init( [] () { return new BlockVector(); } ) );
      cls.def( pybind11::init( [] ( size_type size ) { return new BlockVector( size ); } ), "size"_a );

      cls.def( pybind11::init( [] ( pybind11::buffer buffer ) {
          BlockVector *self = new BlockVector();
          detail::copy( buffer, *self );
          return self;
        } ) );


      // cls.def( "__str__", [] ( const BlockVector &self ) { return to_string( self ); } );

      cls.def( "assign", [] ( BlockVector &self, pybind11::buffer buffer ) { detail::copy( buffer, self ); }, "buffer"_a );

      cls.def_property_readonly( "capacity", [] ( const BlockVector &self ) { return self.capacity(); } );

      cls.def( "resize", [] ( BlockVector &self, size_type size ) { self.resize( size ); }, "size"_a );

    }

    //the auto class is needed so that run.algorithm can properly work
    template< class BlockVector >
    inline pybind11::class_< BlockVector > registerBlockVector ( pybind11::handle scope, const char *clsName = "BlockVector" )
    {
      //typedef typename BlockVector::size_type size_type;

      using pybind11::operator""_a;

      int rows = BlockVector::block_type::dimension;
      std::string vectorTypename = "Dune::BlockVector< Dune::FieldVector< double, "+ std::to_string(rows) + " > >";
      auto cls = Dune::Python::insertClass< BlockVector >( scope, clsName, Dune::Python::GenerateTypeName(vectorTypename), Dune::Python::IncludeFiles{"dune/istl/bvector.hh","dune/python/istl/bvector.hh"});

      if (cls.second)
        registerBlockVector( scope, cls.first );
      return cls.first;
    }



  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_BVECTOR_HH
