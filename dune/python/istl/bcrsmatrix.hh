// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
#ifndef DUNE_PYTHON_ISTL_BCRSMATRIX_HH
#define DUNE_PYTHON_ISTL_BCRSMATRIX_HH

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <dune/python/common/fvecmatregistry.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/operators.hh>

#include <dune/python/pybind11/pybind11.h>
#include <dune/python/pybind11/stl.h>
#include <dune/python/istl/bvector.hh>
#include <dune/python/istl/iterator.hh>
#include <dune/python/istl/operators.hh>

namespace Dune
{

  namespace Python
  {

    namespace detail
    {

      template< class Matrix >
      struct CorrespondingVectors;

      template< class B, class A >
      struct CorrespondingVectors< BCRSMatrix< B, A > >
      {
        typedef BlockVector< typename CorrespondingVectors< B >::Domain, typename std::allocator_traits< A >::template rebind_alloc< typename CorrespondingVectors< B >::Domain > > Domain;
        typedef BlockVector< typename CorrespondingVectors< B >::Range, typename std::allocator_traits< A >::template rebind_alloc< typename CorrespondingVectors< B >::Range > > Range;
      };

      template< class K, int ROWS, int COLS >
      struct CorrespondingVectors< FieldMatrix< K, ROWS, COLS > >
      {
        typedef FieldVector< K, COLS > Domain;
        typedef FieldVector< K, ROWS > Range;
      };

    } // namespace detail



    // CorrespondingDomainVector
    // -------------------------

    template< class Matrix >
    using CorrespondingDomainVector = typename detail::CorrespondingVectors< Matrix >::Domain;




    // CorrespondingRangeVector
    // ------------------------

    template< class Matrix >
    using CorrespondingRangeVector = typename detail::CorrespondingVectors< Matrix >::Range;



    // registerBCRSMatrix
    // ------------------

    template <class BCRSMatrix, class... options>
    void registerBCRSMatrix(pybind11::handle scope,
                            pybind11::class_<BCRSMatrix, options...> cls)
    {
      using pybind11::operator""_a;
      typedef typename BCRSMatrix::block_type block_type;
      typedef typename BCRSMatrix::field_type field_type;
      typedef typename BCRSMatrix::size_type Size;
      typedef typename BCRSMatrix::row_type row_type;

      typedef typename BCRSMatrix::BuildMode BuildMode;

      registerFieldVecMat<block_type>::apply();

      pybind11::class_< row_type > clsRow( scope, "BCRSMatrixRow" );
      registerBlockVector( clsRow );

      pybind11::enum_< typename BCRSMatrix::BuildStage > bs( cls, "BuildStage" );
      bs.value( "notAllocated", BCRSMatrix::notAllocated );
      bs.value( "building", BCRSMatrix::building );
      bs.value( "rowSizesBuilt", BCRSMatrix::rowSizesBuilt );
      bs.value( "built", BCRSMatrix::built );
      bs.export_values();

      cls.def( pybind11::init( [] () { return new BCRSMatrix(); } ) );
      typedef Dune::BCRSMatrix< Dune::FieldMatrix< double, 1, 1 > > Matrix;
      using BuildMode11 = Matrix::BuildMode;
      cls.def( pybind11::init( [] ( Size rows, Size cols, Size nnz, BuildMode11 buildMode ) { return new BCRSMatrix( rows, cols, nnz, static_cast<BuildMode>(buildMode) ); } ), "rows"_a, "cols"_a, "nnz"_a = 0, "buildMode"_a );
      cls.def( pybind11::init( [] ( std::tuple< Size, Size > shape, Size nnz, BuildMode11 buildMode ) { return new BCRSMatrix( std::get< 0 >( shape ), std::get< 1 >( shape ), nnz, static_cast<BuildMode>(buildMode) ); } ), "shape"_a, "nnz"_a = 0, "buildMode"_a );
      cls.def( pybind11::init( [] ( Size rows, Size cols, Size average, double overflow, BuildMode11 buildMode ) { return new BCRSMatrix( rows, cols, average, overflow, static_cast<BuildMode>(buildMode) ); } ), "rows"_a, "cols"_a, "average"_a, "overflow"_a, "buildMode"_a );
      cls.def( pybind11::init( [] ( std::tuple< Size, Size > shape, Size average, double overflow, BuildMode11 buildMode ) { return new BCRSMatrix( std::get< 0 >( shape ), std::get< 1 >( shape ), average, overflow, static_cast<BuildMode>(buildMode) ); } ), "shape"_a, "average"_a, "overflow"_a, "buildMode"_a );
      if (!std::is_same<Matrix,BCRSMatrix>::value)
      {
        pybind11::enum_< BuildMode > bm( cls, "BuildMode" );
        bm.value( "row_wise", BCRSMatrix::row_wise );
        bm.value( "random", BCRSMatrix::random );
        bm.value( "implicit", BCRSMatrix::implicit );
        // bm.value( "unknown", BCRSMatrix::unknown );
        bm.export_values();
        cls.def( pybind11::init( [] ( Size rows, Size cols, Size nnz, BuildMode buildMode ) { return new BCRSMatrix( rows, cols, nnz, static_cast<BuildMode>(buildMode) ); } ), "rows"_a, "cols"_a, "nnz"_a = 0, "buildMode"_a );
        cls.def( pybind11::init( [] ( std::tuple< Size, Size > shape, Size nnz, BuildMode buildMode ) { return new BCRSMatrix( std::get< 0 >( shape ), std::get< 1 >( shape ), nnz, buildMode ); } ), "shape"_a, "nnz"_a = 0, "buildMode"_a );
        cls.def( pybind11::init( [] ( Size rows, Size cols, Size average, double overflow, BuildMode buildMode ) { return new BCRSMatrix( rows, cols, average, overflow, buildMode ); } ), "rows"_a, "cols"_a, "average"_a, "overflow"_a, "buildMode"_a );
        cls.def( pybind11::init( [] ( std::tuple< Size, Size > shape, Size average, double overflow, BuildMode buildMode ) { return new BCRSMatrix( std::get< 0 >( shape ), std::get< 1 >( shape ), average, overflow, buildMode ); } ), "shape"_a, "average"_a, "overflow"_a, "buildMode"_a );
      }

      detail::registerISTLIterators( cls );

      // shape
      cls.def_property_readonly( "rows", [] ( const BCRSMatrix &self ) { return self.N(); } );
      cls.def_property_readonly( "cols", [] ( const BCRSMatrix &self ) { return self.M(); } );
      cls.def_property_readonly( "shape", [] ( const BCRSMatrix &self ) { return std::make_tuple( self.N(), self.M() ); } );
      cls.def_property_readonly( "nonZeroes", [] ( const BCRSMatrix &self ) { return self.nonzeroes(); } );

      cls.def( "setSize", [] ( BCRSMatrix &self, Size rows, Size cols, Size nnz ) { self.setSize( rows, cols, nnz ); }, "rows"_a, "cols"_a, "nnz"_a = 0 );
      cls.def( "setSize", [] ( BCRSMatrix &self, std::tuple< Size, Size > shape, Size nnz ) { self.setSize( std::get< 0 >( shape ), std::get< 1 >( shape ), nnz ); }, "shape"_a, "nnz"_a = 0 );

      // build parameters
      cls.def_property( "buildMode", [] ( const BCRSMatrix &self ) {
          BuildMode bm = self.buildMode();
          if( bm == BCRSMatrix::unknown )
            return pybind11::object();
          else
            return pybind11::cast( bm );
        }, [] ( BCRSMatrix &self, BuildMode buildMode ) {
          self.setBuildMode( buildMode );
        } );
      cls.def_property_readonly( "buildStage", [] ( const BCRSMatrix &self ) { return self.buildStage(); } );
      cls.def( "setImplicitBuildModeParameters", [] ( BCRSMatrix &self, Size average, double overflow ) { self.setImplicitBuildModeParameters( average, overflow ); }, "average"_a, "overflow"_a );

      // random build mode
      cls.def( "setRowSize", [] ( BCRSMatrix &self, Size row, Size size ) { self.setrowsize( row, size ); }, "row"_a, "size"_a );
      cls.def( "getRowSize", [] ( const BCRSMatrix &self, Size row ) { return self.getrowsize( row ); }, "row"_a );
      cls.def( "endRowSizes", [] ( BCRSMatrix &self ) { self.endrowsizes(); } );

      cls.def( "addIndex", [] ( BCRSMatrix &self, Size row, Size col ) { self.addindex( row, col ); }, "row"_a, "col"_a );
      cls.def( "addIndex", [] ( BCRSMatrix &self, std::tuple< Size, Size > index ) { self.addindex( std::get< 0 >( index ), std::get< 1 >( index ) ); }, "index"_a );
      cls.def( "setIndices", [] ( BCRSMatrix &self, Size row, pybind11::buffer buffer ) {
          pybind11::buffer_info info = buffer.request();
          if( info.format != pybind11::format_descriptor< Size >::format() )
            throw std::invalid_argument( "Incompatible buffer format." );
          if( (info.ndim != 1) || (static_cast< Size >( info.shape[ 0 ] ) != self.getrowsize( row )) )
            throw std::invalid_argument( "Indices must be a flat array with length" + std::to_string( self.getrowsize( row ) ) + "." );
          self.setIndices( row, static_cast< const Size * >( info.ptr ), static_cast< const Size * >( info.ptr ) + info.shape[ 0 ] );
        }, "row"_a, "indices"_a );
      cls.def( "setIndices", [] ( BCRSMatrix &self, Size row, std::vector< Size > indices ) {
          if( static_cast< Size >( indices.size() ) != self.getrowsize( row ) )
            throw std::invalid_argument( "len( indices ) must match previously set row size, i.e., " + std::to_string( self.getrowsize( row ) ) + "." );
          self.setIndices( row, indices.begin(), indices.end() );
        } );
      cls.def( "endIndices", [] ( BCRSMatrix &self ) { self.endindices(); } );

      // implicit build mode
      cls.def( "compress", [] ( BCRSMatrix &self ) {
          auto result = self.compress();
          return std::make_tuple( result.maximum, result.overflow_total, result.avg, result.mem_ratio );
        } );

      // index access
      cls.def( "__contains__", [] ( const BCRSMatrix &self, std::tuple< Size, Size > index ) { return self.exists( std::get< 0 >( index ), std::get< 1 >( index ) ); } );

      cls.def( "__getitem__", [] ( BCRSMatrix &self, Size row ) -> row_type & {
          if( row >= self.N() )
            throw pybind11::index_error( "No such row: " + std::to_string( row ) );
          return self[ row ];
        }, pybind11::return_value_policy::reference); // , pybind11::keep_alive< 0, 1 >() );
      cls.def( "__getitem__", [] ( BCRSMatrix &self, std::tuple< Size, Size > index ) -> block_type & {
          const Size row = std::get< 0 >( index );
          if( row >= self.N() )
            throw pybind11::index_error( "No such row: " + std::to_string( row ) );

          const Size col = std::get< 1 >( index );
          if( col >= self.M() )
            throw pybind11::index_error( "No such column: " + std::to_string( col ) );

          if( self.buildMode() != BCRSMatrix::implicit || self.buildStage() == BCRSMatrix::built )
          {
            auto pos = self[ row ].find( col );
            if( pos != self[ row ].end() )
              return *pos;
            else
              throw pybind11::index_error( "Index (" + std::to_string( row ) + ", " + std::to_string( col ) + ") not in sparsity pattern." );
          }
          else
            return self.entry( row, col );
        }, pybind11::return_value_policy::reference, pybind11::keep_alive< 0, 1 >() );

      cls.def( "__setitem__", [] ( BCRSMatrix &self, Size row, row_type &value ) {
          if( row >= self.N() )
            throw pybind11::index_error( "No such row: " + std::to_string( row ) );
          if( &value != &self[ row ] )
            throw pybind11::value_error( "Can only assign row to itself" );
        } );
      cls.def( "__setitem__", [] ( BCRSMatrix &self, std::tuple< Size, Size > index, block_type value ) {
          const Size row = std::get< 0 >( index );
          if( row >= self.N() )
            throw pybind11::index_error( "No such row: " + std::to_string( row ) );

          const Size col = std::get< 1 >( index );
          if( col >= self.M() )
            throw pybind11::index_error( "No such column: " + std::to_string( col ) );

          if( self.buildMode() != BCRSMatrix::implicit || self.buildStage() == BCRSMatrix::built )
          {
            auto pos = self[ row ].find( col );
            if( pos != self[ row ].end() )
              *pos = value;
            else
              throw pybind11::index_error( "Index (" + std::to_string( row ) + ", " + std::to_string( col ) + ") not in sparsity pattern." );
          }
          else
            self.entry( row, col ) = value;
        } );
      cls.def( "__setitem__", [] ( BCRSMatrix &self, std::tuple< Size, pybind11::slice > index, pybind11::iterable value ) {
          const Size row = std::get< 0 >( index );
          if( row > self.N() )
            throw pybind11::index_error( "No such row" );

          std::size_t cstart, cstop, cstep, clength;
          std::get< 1 >( index ).compute( self.M(), &cstart, &cstop, &cstep, &clength );

          if( self.buildMode() != BCRSMatrix::implicit || self.buildStage() == BCRSMatrix::built )
          {
            for( auto v : value )
            {
              if( cstart >= cstop )
                throw pybind11::value_error( "too many values passed" );
              self.entry( row, cstart ) = pybind11::cast< block_type >( v );
              cstart += cstep;
            }
            if( cstart < cstop )
              throw pybind11::value_error( "too few values passed" );
          }
          else
          {
            for( auto v : value )
            {
              if( cstart >= cstop )
                throw pybind11::value_error( "too many values passed" );
              auto pos = self[ row ].find( cstart );
              if( pos != self[ row ].end() )
                *pos = pybind11::cast< block_type >( v );
              else
                throw pybind11::index_error();
              cstart += cstep;
            }
            if( cstart < cstop )
              throw pybind11::value_error( "too few values passed" );
          }
        } );

      // matrix-vector multiplication
      cls.def( "mv", [] ( const BCRSMatrix &self, const CorrespondingDomainVector< BCRSMatrix > &x, CorrespondingRangeVector< BCRSMatrix > &y ) {
          return self.mv( x, y );
        }, "x"_a, "y"_a );
      cls.def( "umv", [] ( const BCRSMatrix &self, const CorrespondingDomainVector< BCRSMatrix > &x, CorrespondingRangeVector< BCRSMatrix > &y ) {
          return self.umv( x, y );
        }, "x"_a, "y"_a );
      cls.def( "mmv", [] ( const BCRSMatrix &self, const CorrespondingDomainVector< BCRSMatrix > &x, CorrespondingRangeVector< BCRSMatrix > &y ) {
          return self.mmv( x, y );
        }, "x"_a, "y"_a );
      cls.def( "usmv", [] ( const BCRSMatrix &self, const field_type &alpha, const CorrespondingDomainVector< BCRSMatrix > &x, CorrespondingRangeVector< BCRSMatrix > &y ) {
          return self.usmv( alpha, x, y );
        }, "alpha"_a, "x"_a, "y"_a );

      cls.def( "mtv", [] ( const BCRSMatrix &self, const CorrespondingRangeVector< BCRSMatrix > &x, CorrespondingDomainVector< BCRSMatrix > &y ) {
          return self.mtv( x, y );
        }, "x"_a, "y"_a );
      cls.def( "umtv", [] ( const BCRSMatrix &self, const CorrespondingRangeVector< BCRSMatrix > &x, CorrespondingDomainVector< BCRSMatrix > &y ) {
          return self.umtv( x, y );
        }, "x"_a, "y"_a );
      cls.def( "mmtv", [] ( const BCRSMatrix &self, const CorrespondingRangeVector< BCRSMatrix > &x, CorrespondingDomainVector< BCRSMatrix > &y ) {
          return self.mmtv( x, y );
        }, "x"_a, "y"_a );
      cls.def( "usmtv", [] ( const BCRSMatrix &self, const field_type &alpha, const CorrespondingRangeVector< BCRSMatrix > &x, CorrespondingDomainVector< BCRSMatrix > &y ) {
          return self.usmtv( alpha, x, y );
        }, "alpha"_a, "x"_a, "y"_a );

      // norms
      cls.def_property_readonly( "frobenius_norm", [] ( const BCRSMatrix &self ) { return self.frobenius_norm(); } );
      cls.def_property_readonly( "frobenius_norm2", [] ( const BCRSMatrix &self ) { return self.frobenius_norm2(); } );
      cls.def_property_readonly( "infinity_norm", [] ( const BCRSMatrix &self ) { return self.infinity_norm(); } );
      cls.def_property_readonly( "infinity_norm_real", [] ( const BCRSMatrix &self ) { return self.infinity_norm_real(); } );

      // io
      cls.def( "load", [] ( BCRSMatrix &self, const std::string &fileName, std::string format ) {
          if( (format == "matrixmarket") || (format == "mm") )
            loadMatrixMarket( self, fileName );
          else
            throw std::invalid_argument( "Unknown format: "  + format );
        }, "fileName"_a, "format"_a );
      cls.def( "store", [] ( const BCRSMatrix &self, const std::string &fileName, std::string format ) {
          if( (format == "matrixmarket") || (format == "mm") )
            storeMatrixMarket( self, fileName );
          else if( format == "matlab" )
            writeMatrixToMatlab( self, fileName );
          else
            throw std::invalid_argument( "Unknown format: "  + format );
        }, "fileName"_a, "format"_a );

      // linear operator
      typedef Dune::LinearOperator< CorrespondingDomainVector< BCRSMatrix >, CorrespondingRangeVector< BCRSMatrix > > LinearOperator;

      cls.def( "asLinearOperator", [] ( const BCRSMatrix &self ) -> LinearOperator * {
          return new MatrixAdapter< BCRSMatrix, CorrespondingDomainVector< BCRSMatrix >, CorrespondingRangeVector< BCRSMatrix > >( self );
        }, pybind11::keep_alive< 0, 1 >() );

      //needed for import and exporting the matrix index set
      cls.def("exportTo", [] ( BCRSMatrix &self, MatrixIndexSet &mis ) {mis.import(self);  } );
      cls.def("importFrom", [] ( BCRSMatrix &self, MatrixIndexSet &mis ) {mis.exportIdx(self);  } );
    }


    template< class BCRSMatrix >
    pybind11::class_< BCRSMatrix > registerBCRSMatrix ( pybind11::handle scope, const char *clsName = "BCRSMatrix" )
    {
      //pybind11::class_< BCRSMatrix > cls( scope, clsName );
      int rows = BCRSMatrix::block_type::rows;
      int cols = BCRSMatrix::block_type::cols;

      std::string matrixTypename = "Dune::BCRSMatrix< Dune::FieldMatrix< double, "+ std::to_string(rows) + ", " + std::to_string(cols) + " > >";

      auto cls = Dune::Python::insertClass< BCRSMatrix >( scope, clsName, Dune::Python::GenerateTypeName(matrixTypename), Dune::Python::IncludeFiles{"dune/istl/bcrsmatrix.hh","dune/python/istl/bcrsmatrix.hh"});
      if(cls.second)
      {
        registerBCRSMatrix( scope, cls.first );
      }
      return cls.first;
    }

  } // namespace Python

} // namespace Dune

#endif // #ifndef DUNE_PYTHON_ISTL_BCRSMATRIX_HH
