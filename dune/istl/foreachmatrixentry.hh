// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FOREACHMATRIXENTRY_HH
#define DUNE_ISTL_FOREACHMATRIXENTRY_HH

#include<type_traits>

#include<dune/common/hybridutilities.hh>
#include<dune/common/indices.hh>
#include<dune/common/typetraits.hh>
#include <dune/common/std/type_traits.hh>

#include<dune/istl/multitypeblockvector.hh>

namespace Dune {

  namespace Detail {

    // stolen from dune-functions
    template<class C>
    using staticIndexAccess_t = decltype(std::declval<C>()[Dune::Indices::_0]);

    template<class C>
    using isScalar = Dune::Std::bool_constant<not Dune::Std::is_detected_v<staticIndexAccess_t, std::remove_reference_t<C>>>;

    //! a small helper that removes the first entry of an std::tuple
    template<class T, class... Args>
    auto tupleTail(const std::tuple<T,Args...>& t)
    {
      // apply make_tuple to the tail
      return std::apply(
        [](auto&& /*firstElement*/, auto&& ... tail) { return std::make_tuple(tail...); }, t
      );
    }

    //! a small helper that picks the first entry of an std::tuple
    template<class... Args>
    auto tupleHead(const std::tuple<Args...>& t)
    {
      return std::get<0>(t);
    }

    template<class T>
    struct ForEachChooser;

    template<class M, class GlobalRowIndex, class RowIndex, class ColIndex, class F>
    void forEachScalarRowEntry( const M& m, const GlobalRowIndex& globalRowIndex, const RowIndex& rowIndex, const ColIndex& colIndex, F&& f )
    {
      if constexpr( std::tuple_size_v<RowIndex> == 0 )
        f( globalRowIndex, colIndex, m );
      else
      {
        auto firstIndex = Detail::tupleHead(rowIndex);
        auto indexTail  = Detail::tupleTail(rowIndex);
        auto row = m[firstIndex];
        for ( auto colIt = row.begin(); colIt != row.end(); colIt++ )
        {
          auto&& block = *colIt;
          using B = std::decay_t<decltype(block)>;
          auto&& newColIndex = std::tuple_cat( colIndex, std::tuple{ colIt.index() } );
          ForEachChooser<B>::entry( block, globalRowIndex, indexTail, newColIndex, f );
        }
      }
    }


    template<class FirstRow, class... Rows, class GlobalRowIndex, class RowIndex, class ColIndex, class F>
    void multiTypeForEachScalarRowEntry( const MultiTypeBlockMatrix<FirstRow,Rows...>& m, const GlobalRowIndex& globalRowIndex, const RowIndex& rowIndex, const ColIndex& colIndex, F&& f )
    {
      auto firstIndex = Detail::tupleHead(rowIndex);
      auto indexTail  = Detail::tupleTail(rowIndex);
      auto cols = index_constant<FirstRow::size()>();
      Hybrid::forEach(Hybrid::integralRange(cols), [&](auto&& i){
        auto&& block = m[firstIndex][i];
        auto&& newColIndex = std::tuple_cat( colIndex, std::tuple{i} );
        using B = std::decay_t<decltype(block)>;
        ForEachChooser<B>::entry( block, globalRowIndex, indexTail, newColIndex, f );
      });;
    }


    template<class FirstRow, class... Rows, class I, class F>
    void multiTypeForEachScalarRow( const MultiTypeBlockMatrix<FirstRow,Rows...>& m, const I& i, F&& f )
    {
      auto rows = index_constant<MultiTypeBlockMatrix<FirstRow,Rows...>::N()>();
      Hybrid::forEach(Hybrid::integralRange(rows), [&](auto&& rowIdx){
        auto&& block = m[rowIdx][Indices::_0];
        using B = std::decay_t<decltype(block)>;
        ForEachChooser<B>::row( block, std::tuple_cat( i, std::tuple {rowIdx} ), f );
      });;
    }

    template<class M, class I, class F >
    void forEachScalarRow_impl( const M& m, const I& i, F&& f )
    {
      if constexpr( isScalar<std::decay_t<decltype(m)>>::value )
        f(i);
      else
        for ( auto rowIt = m.begin(); rowIt != m.end(); rowIt++ )
        {
          auto&& row = *rowIt;
          auto&& newRowIndex = std::tuple_cat( i, std::tuple { rowIt.index() } );
          auto it = row.begin();
          // If there is no block we have no other chance than creating a static dummy!
          // From now on we must assume that all blocks have static size!
          using B = typename M::block_type;
          auto&& block = ( it != row.end() ) ? *it : B() ;
          ForEachChooser<B>::row( block, newRowIndex, f );
        }
    }

    template<class M>
    struct ForEachChooser
    {
      template<class I, class F>
      static void row( const M& m, const I& i, F&& f )
      {
        forEachScalarRow_impl(m,i,f);
      }
      template<class GlobalRowIndex, class RowIndex, class ColIndex, class F>
      static void entry( const M& m, const GlobalRowIndex& globalRowIndex, const RowIndex& rowIndex, const ColIndex& colIndex, F&& f )
      {
        forEachScalarRowEntry( m, globalRowIndex, rowIndex, colIndex, f );
      }
    };

    template<class FirstRow, class... Rows>
    struct ForEachChooser<MultiTypeBlockMatrix<FirstRow,Rows...>>
    {
      template<class I, class F>
      static void row( const MultiTypeBlockMatrix<FirstRow,Rows...>& m, const I& i, F&& f )
      {
        multiTypeForEachScalarRow(m,i,f);
      }
      template<class GlobalRowIndex, class RowIndex, class ColIndex, class F>
      static void entry( const MultiTypeBlockMatrix<FirstRow,Rows...>& m, const GlobalRowIndex& globalRowIndex, const RowIndex& rowIndex, const ColIndex& colIndex, F&& f )
      {
        multiTypeForEachScalarRowEntry( m, globalRowIndex, rowIndex, colIndex, f );
      }
    };

  } // namespace Detail




template<class M, class F>
void forEachScalarMatrixRow( const M& m, F&& f)
{
  Detail::ForEachChooser<M>::row( m, std::tuple<>(), f );
}


template<class M, class F>
void forEachScalarMatrixEntry( const M& m, F&& f)
{
  auto rowAction = [&](const auto& globalRowIndex)
  {
    Detail::ForEachChooser<M>::entry( m, globalRowIndex, globalRowIndex, std::tuple<>(), f );
  };
  Detail::ForEachChooser<M>::row( m, std::tuple<>(), rowAction );
}

} // namespace Dune

#endif
