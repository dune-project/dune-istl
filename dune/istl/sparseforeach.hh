// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_SPARSEFOREACH_HH
#define DUNE_ISTL_SPARSEFOREACH_HH

#include<type_traits>

#include<dune/common/fvector.hh>
#include<dune/common/hybridutilities.hh>
#include<dune/common/indices.hh>
#include<dune/common/typetraits.hh>
#include <dune/common/std/type_traits.hh>

#include<dune/istl/blocklevel.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/flatvectorview.hh>
#include<dune/istl/multitypeblockvector.hh>

namespace Dune {

  namespace Detail {

    // stolen from dune-functions
    template<class C>
    using staticIndexAccess_t = decltype(std::declval<C>()[Dune::Indices::_0]);

    template<class C>
    using isScalar = Dune::Std::bool_constant<not Dune::Std::is_detected_v<staticIndexAccess_t, std::remove_reference_t<C>>>;

    /**
    * \brief Flat index forEach loop over a container
    *
    * \tparam V Type of given container
    * \tparam F Type of given predicate
    *
    * \param v The container to loop over
    * \param f A predicate that will be called with the flat index and entry at each entry
    *
    * This is currently supported for all containers that are
    * supported by Hybrid::forEach.
    */

    template<class V, class F>
    void flatSparseForEach(V&& v, F&& f, std::size_t& index)
    {
      Hybrid::forEach(v, [&](auto&& vi) {
        if constexpr( isScalar<std::decay_t<decltype(vi)>>::value )
          f(index++, vi);
        else
          flatSparseForEach(vi, f, index);
      });
    }
  } // namespace Detail



template<class V, class F>
void sparseForEach(FlatVectorView<V>& fvv, F&& f) {
  std::size_t index = 0;
  Detail::flatSparseForEach(fvv.rawVector(), f, index);
}

} // namespace Dune

#endif
