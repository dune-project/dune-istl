// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_FORWARDDECLARATIONS_HH
#define DUNE_ISTL_FORWARDDECLARATIONS_HH

#include <dune/common/memory/domain.hh>

namespace Dune {

  template<typename F, int n>
  class FieldVector;

  template<typename F, int n, int m>
  class FieldMatrix;

  template<typename F>
  class DynamicVector;

  template<typename F>
  class DynamicMatrix;

  template<typename Block, typename Alloc>
  class BlockVector;

  template<typename Block, typename Alloc>
  class BCRSMatrix;

  namespace ISTL {

    template<typename F_, typename A_, typename D_ = typename Memory::allocator_domain<A_>::type>
    class Vector;

    template<typename F_, typename A_, typename D_ = typename Memory::allocator_domain<A_>::type>
    class ELLMatrix;


    template<typename F_, typename A_, typename D_ = typename Memory::allocator_domain<A_>::type>
    class BlockVector;

    template<typename F_, typename A_, typename D_ = typename Memory::allocator_domain<A_>::type>
    class BELLMatrix;

  } // namespace ISTL
} // namespace Dune

#endif // DUNE_ISTL_FORWARDDECLARATIONS_HH
