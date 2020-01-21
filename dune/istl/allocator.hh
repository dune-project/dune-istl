#ifndef DUNE_ISTL_ALLOCATOR_HH
#define DUNE_ISTL_ALLOCATOR_HH

#include <dune/common/typetraits.hh>
#include <memory>

namespace Dune {

    template<typename T>
    struct exists{
        static const bool value = true;
    };

    template<typename T, typename = void>
    struct DefaultAllocatorTraits
    {
        using type = std::allocator<T>;
    };

    template<typename T>
    struct DefaultAllocatorTraits<T, void_t<typename T::allocator_type> >
    {
        using type = typename T::allocator_type;
    };

    template<typename T>
    struct AllocatorTraits : public DefaultAllocatorTraits<T> {};

    template<typename T>
    using AllocatorType = typename AllocatorTraits<T>::type;

    template<typename T, typename X>
    using ReboundAllocatorType = typename std::allocator_traits<typename AllocatorTraits<T>::type>::template rebind_alloc<X>;

} // end namespace Dune

#endif // DUNE_ISTL_ALLOCATOR_HH
