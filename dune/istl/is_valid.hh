#ifndef DUNE_ISTL_IS_VALID_HH
#define DUNE_ISTL_IS_VALID_HH

#include <type_traits>

namespace Dune
{
  /// @cond
  template <class,class> class LinearOperator;
  template <class,class> class Preconditioner;
  template <class> class ScalarProduct;
  /// @endcond

  /** @addtogroup ISTL_Solvers
   *  @{
   */

  /// Is std::true_type if `Operator` is derived from `Dune::LinearOperator<Domain,Range>`, else std::false_type.
  template <class Operator, class Domain, class Range = Domain>
  using IsValidOperator = std::is_base_of< Dune::LinearOperator<Domain,Range>, typename std::decay<Operator>::type >;

  /// Is std::true_type if `Preconditioner` is derived from `Dune::Preconditioner<Domain,Range>`, else std::false_type.
  template <class Preconditioner, class Domain, class Range = Domain>
  using IsValidPreconditioner = std::is_base_of< Dune::Preconditioner<Domain,Range>, typename std::decay<Preconditioner>::type >;

  /// Is std::true_type if `ScalarProduct` is derived from `Dune::ScalarProduct<Domain>`, else std::false_type.
  template <class ScalarProduct, class Domain>
  using IsValidScalarProduct = std::is_base_of< Dune::ScalarProduct<Domain>, typename std::decay<ScalarProduct>::type >;
  /** @} */
}

#endif // DUNE_ISTL_IS_VALID_HH
