// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKKRYLOV_BLOCKINNERPRODUCT_HH
#define DUNE_ISTL_BLOCKKRYLOV_BLOCKINNERPRODUCT_HH

/** \file
 * \brief Provides a BlockInnerProduct to be used in block Krylov methods
 */

#include <dune/common/parallel/future.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/blockkrylov/matrixalgebra.hh>

namespace Dune{

  /**
     \brief Extends the `ScalarProduct` by functions for computing block
     inner products needed for block Krylov methods based on a given *-subalgebra
     of \f$K^{s\times s}\f$ where \f$K\f$ is the field type of
     the operator and \f$s\f$ is the number of right-hand sides.

     \tparam X vector type
  */
  template<class Algebra>
  class BlockInnerProduct
    : virtual public ScalarProduct<typename Algebra::vector_type> {
    typedef typename Algebra::vector_type X;

  public:
    BlockInnerProduct(size_t cholQR_it = 1)
      : _cholQR_it(cholQR_it)
    {}

    /**
       Block inner product. Returns the evaluation of the block inner product.
       The *-subalgebra in that the evaluation is performed is specified in
       the implementation classes.

       \param[in] x
       \param[in] y

       \returns \f$\langle x, y \rangle_S\f$
    */
    virtual Future<Algebra> bdot(const X&, const X&) = 0;

    /** Normalization of a block vector, i.e. the orthonormalization of the
        columns. The CholeskyQR algorithm is used for that. It can be configured
        whether it should be repeated to improve the stability (default=1). If
        at least 3 iterations are performed a shift is introduced. This approach
        was presented in the paper Fukaya et al. 2018, Shifted CholeskyQR for
        computing the QR factorization of ill-conditioned matrices.

        \param[inout] x block vector to be normalized
        \param[inout] y Mx
        if the orthonormalization should be carried out w.r.t. the M
        block inner product, x otherwise.

        \returns The R factor of the QR-decomposition
    */
    Future<Algebra> bnormalize(X& x, X& y) {
      // TODO make it really async with a PostProcessFuture
      Algebra alpha;
      typedef typename X::field_type field_type;
      typedef Simd::Scalar<field_type> scalar_field_type;
      for(size_t i=_cholQR_it; i>0  ;--i){
        Algebra alpha2 = this->bdot(x,y).get();
        // shift
        if(i > 2){
          constexpr size_t s = Simd::lanes<field_type>();
          constexpr scalar_field_type eps = std::numeric_limits<scalar_field_type>::epsilon();
          Algebra shift = alpha2;
          shift = Algebra::identity(11.0 * (x.size() * s + s*(s+1)) * eps * std::sqrt(alpha2.frobenius_norm()));
          alpha2.add(shift);
        }
        auto dependent = alpha2.cholesky();
        alpha2.transpose();
        auto beta = alpha2;
        beta.eliminate(dependent);
        beta.invert();
        beta.mv(x);
        if(&x != &y){
          beta.mv(y);
        }
        if(i==_cholQR_it)
          alpha = alpha2;
        else
          alpha.leftmultiply(alpha2);
      }
      return PseudoFuture<Algebra>(alpha);
    }

  private:
    size_t _cholQR_it = 1;
  };

  /**
     \brief Sequential implementation of BlockInnerProduct.

     \tparam X vector type
     \tparam Algebra *-subalgebra implementation (see `MatrixAlgebra`)
  */
  template<class Algebra>
  class SeqBlockInnerProduct
    : public BlockInnerProduct<Algebra>,
      public SeqScalarProduct<typename Algebra::vector_type>{
    using X = typename Algebra::vector_type;
  public:
    SeqBlockInnerProduct(const ParameterTree& config = {})
      : BlockInnerProduct<Algebra>(config.get("cholQR_it", 1))
    {}

    Future<Algebra> bdot(const X& x, const X& y) override {
      return PseudoFuture<Algebra>(Algebra::dot(x,y));
    }
  };

  /**
     \brief Parallel implementation of BlockInnerProduct using a given Communicator.

     \tparam X vector type
     \tparam Comm communicator type
     \tparam Algebra *-subalgebra implementation (see `MatrixAlgebra`)
  */
  template<class Comm, class Algebra>
  class ParallelBlockInnerProduct
    : public ParallelScalarProduct<typename Algebra::vector_type, Comm>
    , public BlockInnerProduct<Algebra>{
    using X = typename Algebra::vector_type;
    using Base = ParallelScalarProduct<X, Comm>;
    using field_type = typename X::field_type;
  public:

    ParallelBlockInnerProduct (const Comm& com,
                               SolverCategory::Category cat,
                               const ParameterTree& config = {})
      : ParallelScalarProduct<X,Comm>(com,cat)
      , BlockInnerProduct<Algebra>(config.get("cholQR_it", 1))
    {}

    using Base::category;

    Future<Algebra> bdot(const X& x, const X& y) override {
      Algebra local_dot = Algebra::dot(x,y);
      return _communication->communicator().template iallreduce<std::plus<Algebra>>(std::move(local_dot));
    }

  protected:
    using Base::_communication;
  };

  /**
     \brief Factory function for BlockInnerProducts configured by a ParameterTree.
  */
  template<class X>
  std::shared_ptr<ScalarProduct<X>> createBlockInnerProduct(size_t p, const ParameterTree& config = {}){
    constexpr size_t K = Simd::lanes<typename X::field_type>();
    std::shared_ptr<ScalarProduct<X>> bip;
    Hybrid::forEach(std::make_index_sequence<K>(),
                    [&](auto pp){
                      if constexpr (K%(pp+1) == 0){
                        if (p==(pp+1)){
                          bip = std::make_shared<SeqBlockInnerProduct<ParallelMatrixAlgebra<X,(pp+1)>>>(config);
                        }
                      }
                    });
    return bip;
  }

  /**
     \brief Factory function for BlockInnerProducts configured by a ParameterTree for the parallel implementation.
  */
  template<class X, class Comm>
  std::shared_ptr<ScalarProduct<X>> createBlockInnerProduct(size_t p, const ParameterTree& config,
                                                            Comm& comm, SolverCategory::Category cat){
    constexpr size_t K = Simd::lanes<typename X::field_type>();
    std::shared_ptr<ScalarProduct<X>> bip;
    Hybrid::forEach(std::make_index_sequence<K>(),
                    [&](auto pp){
                      if constexpr (K%(pp+1) == 0){
                        if (p==(pp+1)){
                          bip = std::make_shared<ParallelBlockInnerProduct<Comm, ParallelMatrixAlgebra<X,(pp+1)>>>(comm, cat, config);
                        }
                      }
                    });
    return bip;
  }

}

#endif
