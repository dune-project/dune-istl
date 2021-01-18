// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_VECTORITERATOR_HH
#define DUNE_ISTL_VECTORITERATOR_HH

#include<tuple>

#include<dune/common/fvector.hh>
#include<dune/common/hybridutilities.hh>
#include<dune/common/indices.hh>
#include<dune/common/typetraits.hh>

#include<dune/istl/bvector.hh>
#include<dune/istl/multitypeblockvector.hh>

namespace Dune
{

/** \brief Primary class for the iteration of arbitrary vector types
 *
 * \tparam Vector Vector type
 */
template<class Vector>
class VectorIterator
{
public:

  // to be sure
  static_assert( IsNumber<Vector>::value, " VectorIterator has no specialization for your vector type ");

  VectorIterator(const Vector& vector)
  : vector_(vector)
  {};

  //! in the scalar case call the entryAction
  template<class EntryAction, class Index = std::tuple<> >
  void iterate(EntryAction entryAction, const Index& index) const
  {
      // scalar case: call the action
      entryAction( index, vector_ );
  }

private:
  const Vector& vector_;
};


/** \brief Specialization of VectorIterator for FieldVector that iterates through the vector statically */
template<class T, int n>
class VectorIterator<FieldVector<T,n>>
{
public:

  using Vector = FieldVector<T,n>;

  VectorIterator(const Vector& vector)
  : vector_(vector)
  {};

  //! loop over the entries and append the current index
  template<class EntryAction, class Index>
  void iterate(EntryAction entryAction, const Index& index) const
  {
    // we have to append a new index
    for ( size_t i = 0; i < n; i++ )
    {
      auto newIndex = std::tuple_cat( index, std::make_tuple( i ) );
      auto&& vector = vector_[i];
      VectorIterator<T> mi(vector);
      mi.iterate( entryAction, newIndex );
    }
  }

private:
  const Vector& vector_;
};


/** \brief Specialization of VectorIterator for BlockVector that iterates through the vector dynamically */
template<class B, class A>
class VectorIterator<BlockVector<B,A>>
{
public:

  using Vector = BlockVector<B,A>;

  VectorIterator(const Vector& vector)
  : vector_(vector)
  {};

  template<class EntryAction, class Index>
  void iterate(EntryAction entryAction, const Index& index) const
  {
    // we have to append a new index
    auto n = vector_.size();
    for ( size_t i = 0; i < n; i++ )
    {
      auto newIndex = std::tuple_cat( index, std::make_tuple( i ) );
      auto&& vector = vector_[i];
      VectorIterator<B> mi(vector);
      mi.iterate( entryAction, newIndex );
    }
  }

private:
  const Vector& vector_;
};


/** \brief Specialization of VectorIterator for MultiTypeBlockVector that iterates through the vector statically */
template<class FirstBlock, class... Args>
class VectorIterator<MultiTypeBlockVector<FirstBlock, Args...>>
{
public:

  using Vector = MultiTypeBlockVector<FirstBlock, Args...>;

  VectorIterator(const Vector& vector)
  : vector_(vector)
  {};

  template<class EntryAction, class Index>
  void iterate(const EntryAction& entryAction, const Index& index) const
  {
    // we have to append a new index
    auto N = index_constant<Vector::N()>();
    using namespace Dune::Hybrid;
    forEach(integralRange(N), [&](auto&& i)
    {
      auto newIndex = std::tuple_cat( index, std::make_tuple( i ) );
      auto&& vector = vector_[i];
      using B = std::decay_t<decltype(vector)>;
      VectorIterator<B> mi(vector);
      mi.iterate( entryAction, newIndex );
    });
  }

private:
  const Vector& vector_;
};


} // namespace Dune

#endif
