// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_VECTORFLATVIEW_HH
#define DUNE_ISTL_VECTORFLATVIEW_HH

#include<tuple>

#include<dune/common/fvector.hh>
#include<dune/common/hybridutilities.hh>
#include<dune/common/indices.hh>
#include<dune/common/typetraits.hh>

#include<dune/istl/blocklevel.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/multitypeblockvector.hh>

namespace Dune
{


template<class Vector>
class VectorFlatView
{
public:

  /** \brief Type used for vector sizes */
  using size_type = std::size_t;

  /** \brief The type used for scalars */
  using field_type = Vector::field_type;

  /** \brief Default constructor referencing the original vector */
  VectorFlatView(const Vector& vector)
  : vector_(vector)
  {}

  /** \brief Return the number of non-zero vector entries
    *
    * Since we wrap a possibly blocked vector we redirect it to the original dim()
    */
  size_type size() const
  {
    return vector_.dim();
  }

  /** \brief Number of elements */
  size_type N() const
  {
    return size();
  }

  /** \brief Number of scalar elements
   *
   *  This is the same as size() for a flat vector
   */
  size_type dim() const
  {
    return size();
  }


private:
  Vector& vector_;

};

}

#endif
