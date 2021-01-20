// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FLATVECTORVIEW_HH
#define DUNE_ISTL_FLATVECTORVIEW_HH

#include <cstddef>


namespace Dune
{



/** \brief Wrapper for blocked vector types to export a flat vector interface
 *
 * \tparam Vector The original vector type that is wrapped
 */
template<class Vector>
class FlatVectorView
{
public:

  /** \brief Type used for vector sizes */
  using size_type = std::size_t;

  /** \brief The type used for scalars */
  using field_type = typename Vector::field_type;

  /** \brief Default constructor referencing the original vector */
  FlatVectorView(const Vector& vector)
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
   *  For a flat vector size() and dim() are the same
   */
  size_type dim() const
  {
    return size();
  }

  /** \brief Return reference to the stored original vector */
  const Vector& rawVector() const
  {
    return vector_;
  }


private:
  const Vector& vector_;

};

}

#endif
