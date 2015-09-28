// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MULTITYPEBLOCKVECTOR_HH
#define DUNE_ISTL_MULTITYPEBLOCKVECTOR_HH

#include <cmath>
#include <iostream>
#include <tuple>

#include <dune/common/dotproduct.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/std/constexpr.hh>

#include "istlexception.hh"

// forward declaration
namespace Dune {
  template < typename... Args >
  class MultiTypeBlockVector;
}

#include "gsetc.hh"

namespace Dune {

  /**
      @addtogroup ISTL_SPMV
      @{
   */




  /**
     @brief prints out a vector (type MultiTypeBlockVector)

     The parameter "current_element" is the index of
     the element to be printed out. Via recursive calling
     all other elements (number is "remaining_elements")
     are printed, too. This is internally called when
     a MultiTypeBlockVector object is passed to an output stream.
     Example:
     \code
     MultiTypeBlockVector<int,int> CVect;
     std::cout << CVect;
     \endcode
   */

  template<int current_element, int remaining_elements, typename TVec>
  class MultiTypeBlockVector_Print {
  public:
    /**
     * print out the current vector element and all following
     */
    static void print(const TVec& v) {
      std::cout << "\t(" << current_element << "):\n" << std::get<current_element>(v) << "\n";
      MultiTypeBlockVector_Print<current_element+1,remaining_elements-1,TVec>::print(v);   //next element
    }
  };
  template<int current_element, typename TVec>            //recursion end (remaining_elements=0)
  class MultiTypeBlockVector_Print<current_element,0,TVec> {
  public:
    static void print(const TVec&) {std::cout << "\n";}
  };



  /**
     @brief Set a MultiTypeBlockVector to some specific value

     This class is used by the MultiTypeBlockVector class' internal = operator.
     Whenever a vector is assigned to a value, each vector element
     has to provide the = operator for the right side. Example:
     \code
     MultiTypeBlockVector<int,int,int> CVect;
     CVect = 3;                  //sets all integer elements to 3
     \endcode
   */
  template<int count, typename T1, typename T2>
  class MultiTypeBlockVector_Ident {
  public:

    /**
     * equalize two vectors' element (index is template parameter count)
     * note: each MultiTypeBlockVector element has to provide the = operator with type T2
     */
    static void equalize(T1& a, const T2& b) {
      std::get<count-1>(a) = b;           //equalize current elements
      MultiTypeBlockVector_Ident<count-1,T1,T2>::equalize(a,b);    //next elements
    }
  };
  template<typename T1, typename T2>                      //recursion end (count=0)
  class MultiTypeBlockVector_Ident<0,T1,T2> {public: static void equalize (T1&, const T2&) {} };






  /**
     @brief Add/subtract second vector to/from the first (v1 += v2)

     This class implements vector addition/subtraction for any MultiTypeBlockVector-Class type.
   */
  template<int count, typename T>
  class MultiTypeBlockVector_Add {
  public:

    /**
     * add vector to vector
     */
    static void add (T& a, const T& b) {    //add vector elements
      std::get<(count-1)>(a) += std::get<(count-1)>(b);
      MultiTypeBlockVector_Add<count-1,T>::add(a,b);
    }

    /**
     * Subtract vector from vector
     */
    static void sub (T& a, const T& b) {    //sub vector elements
      std::get<(count-1)>(a) -= std::get<(count-1)>(b);
      MultiTypeBlockVector_Add<count-1,T>::sub(a,b);
    }
  };
  template<typename T>                                    //recursion end; specialization for count=0
  class MultiTypeBlockVector_Add<0,T> {public: static void add (T&, const T&) {} static void sub (T&, const T&) {} };



  /**
     @brief AXPY operation on vectors

     calculates x += a * y
   */
  template<int count, typename TVec, typename Ta>
  class MultiTypeBlockVector_AXPY {
  public:

    /**
     * calculates x += a * y
     */
    static void axpy(TVec& x, const Ta& a, const TVec& y) {
      std::get<(count-1)>(x).axpy(a,std::get<(count-1)>(y));
      MultiTypeBlockVector_AXPY<count-1,TVec,Ta>::axpy(x,a,y);
    }
  };
  template<typename TVec, typename Ta>                    //specialization for count=0
  class MultiTypeBlockVector_AXPY<0,TVec,Ta> {public: static void axpy (TVec&, const Ta&, const TVec&) {} };


  /** @brief In-place multiplication with a scalar
   *
   * Calculates v *= a for each element of the given vector.
   */
  template<int count, typename TVec, typename Ta>
  class MultiTypeBlockVector_Mulscal {
  public:

    /**
     * calculates x *= a
     */
    static void mul(TVec& x, const Ta& a) {
      std::get<(count-1)>(x) *= a;
      MultiTypeBlockVector_Mulscal<count-1,TVec,Ta>::mul(x,a);
    }
  };
  template<typename TVec, typename Ta>                    //specialization for count=0
  class MultiTypeBlockVector_Mulscal<0,TVec,Ta> {public: static void mul (TVec&, const Ta&) {} };



  /** @brief Scalar products
   *
   * multiplies the current elements of x and y pairwise, and sum up the results.
   * Provides two variants:
   * 1) 'mul'  computes the indefinite inner product and
   * 2) 'dot'  provides an inner product by conjugating the first argument
   */
  template<int count, typename TVec>
  class MultiTypeBlockVector_Mul {
  public:
    static typename TVec::field_type mul(const TVec& x, const TVec& y)
    {
      return (std::get<count-1>(x) * std::get<count-1>(y)) + MultiTypeBlockVector_Mul<count-1,TVec>::mul(x,y);
    }

    static typename TVec::field_type dot(const TVec& x, const TVec& y)
    {
      return Dune::dot(std::get<count-1>(x),std::get<count-1>(y)) + MultiTypeBlockVector_Mul<count-1,TVec>::dot(x,y);
    }
  };

  template<typename TVec>
  class MultiTypeBlockVector_Mul<0,TVec> {
  public:
    static typename TVec::field_type mul(const TVec&, const TVec&) {return 0;}
    static typename TVec::field_type dot(const TVec&, const TVec&) {return 0;}
  };





  /** \brief Calculate the 2-norm

     Each element of the vector has to provide the method "two_norm2()"
     in order to calculate the whole vector's 2-norm.
   */
  template<int count, typename T>
  class MultiTypeBlockVector_Norm {
  public:
    typedef typename T::field_type field_type;
    typedef typename FieldTraits<field_type>::real_type real_type;

    /**
     * sum up all elements' 2-norms
     */
    static real_type result (const T& a) {             //result = sum of all elements' 2-norms
      return std::get<count-1>(a).two_norm2() + MultiTypeBlockVector_Norm<count-1,T>::result(a);
    }
  };

  template<typename T>                                    //recursion end: no more vector elements to add...
  class MultiTypeBlockVector_Norm<0,T> {
  public:
    typedef typename T::field_type field_type;
    typedef typename FieldTraits<field_type>::real_type real_type;
    static real_type result (const T&) {return 0.0;}
  };

  /**
      @brief A Vector class to support different block types

      This vector class combines elements of different types known at compile-time.
   */
  template < typename... Args >
  class MultiTypeBlockVector
  : public std::tuple<Args...>
  {
    /** \brief Helper type */
    typedef std::tuple<Args...> tupleType;
  public:

    /**
     * own class' type
     */
    typedef MultiTypeBlockVector<Args...> type;

    /** \brief The type used for scalars
     *
     * The current code hardwires it to 'double', which is far from nice.
     * On the other hand, it is not clear what the correct type is.  If the MultiTypeBlockVector class
     * is instantiated with several vectors of different field_types, what should the resulting
     * field_type be?
     */
    typedef double field_type;

    /** \brief Return the number of vector entries */
    static DUNE_CONSTEXPR std::size_t size()
    {
      return sizeof...(Args);
    }

    /**
     * number of elements
     */
    int count()
    {
      return sizeof...(Args);
    }

    /** \brief Random-access operator
     *
     * This method mimicks the behavior of normal vector access with square brackets like, e.g., v[5] = 1.
     * The problem is that the return type is different for each value of the argument in the brackets.
     * Therefore we implement a trick using std::integral_constant.  To access the first entry of
     * a MultiTypeBlockVector named v write
     * \code
     *  MultiTypeBlockVector<A,B,C,D> v;
     *  std::integral_constant<int,0> _0;
     *  v[_0] = ...
     * \endcode
     * The name '_0' used here as a static replacement of the integer number zero is arbitrary.
     * Any other variable name can be used.  If you don't like the separate variable, you can writee
     * \code
     *  MultiTypeBlockVector<A,B,C,D> v;
     *  v[std::integral_constant<int,0>()] = ...
     * \endcode
     */
    template< int index >
    typename std::tuple_element<index,tupleType>::type&
    operator[] ( const std::integral_constant< int, index > indexVariable )
    {
      DUNE_UNUSED_PARAMETER(indexVariable);
      return std::get<index>(*this);
    }

    /** \brief Random-access operator
     *
     * This method mimicks the behavior of normal vector access with square brackets like, e.g., v[5] = 1.
     * The problem is that the return type is different for each value of the argument in the brackets.
     * Therefore we implement a trick using std::integral_constant.  To access the first entry of
     * a MultiTypeBlockVector named v write
     * \code
     *  MultiTypeBlockVector<A,B,C,D> v;
     *  std::integral_constant<std::size_t,0> _0;
     *  v[_0] = ...
     * \endcode
     * The name '_0' used here as a static replacement of the integer number zero is arbitrary.
     * Any other variable name can be used.  If you don't like the separate variable, you can writee
     * \code
     *  MultiTypeBlockVector<A,B,C,D> v;
     *  v[std::integral_constant<std::size_t,0>()] = ...
     * \endcode
     */
    template< std::size_t index >
    typename std::tuple_element<index,tupleType>::type&
    operator[] ( const std::integral_constant< std::size_t, index > indexVariable )
    {
      DUNE_UNUSED_PARAMETER(indexVariable);
      return std::get<index>(*this);
    }

    /** \brief Const random-access operator
     *
     * This is the const version of the random-access operator.  See the non-const version for a full
     * explanation of how to use it.
     */
    template< std::size_t index >
    const typename std::tuple_element<index,tupleType>::type&
    operator[] ( const std::integral_constant< std::size_t, index > indexVariable ) const
    {
      DUNE_UNUSED_PARAMETER(indexVariable);
      return std::get<index>(*this);
    }

    /** \brief Const random-access operator
     *
     * This is the const version of the random-access operator.  See the non-const version for a full
     * explanation of how to use it.
     */
    template< int index >
    const typename std::tuple_element<index,tupleType>::type&
    operator[] ( const std::integral_constant< int, index > indexVariable ) const
    {
      DUNE_UNUSED_PARAMETER(indexVariable);
      return std::get<index>(*this);
    }

    /** \brief Assignment operator
     */
    template<typename T>
    void operator= (const T& newval) {MultiTypeBlockVector_Ident<sizeof...(Args),type,T>::equalize(*this, newval); }

    /**
     * operator for MultiTypeBlockVector += MultiTypeBlockVector operations
     */
    void operator+= (const type& newv) {MultiTypeBlockVector_Add<sizeof...(Args),type>::add(*this,newv);}

    /**
     * operator for MultiTypeBlockVector -= MultiTypeBlockVector operations
     */
    void operator-= (const type& newv) {MultiTypeBlockVector_Add<sizeof...(Args),type>::sub(*this,newv);}

    void operator*= (const int& w) {MultiTypeBlockVector_Mulscal<sizeof...(Args),type,const int>::mul(*this,w);}
    void operator*= (const float& w) {MultiTypeBlockVector_Mulscal<sizeof...(Args),type,const float>::mul(*this,w);}
    void operator*= (const double& w) {MultiTypeBlockVector_Mulscal<sizeof...(Args),type,const double>::mul(*this,w);}

    field_type operator* (const type& newv) const {return MultiTypeBlockVector_Mul<sizeof...(Args),type>::mul(*this,newv);}
    field_type dot (const type& newv) const {return MultiTypeBlockVector_Mul<sizeof...(Args),type>::dot(*this,newv);}

    /** \brief Compute the squared Euclidean norm
     */
    typename FieldTraits<field_type>::real_type two_norm2() const {return MultiTypeBlockVector_Norm<sizeof...(Args),type>::result(*this);}

    /** \brief Compute the Euclidean norm
     */
    typename FieldTraits<field_type>::real_type two_norm() const {return sqrt(this->two_norm2());}

    /** \brief Axpy operation on this vector (*this += a * y)
     *
     * \tparam Ta Type of the scalar 'a'
     */
    template<typename Ta>
    void axpy (const Ta& a, const type& y) {
      MultiTypeBlockVector_AXPY<sizeof...(Args),type,Ta>::axpy(*this,a,y);
    }

  };



  /** \brief Send MultiTypeBlockVector to an outstream
   */
  template <typename... Args>
  std::ostream& operator<< (std::ostream& s, const MultiTypeBlockVector<Args...>& v) {
    MultiTypeBlockVector_Print<0,sizeof...(Args),MultiTypeBlockVector<Args...> >::print(v);
    return s;
  }



} // end namespace

#endif
