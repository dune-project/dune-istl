// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MULTITYPEBLOCKVECTOR_HH
#define DUNE_ISTL_MULTITYPEBLOCKVECTOR_HH

#if HAVE_DUNE_BOOST
#ifdef HAVE_BOOST_FUSION

#include <cmath>
#include <iostream>

#include <dune/common/dotproduct.hh>
#include <dune/common/ftraits.hh>

#include "istlexception.hh"

#include <boost/fusion/sequence.hpp>
#include <boost/fusion/container.hpp>
#include <boost/fusion/iterator.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/fusion/algorithm.hpp>

namespace mpl=boost::mpl;
namespace fusion=boost::fusion;

// forward decl
namespace Dune {
  template<typename T1, typename T2=fusion::void_, typename T3=fusion::void_, typename T4=fusion::void_,
      typename T5=fusion::void_, typename T6=fusion::void_, typename T7=fusion::void_,
      typename T8=fusion::void_, typename T9=fusion::void_>
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
      std::cout << "\t(" << current_element << "):\n" << fusion::at_c<current_element>(v) << "\n";
      MultiTypeBlockVector_Print<current_element+1,remaining_elements-1,TVec>::print(v);   //next element
    }
  };
  template<int current_element, typename TVec>            //recursion end (remaining_elements=0)
  class MultiTypeBlockVector_Print<current_element,0,TVec> {
  public:
    static void print(const TVec& v) {std::cout << "\n";}
  };



  /**
     @brief set a MultiTypeBlockVector to some specific value

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
      fusion::at_c<count-1>(a) = b;           //equalize current elements
      MultiTypeBlockVector_Ident<count-1,T1,T2>::equalize(a,b);    //next elements
    }
  };
  template<typename T1, typename T2>                      //recursion end (count=0)
  class MultiTypeBlockVector_Ident<0,T1,T2> {public: static void equalize (T1& a, const T2& b) {} };






  /**
     @brief add/sub second vector to/from the first (v1 += v2)

     This class implements vector addition/subtraction for any MultiTypeBlockVector-Class type.
   */
  template<int count, typename T>
  class MultiTypeBlockVector_Add {
  public:

    /**
     * add vector to vector
     */
    static void add (T& a, const T& b) {    //add vector elements
      fusion::at_c<(count-1)>(a) += fusion::at_c<(count-1)>(b);
      MultiTypeBlockVector_Add<count-1,T>::add(a,b);
    }

    /**
     * sub vector from vector
     */
    static void sub (T& a, const T& b) {    //sub vector elements
      fusion::at_c<(count-1)>(a) -= fusion::at_c<(count-1)>(b);
      MultiTypeBlockVector_Add<count-1,T>::sub(a,b);
    }
  };
  template<typename T>                                    //recursion end; specialization for count=0
  class MultiTypeBlockVector_Add<0,T> {public: static void add (T& a, const T& b) {} static void sub (T& a, const T& b) {} };



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
      fusion::at_c<(count-1)>(x).axpy(a,fusion::at_c<(count-1)>(y));
      MultiTypeBlockVector_AXPY<count-1,TVec,Ta>::axpy(x,a,y);
    }
  };
  template<typename TVec, typename Ta>                    //specialization for count=0
  class MultiTypeBlockVector_AXPY<0,TVec,Ta> {public: static void axpy (TVec& x, const Ta& a, const TVec& y) {} };


  /**
     @brief Scalar * Vector Multiplication

     calculates v *= a for each element of the given vector
   */
  template<int count, typename TVec, typename Ta>
  class MultiTypeBlockVector_Mulscal {
  public:

    /**
     * calculates x *= a
     */
    static void mul(TVec& x, const Ta& a) {
      fusion::at_c<(count-1)>(x) *= a;
      MultiTypeBlockVector_Mulscal<count-1,TVec,Ta>::mul(x,a);
    }
  };
  template<typename TVec, typename Ta>                    //specialization for count=0
  class MultiTypeBlockVector_Mulscal<0,TVec,Ta> {public: static void mul (TVec& x, const Ta& a) {} };



  /**
     @brief Vector scalar multiplication

     multiplies the the current elements of x and y and recursively
     and sums it all up. Provides to variants:
     1) 'mul'  computes the indefinite inner product and
     2) 'dot'  provides an inner product by conjugating the first argument
   */
  template<int count, typename TVec>
  class MultiTypeBlockVector_Mul {
  public:
    static typename TVec::field_type mul(const TVec& x, const TVec& y) { return (fusion::at_c<count-1>(x) * fusion::at_c<count-1>(y)) + MultiTypeBlockVector_Mul<count-1,TVec>::mul(x,y); }
    static typename TVec::field_type dot(const TVec& x, const TVec& y) { return (Dune::dot(fusion::at_c<count-1>(x),fusion::at_c<count-1>(y))) + MultiTypeBlockVector_Mul<count-1,TVec>::dot(x,y); }
  };
  template<typename TVec>
  class MultiTypeBlockVector_Mul<0,TVec> {
  public:
    static typename TVec::field_type mul(const TVec& x, const TVec& y) {return 0;}
    static typename TVec::field_type dot(const TVec& x, const TVec& y) {return 0;}
  };





  /**
     @brief calulate the 2-norm out of vector elements

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
      return fusion::at_c<count-1>(a).two_norm2() + MultiTypeBlockVector_Norm<count-1,T>::result(a);
    }
  };

  template<typename T>                                    //recursion end: no more vector elements to add...
  class MultiTypeBlockVector_Norm<0,T> {
  public:
    typedef typename T::field_type field_type;
    typedef typename FieldTraits<field_type>::real_type real_type;
    static real_type result (const T& a) {return 0.0;}
  };

  /**
      @brief A Vector class to support different block types

      This vector class combines elements of different types known at compile-time.

      You must add BOOST_CPPFLAGS and BOOT_LDFLAGS to the CPPFLAGS and LDFLAGS during
      compilation, respectively, to use this class
   */
  template<typename T1, typename T2, typename T3, typename T4,
      typename T5, typename T6, typename T7, typename T8, typename T9>
  class MultiTypeBlockVector : public fusion::vector<T1, T2, T3, T4, T5, T6, T7, T8, T9> {

  public:

    /**
     * own class' type
     */
    typedef MultiTypeBlockVector<T1, T2, T3, T4, T5, T6, T7, T8, T9> type;

    typedef typename T1::field_type field_type;

    /**
     * number of elements
     */
    int count() {return mpl::size<type>::value;}

    /**
     * assignment operator
     */
    template<typename T>
    void operator= (const T& newval) {MultiTypeBlockVector_Ident<mpl::size<type>::value,type,T>::equalize(*this, newval); }

    /**
     * operator for MultiTypeBlockVector += MultiTypeBlockVector operations
     */
    void operator+= (const type& newv) {MultiTypeBlockVector_Add<mpl::size<type>::value,type>::add(*this,newv);}

    /**
     * operator for MultiTypeBlockVector -= MultiTypeBlockVector operations
     */
    void operator-= (const type& newv) {MultiTypeBlockVector_Add<mpl::size<type>::value,type>::sub(*this,newv);}

    void operator*= (const int& w) {MultiTypeBlockVector_Mulscal<mpl::size<type>::value,type,const int>::mul(*this,w);}
    void operator*= (const float& w) {MultiTypeBlockVector_Mulscal<mpl::size<type>::value,type,const float>::mul(*this,w);}
    void operator*= (const double& w) {MultiTypeBlockVector_Mulscal<mpl::size<type>::value,type,const double>::mul(*this,w);}

    field_type operator* (const type& newv) const {return MultiTypeBlockVector_Mul<mpl::size<type>::value,type>::mul(*this,newv);}
    field_type dot (const type& newv) const {return MultiTypeBlockVector_Mul<mpl::size<type>::value,type>::dot(*this,newv);}

    /**
     * two-norm^2
     */
    typename FieldTraits<field_type>::real_type two_norm2() const {return MultiTypeBlockVector_Norm<mpl::size<type>::value,type>::result(*this);}

    /**
     * the real two-norm
     */
    typename FieldTraits<field_type>::real_type two_norm() const {return sqrt(this->two_norm2());}

    /**
     * axpy operation on this vector (*this += a * y)
     */
    template<typename Ta>
    void axpy (const Ta& a, const type& y) {
      MultiTypeBlockVector_AXPY<mpl::size<type>::value,type,Ta>::axpy(*this,a,y);
    }

  };



  /**
     @brief << operator for a MultiTypeBlockVector

     operator<< for printing out a MultiTypeBlockVector
   */
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
  std::ostream& operator<< (std::ostream& s, const MultiTypeBlockVector<T1,T2,T3,T4,T5,T6,T7,T8,T9>& v) {
    MultiTypeBlockVector_Print<0,mpl::size<MultiTypeBlockVector<T1,T2,T3,T4,T5,T6,T7,T8,T9> >::value,MultiTypeBlockVector<T1,T2,T3,T4,T5,T6,T7,T8,T9> >::print(v);
    return s;
  }



} // end namespace

#endif // end HAVE_BOOST_FUSION
#endif // end HAVE_DUNE_BOOST

#endif
