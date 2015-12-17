// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MULTITYPEBLOCKMATRIX_HH
#define DUNE_ISTL_MULTITYPEBLOCKMATRIX_HH

#include <cmath>
#include <iostream>
#include <tuple>

#include "istlexception.hh"

// forward declaration
namespace Dune
{
  template<typename FirstRow, typename... Args>
  class MultiTypeBlockMatrix;

  template<int I, int crow, int remain_row>
  class MultiTypeBlockMatrix_Solver;
}

#include "gsetc.hh"

namespace Dune {

  /**
      @addtogroup ISTL_SPMV
      @{
   */




  /**
     @brief prints out a matrix (type MultiTypeBlockMatrix)

     The parameters "crow" and "ccol" are the indices of
     the element to be printed out. Via recursive calling
     all other elements are printed, too. This is internally
     called when a MultiTypeBlockMatrix object is passed to an output stream.
   */

  template<int crow, int remain_rows, int ccol, int remain_cols,
      typename TMatrix>
  class MultiTypeBlockMatrix_Print {
  public:

    /**
     * print out a matrix block and all following
     */
    static void print(const TMatrix& m) {
      std::cout << "\t(" << crow << ", " << ccol << "): \n" << std::get<ccol>( std::get<crow>(m));
      MultiTypeBlockMatrix_Print<crow,remain_rows,ccol+1,remain_cols-1,TMatrix>::print(m);         //next column
    }
  };
  template<int crow, int remain_rows, int ccol, typename TMatrix> //specialization for remain_cols=0
  class MultiTypeBlockMatrix_Print<crow,remain_rows,ccol,0,TMatrix> {
  public: static void print(const TMatrix& m) {
      MultiTypeBlockMatrix_Print<crow+1,remain_rows-1,0,TMatrix::N(),TMatrix>::print(m);                 //next row
    }
  };

  template<int crow, int ccol, int remain_cols, typename TMatrix> //recursion end: specialization for remain_rows=0
  class MultiTypeBlockMatrix_Print<crow,0,ccol,remain_cols,TMatrix> {
  public:
    static void print(const TMatrix& m)
    {std::cout << std::endl;}
  };



  //make MultiTypeBlockVector_Ident known (for MultiTypeBlockMatrix_Ident)
  template<int count, typename T1, typename T2>
  class MultiTypeBlockVector_Ident;


  /**
     @brief Set a MultiTypeBlockMatrix to some specific scalar value

     This class is used by the MultiTypeBlockMatrix class' internal assignment operator.
     Whenever a vector is assigned a scalar value, each block element
     has to provide operator= for the right side. Example:
     \code
     typedef MultiTypeBlockVector<int,int,int> CVect;
     MultiTypeBlockMatrix<CVect,CVect> CMat;
     CMat = 3;                   //sets all 3x2 integer elements to 3
     \endcode
   */
  template<int rowcount, typename T1, typename T2>
  class MultiTypeBlockMatrix_Ident {
  public:

    /**
     * equalize two matrix' element
     * note: uses MultiTypeBlockVector_Ident to equalize each row (which is of MultiTypeBlockVector type)
     */
    static void equalize(T1& a, const T2& b) {
      MultiTypeBlockVector_Ident<T1::M(),T1,T2>::equalize(a,b);              //rows are cvectors
      MultiTypeBlockMatrix_Ident<rowcount-1,T1,T2>::equalize(a,b);         //iterate over rows
    }
  };

  //recursion end for rowcount=0
  template<typename T1, typename T2>
  class MultiTypeBlockMatrix_Ident<0,T1,T2> {
  public:
    static void equalize (T1&, const T2&)
    {}
  };

  /**
     @brief Matrix-vector multiplication

     This class implements matrix vector multiplication for MultiTypeBlockMatrix/MultiTypeBlockVector types
   */
  template<int crow, int remain_rows, int ccol, int remain_cols,
      typename TVecY, typename TMatrix, typename TVecX>
  class MultiTypeBlockMatrix_VectMul {
  public:

    /** \brief y += A x
     */
    static void umv(TVecY& y, const TMatrix& A, const TVecX& x) {
      std::get<ccol>( std::get<crow>(A) ).umv( std::get<ccol>(x), std::get<crow>(y) );
      MultiTypeBlockMatrix_VectMul<crow,remain_rows,ccol+1,remain_cols-1,TVecY,TMatrix,TVecX>::umv(y, A, x);
    }

    /** \brief y -= A x
     */
    static void mmv(TVecY& y, const TMatrix& A, const TVecX& x) {
      std::get<ccol>( std::get<crow>(A) ).mmv( std::get<ccol>(x), std::get<crow>(y) );
      MultiTypeBlockMatrix_VectMul<crow,remain_rows,ccol+1,remain_cols-1,TVecY,TMatrix,TVecX>::mmv(y, A, x);
    }

    /** \brief y += alpha A x
     * \tparam AlphaType Type used for the scalar factor 'alpha'
     */
    template<typename AlphaType>
    static void usmv(const AlphaType& alpha, TVecY& y, const TMatrix& A, const TVecX& x) {
      std::get<ccol>( std::get<crow>(A) ).usmv(alpha, std::get<ccol>(x), std::get<crow>(y) );
      MultiTypeBlockMatrix_VectMul<crow,remain_rows,ccol+1,remain_cols-1,TVecY,TMatrix,TVecX>::usmv(alpha,y, A, x);
    }


  };

  //specialization for remain_cols = 0
  template<int crow, int remain_rows,int ccol, typename TVecY,
      typename TMatrix, typename TVecX>
  class MultiTypeBlockMatrix_VectMul<crow,remain_rows,ccol,0,TVecY,TMatrix,TVecX> {                                    //start iteration over next row

  public:
    /**
     * do y += A x in next row
     */
    static void umv(TVecY& y, const TMatrix& A, const TVecX& x) {
      MultiTypeBlockMatrix_VectMul<crow+1,remain_rows-1,0,TMatrix::M(),TVecY,TMatrix,TVecX>::umv(y, A, x);
    }

    /**
     * do y -= A x in next row
     */
    static void mmv(TVecY& y, const TMatrix& A, const TVecX& x) {
      MultiTypeBlockMatrix_VectMul<crow+1,remain_rows-1,0,TMatrix::M(),TVecY,TMatrix,TVecX>::mmv(y, A, x);
    }

    template <typename AlphaType>
    static void usmv(const AlphaType& alpha, TVecY& y, const TMatrix& A, const TVecX& x) {
      MultiTypeBlockMatrix_VectMul<crow+1,remain_rows-1,0,TMatrix::M(),TVecY,TMatrix,TVecX>::usmv(alpha,y, A, x);
    }
  };

  //specialization for remain_rows = 0
  template<int crow, int ccol, int remain_cols, typename TVecY,
      typename TMatrix, typename TVecX>
  class MultiTypeBlockMatrix_VectMul<crow,0,ccol,remain_cols,TVecY,TMatrix,TVecX> {
    //end recursion
  public:
    static void umv(TVecY&, const TMatrix&, const TVecX&) {}
    static void mmv(TVecY&, const TMatrix&, const TVecX&) {}

    template<typename AlphaType>
    static void usmv(const AlphaType&, TVecY&, const TMatrix&, const TVecX&) {}
  };






  /**
      @brief A Matrix class to support different block types

      This matrix class combines MultiTypeBlockVector elements as rows.
   */
  template<typename FirstRow, typename... Args>
  class MultiTypeBlockMatrix
  : public std::tuple<FirstRow, Args...>
  {
  public:

    /**
     * own class' type
     */
    typedef MultiTypeBlockMatrix<FirstRow, Args...> type;

    typedef typename FirstRow::field_type field_type;

    /** \brief Return the number of matrix rows */
    static constexpr std::size_t N()
    {
      return 1+sizeof...(Args);
    }

    /** \brief Return the number of matrix columns */
    static constexpr std::size_t M()
    {
      return FirstRow::size();
    }

    /** \brief Random-access operator
     *
     * This method mimicks the behavior of normal vector access with square brackets like, e.g., m[5] = ....
     * The problem is that the return type is different for each value of the argument in the brackets.
     * Therefore we implement a trick using std::integral_constant.  To access the first row of
     * a MultiTypeBlockMatrix named m write
     * \code
     *  std::integral_constant<int,0> _0;
     *  m[_0] = ...
     * \endcode
     * The name '_0' used here as a static replacement of the integer number zero is arbitrary.
     * Any other variable name can be used.  If you don't like the separate variable, you can writee
     * \code
     *  m[std::integral_constant<int,0>()] = ...
     * \endcode
     */
    template< int index >
    auto
    operator[] ( const std::integral_constant< int, index > indexVariable ) -> decltype(std::get<index>(*this))
    {
      DUNE_UNUSED_PARAMETER(indexVariable);
      return std::get<index>(*this);
    }

    /** \brief Random-access operator
     *
     * This method mimicks the behavior of normal vector access with square brackets like, e.g., m[5] = ....
     * The problem is that the return type is different for each value of the argument in the brackets.
     * Therefore we implement a trick using std::integral_constant.  To access the first row of
     * a MultiTypeBlockMatrix named m write
     * \code
     *  std::integral_constant<std::size_t,0> _0;
     *  m[_0] = ...
     * \endcode
     * The name '_0' used here as a static replacement of the integer number zero is arbitrary.
     * Any other variable name can be used.  If you don't like the separate variable, you can writee
     * \code
     *  m[std::integral_constant<std::size_t,0>()] = ...
     * \endcode
     */
    template< std::size_t index >
    auto
    operator[] ( const std::integral_constant< std::size_t, index > indexVariable ) -> decltype(std::get<index>(*this))
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
    auto
    operator[] ( const std::integral_constant< int, index > indexVariable ) const -> decltype(std::get<index>(*this))
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
    auto
    operator[] ( const std::integral_constant< std::size_t, index > indexVariable ) const -> decltype(std::get<index>(*this))
    {
      DUNE_UNUSED_PARAMETER(indexVariable);
      return std::get<index>(*this);
    }

    /**
     * assignment operator
     */
    template<typename T>
    void operator= (const T& newval) {MultiTypeBlockMatrix_Ident<N(),type,T>::equalize(*this, newval); }

    /** \brief y = A x
     */
    template<typename X, typename Y>
    void mv (const X& x, Y& y) const {
      static_assert(X::size() == M(), "length of x does not match row length");
      static_assert(Y::size() == N(), "length of y does not match row count");
      y = 0;                                                                  //reset y (for mv uses umv)
      MultiTypeBlockMatrix_VectMul<0,N(),0,M(),Y,type,X>::umv(y, *this, x);    //iterate over all matrix elements
    }

    /** \brief y += A x
     */
    template<typename X, typename Y>
    void umv (const X& x, Y& y) const {
      static_assert(X::size() == M(), "length of x does not match row length");
      static_assert(Y::size() == N(), "length of y does not match row count");
      MultiTypeBlockMatrix_VectMul<0,N(),0,M(),Y,type,X>::umv(y, *this, x);    //iterate over all matrix elements
    }

    /** \brief y -= A x
     */
    template<typename X, typename Y>
    void mmv (const X& x, Y& y) const {
      static_assert(X::size() == M(), "length of x does not match row length");
      static_assert(Y::size() == N(), "length of y does not match row count");
      MultiTypeBlockMatrix_VectMul<0,N(),0,M(),Y,type,X>::mmv(y, *this, x);    //iterate over all matrix elements
    }

    /** \brief y += alpha A x
     */
    template<typename AlphaType, typename X, typename Y>
    void usmv (const AlphaType& alpha, const X& x, Y& y) const {
      static_assert(X::size() == M(), "length of x does not match row length");
      static_assert(Y::size() == N(), "length of y does not match row count");
      MultiTypeBlockMatrix_VectMul<0,N(),0,M(),Y,type,X>::usmv(alpha,y, *this, x);     //iterate over all matrix elements
    }

  };



  /**
     @brief << operator for a MultiTypeBlockMatrix

     operator<< for printing out a MultiTypeBlockMatrix
   */
  template<typename T1, typename... Args>
  std::ostream& operator<< (std::ostream& s, const MultiTypeBlockMatrix<T1,Args...>& m) {
    static const int N = MultiTypeBlockMatrix<T1,Args...>::N();
    static const int M = MultiTypeBlockMatrix<T1,Args...>::M();
    MultiTypeBlockMatrix_Print<0,N,0,M,MultiTypeBlockMatrix<T1,Args...> >::print(m);
    return s;
  }

  //make algmeta_itsteps known
  template<int I, typename M>
  struct algmeta_itsteps;

  /**
     @brief part of solvers for MultiTypeBlockVector & MultiTypeBlockMatrix types

     For the given row (index "crow") each element is used to
     calculate the equation's right side.
   */
  template<int I, int crow, int ccol, int remain_col>                             //MultiTypeBlockMatrix_Solver_Col: iterating over one row
  class MultiTypeBlockMatrix_Solver_Col {                                                      //calculating b- A[i][j]*x[j]
  public:
    /**
     * iterating over one row in MultiTypeBlockMatrix to calculate right side b-A[i][j]*x[j]
     */
    template <typename Trhs, typename TVector, typename TMatrix, typename K>
    static void calc_rhs(const TMatrix& A, TVector& x, TVector& v, Trhs& b, const K& w) {
      std::get<ccol>( std::get<crow>(A) ).mmv( std::get<ccol>(x), b );
      MultiTypeBlockMatrix_Solver_Col<I, crow, ccol+1, remain_col-1>::calc_rhs(A,x,v,b,w); //next column element
    }

  };
  template<int I, int crow, int ccol>                                             //MultiTypeBlockMatrix_Solver_Col recursion end
  class MultiTypeBlockMatrix_Solver_Col<I,crow,ccol,0> {
  public:
    template <typename Trhs, typename TVector, typename TMatrix, typename K>
    static void calc_rhs(const TMatrix&, TVector&, TVector&, Trhs&, const K&) {}
  };



  /**
     @brief solver for MultiTypeBlockVector & MultiTypeBlockMatrix types

     The methods of this class are called by the solver specializations
     for MultiTypeBlockVector & MultiTypeBlockMatrix types (dbgs, bsorf, bsorb, dbjac).
   */
  template<int I, int crow, int remain_row>
  class MultiTypeBlockMatrix_Solver {
  public:

    /**
     * Gauss-Seidel solver for MultiTypeBlockMatrix & MultiTypeBlockVector types
     */
    template <typename TVector, typename TMatrix, typename K>
    static void dbgs(const TMatrix& A, TVector& x, const TVector& b, const K& w) {
      TVector xold(x);
      xold=x;                                                         //store old x values
      MultiTypeBlockMatrix_Solver<I,crow,remain_row>::dbgs(A,x,x,b,w);
      x *= w;
      x.axpy(1-w,xold);                                                       //improve x
    }
    template <typename TVector, typename TMatrix, typename K>
    static void dbgs(const TMatrix& A, TVector& x, TVector& v, const TVector& b, const K& w) {
      auto rhs = std::get<crow> (b);

      MultiTypeBlockMatrix_Solver_Col<I,crow,0, TMatrix::M()>::calc_rhs(A,x,v,rhs,w);  // calculate right side of equation
      //solve on blocklevel I-1
      using M =
        typename std::remove_cv<
          typename std::remove_reference<
            decltype(std::get<crow>( std::get<crow>(A)))
          >::type
        >::type;
      algmeta_itsteps<I-1,M>::dbgs(std::get<crow>( std::get<crow>(A)), std::get<crow>(x),rhs,w);
      MultiTypeBlockMatrix_Solver<I,crow+1,remain_row-1>::dbgs(A,x,v,b,w); //next row
    }



    /**
     * bsorf for MultiTypeBlockMatrix & MultiTypeBlockVector types
     */
    template <typename TVector, typename TMatrix, typename K>
    static void bsorf(const TMatrix& A, TVector& x, const TVector& b, const K& w) {
      TVector v;
      v=x;                                                            //use latest x values in right side calculation
      MultiTypeBlockMatrix_Solver<I,crow,remain_row>::bsorf(A,x,v,b,w);

    }
    template <typename TVector, typename TMatrix, typename K>               //recursion over all matrix rows (A)
    static void bsorf(const TMatrix& A, TVector& x, TVector& v, const TVector& b, const K& w) {
      auto rhs = std::get<crow> (b);

      MultiTypeBlockMatrix_Solver_Col<I,crow,0,TMatrix::M()>::calc_rhs(A,x,v,rhs,w);  // calculate right side of equation
      //solve on blocklevel I-1
      using M =
        typename std::remove_cv<
          typename std::remove_reference<
            decltype(std::get<crow>( std::get<crow>(A)))
          >::type
        >::type;
      algmeta_itsteps<I-1,M>::bsorf(std::get<crow>( std::get<crow>(A)), std::get<crow>(v),rhs,w);
      std::get<crow>(x).axpy(w,std::get<crow>(v));
      MultiTypeBlockMatrix_Solver<I,crow+1,remain_row-1>::bsorf(A,x,v,b,w);        //next row
    }

    /**
     * bsorb for MultiTypeBlockMatrix & MultiTypeBlockVector types
     */
    template <typename TVector, typename TMatrix, typename K>
    static void bsorb(const TMatrix& A, TVector& x, const TVector& b, const K& w) {
      TVector v;
      v=x;                                                            //use latest x values in right side calculation
      MultiTypeBlockMatrix_Solver<I,crow,remain_row>::bsorb(A,x,v,b,w);

    }
    template <typename TVector, typename TMatrix, typename K>               //recursion over all matrix rows (A)
    static void bsorb(const TMatrix& A, TVector& x, TVector& v, const TVector& b, const K& w) {
      auto rhs = std::get<crow> (b);

      MultiTypeBlockMatrix_Solver_Col<I,crow,0, TMatrix::M()>::calc_rhs(A,x,v,rhs,w);  // calculate right side of equation
      //solve on blocklevel I-1
      using M =
        typename std::remove_cv<
          typename std::remove_reference<
            decltype(std::get<crow>( std::get<crow>(A)))
          >::type
        >::type;
      algmeta_itsteps<I-1,M>::bsorb(std::get<crow>( std::get<crow>(A)), std::get<crow>(v),rhs,w);
      std::get<crow>(x).axpy(w,std::get<crow>(v));
      MultiTypeBlockMatrix_Solver<I,crow-1,remain_row-1>::bsorb(A,x,v,b,w);        //next row
    }


    /**
     * Jacobi solver for MultiTypeBlockMatrix & MultiTypeBlockVector types
     */
    template <typename TVector, typename TMatrix, typename K>
    static void dbjac(const TMatrix& A, TVector& x, const TVector& b, const K& w) {
      TVector v(x);
      v=0;                                                            //calc new x in v
      MultiTypeBlockMatrix_Solver<I,crow,remain_row>::dbjac(A,x,v,b,w);
      x.axpy(w,v);                                                    //improve x
    }
    template <typename TVector, typename TMatrix, typename K>
    static void dbjac(const TMatrix& A, TVector& x, TVector& v, const TVector& b, const K& w) {
      auto rhs = std::get<crow> (b);

      MultiTypeBlockMatrix_Solver_Col<I,crow,0, TMatrix::M()>::calc_rhs(A,x,v,rhs,w);  // calculate right side of equation
      //solve on blocklevel I-1
      using M =
        typename std::remove_cv<
          typename std::remove_reference<
            decltype(std::get<crow>( std::get<crow>(A)))
          >::type
        >::type;
      algmeta_itsteps<I-1,M>::dbjac(std::get<crow>( std::get<crow>(A)), std::get<crow>(v),rhs,w);
      MultiTypeBlockMatrix_Solver<I,crow+1,remain_row-1>::dbjac(A,x,v,b,w);        //next row
    }




  };
  template<int I, int crow>                                                       //recursion end for remain_row = 0
  class MultiTypeBlockMatrix_Solver<I,crow,0> {
  public:
    template <typename TVector, typename TMatrix, typename K>
    static void dbgs(const TMatrix&, TVector&, TVector&,
                     const TVector&, const K&) {}

    template <typename TVector, typename TMatrix, typename K>
    static void bsorf(const TMatrix&, TVector&, TVector&,
                      const TVector&, const K&) {}

    template <typename TVector, typename TMatrix, typename K>
    static void bsorb(const TMatrix&, TVector&, TVector&,
                      const TVector&, const K&) {}

    template <typename TVector, typename TMatrix, typename K>
    static void dbjac(const TMatrix&, TVector&, TVector&,
                      const TVector&, const K&) {}
  };

} // end namespace

#endif
