// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GSETC_HH
#define DUNE_GSETC_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>
#include "multitypeblockvector.hh"
#include "multitypeblockmatrix.hh"

#include "istlexception.hh"


/*! \file
   \brief Simple iterative methods like Jacobi, Gauss-Seidel, SOR, SSOR, etc.
    in a generic way
 */

namespace Dune {

  /**
   * @defgroup ISTL_Kernel Block Recursive Iterative Kernels
   * @ingroup ISTL_SPMV
   *
   * Generic iterative kernels for the solvers which work on the block recursive
   * structure of the matrices and vectors.
   * @addtogroup ISTL_Kernel
   * @{
   */

  //============================================================
  // parameter types
  //============================================================

  //! compile-time parameter for block recursion depth
  template<int l>
  struct BL {
    enum {recursion_level = l};
  };

  enum WithDiagType {
    withdiag=1,
    nodiag=0
  };

  enum WithRelaxType {
    withrelax=1,
    norelax=0
  };

  //============================================================
  // generic triangular solves
  // consider block decomposition A = L + D + U
  // we can invert L, L+D, U, U+D
  // we can apply relaxation or not
  // we can recurse over a fixed number of levels
  //============================================================

  // template meta program for triangular solves
  template<int I, WithDiagType diag, WithRelaxType relax>
  struct algmeta_btsolve {
    template<class M, class X, class Y, class K>
    static void bltsolve (const M& A, X& v, const Y& d, const K& w)
    {
      // iterator types
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;
      typedef typename Y::block_type bblock;

      // local solve at each block and immediate update
      rowiterator endi=A.end();
      for (rowiterator i=A.begin(); i!=endi; ++i)
      {
        bblock rhs(d[i.index()]);
        coliterator j;
        for (j=(*i).begin(); j.index()<i.index(); ++j)
          (*j).mmv(v[j.index()],rhs);
        algmeta_btsolve<I-1,diag,relax>::bltsolve(*j,v[i.index()],rhs,w);
      }
    }
    template<class M, class X, class Y, class K>
    static void butsolve (const M& A, X& v, const Y& d, const K& w)
    {
      // iterator types
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;
      typedef typename Y::block_type bblock;

      // local solve at each block and immediate update
      rowiterator rendi=A.beforeBegin();
      for (rowiterator i=A.beforeEnd(); i!=rendi; --i)
      {
        bblock rhs(d[i.index()]);
        coliterator j;
        for (j=(*i).beforeEnd(); j.index()>i.index(); --j)
          (*j).mmv(v[j.index()],rhs);
        algmeta_btsolve<I-1,diag,relax>::butsolve(*j,v[i.index()],rhs,w);
      }
    }
  };

  // recursion end ...
  template<>
  struct algmeta_btsolve<0,withdiag,withrelax> {
    template<class M, class X, class Y, class K>
    static void bltsolve (const M& A, X& v, const Y& d, const K& w)
    {
      A.solve(v,d);
      v *= w;
    }
    template<class M, class X, class Y, class K>
    static void butsolve (const M& A, X& v, const Y& d, const K& w)
    {
      A.solve(v,d);
      v *= w;
    }
  };
  template<>
  struct algmeta_btsolve<0,withdiag,norelax> {
    template<class M, class X, class Y, class K>
    static void bltsolve (const M& A, X& v, const Y& d, const K& /*w*/)
    {
      A.solve(v,d);
    }
    template<class M, class X, class Y, class K>
    static void butsolve (const M& A, X& v, const Y& d, const K& /*w*/)
    {
      A.solve(v,d);
    }
  };
  template<>
  struct algmeta_btsolve<0,nodiag,withrelax> {
    template<class M, class X, class Y, class K>
    static void bltsolve (const M& /*A*/, X& v, const Y& d, const K& w)
    {
      v = d;
      v *= w;
    }
    template<class M, class X, class Y, class K>
    static void butsolve (const M& /*A*/, X& v, const Y& d, const K& w)
    {
      v = d;
      v *= w;
    }
  };
  template<>
  struct algmeta_btsolve<0,nodiag,norelax> {
    template<class M, class X, class Y, class K>
    static void bltsolve (const M& /*A*/, X& v, const Y& d, const K& /*w*/)
    {
      v = d;
    }
    template<class M, class X, class Y, class K>
    static void butsolve (const M& /*A*/, X& v, const Y& d, const K& /*w*/)
    {
      v = d;
    }
  };


  // user calls

  // default block recursion level = 1

  //! block lower triangular solve
  template<class M, class X, class Y>
  void bltsolve (const M& A, X& v, const Y& d)
  {
    typename X::field_type w=1;
    algmeta_btsolve<1,withdiag,norelax>::bltsolve(A,v,d,w);
  }
  //! relaxed block lower triangular solve
  template<class M, class X, class Y, class K>
  void bltsolve (const M& A, X& v, const Y& d, const K& w)
  {
    algmeta_btsolve<1,withdiag,withrelax>::bltsolve(A,v,d,w);
  }
  //! unit block lower triangular solve
  template<class M, class X, class Y>
  void ubltsolve (const M& A, X& v, const Y& d)
  {
    typename X::field_type w=1;
    algmeta_btsolve<1,nodiag,norelax>::bltsolve(A,v,d,w);
  }
  //! relaxed unit block lower triangular solve
  template<class M, class X, class Y, class K>
  void ubltsolve (const M& A, X& v, const Y& d, const K& w)
  {
    algmeta_btsolve<1,nodiag,withrelax>::bltsolve(A,v,d,w);
  }

  //! block upper triangular solve
  template<class M, class X, class Y>
  void butsolve (const M& A, X& v, const Y& d)
  {
    typename X::field_type w=1;
    algmeta_btsolve<1,withdiag,norelax>::butsolve(A,v,d,w);
  }
  //! relaxed block upper triangular solve
  template<class M, class X, class Y, class K>
  void butsolve (const M& A, X& v, const Y& d, const K& w)
  {
    algmeta_btsolve<1,withdiag,withrelax>::butsolve(A,v,d,w);
  }
  //! unit block upper triangular solve
  template<class M, class X, class Y>
  void ubutsolve (const M& A, X& v, const Y& d)
  {
    typename X::field_type w=1;
    algmeta_btsolve<1,nodiag,norelax>::butsolve(A,v,d,w);
  }
  //! relaxed unit block upper triangular solve
  template<class M, class X, class Y, class K>
  void ubutsolve (const M& A, X& v, const Y& d, const K& w)
  {
    algmeta_btsolve<1,nodiag,withrelax>::butsolve(A,v,d,w);
  }

  // general block recursion level >= 0

  //! block lower triangular solve
  template<class M, class X, class Y, int l>
  void bltsolve (const M& A, X& v, const Y& d, BL<l> /*bl*/)
  {
    typename X::field_type w=1;
    algmeta_btsolve<l,withdiag,norelax>::bltsolve(A,v,d,w);
  }
  //! relaxed block lower triangular solve
  template<class M, class X, class Y, class K, int l>
  void bltsolve (const M& A, X& v, const Y& d, const K& w, BL<l> /*bl*/)
  {
    algmeta_btsolve<l,withdiag,withrelax>::bltsolve(A,v,d,w);
  }
  //! unit block lower triangular solve
  template<class M, class X, class Y, int l>
  void ubltsolve (const M& A, X& v, const Y& d, BL<l> /*bl*/)
  {
    typename X::field_type w=1;
    algmeta_btsolve<l,nodiag,norelax>::bltsolve(A,v,d,w);
  }
  //! relaxed unit block lower triangular solve
  template<class M, class X, class Y, class K, int l>
  void ubltsolve (const M& A, X& v, const Y& d, const K& w, BL<l> /*bl*/)
  {
    algmeta_btsolve<l,nodiag,withrelax>::bltsolve(A,v,d,w);
  }

  //! block upper triangular solve
  template<class M, class X, class Y, int l>
  void butsolve (const M& A, X& v, const Y& d, BL<l> bl)
  {
    typename X::field_type w=1;
    algmeta_btsolve<l,withdiag,norelax>::butsolve(A,v,d,w);
  }
  //! relaxed block upper triangular solve
  template<class M, class X, class Y, class K, int l>
  void butsolve (const M& A, X& v, const Y& d, const K& w, BL<l> bl)
  {
    algmeta_btsolve<l,withdiag,withrelax>::butsolve(A,v,d,w);
  }
  //! unit block upper triangular solve
  template<class M, class X, class Y, int l>
  void ubutsolve (const M& A, X& v, const Y& d, BL<l> bl)
  {
    typename X::field_type w=1;
    algmeta_btsolve<l,nodiag,norelax>::butsolve(A,v,d,w);
  }
  //! relaxed unit block upper triangular solve
  template<class M, class X, class Y, class K, int l>
  void ubutsolve (const M& A, X& v, const Y& d, const K& w, BL<l> bl)
  {
    algmeta_btsolve<l,nodiag,withrelax>::butsolve(A,v,d,w);
  }



  //============================================================
  // generic block diagonal solves
  // consider block decomposition A = L + D + U
  // we can apply relaxation or not
  // we can recurse over a fixed number of levels
  //============================================================

  // template meta program for diagonal solves
  template<int I, WithRelaxType relax>
  struct algmeta_bdsolve {
    template<class M, class X, class Y, class K>
    static void bdsolve (const M& A, X& v, const Y& d, const K& w)
    {
      // iterator types
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;

      // local solve at each block and immediate update
      rowiterator rendi=A.beforeBegin();
      for (rowiterator i=A.beforeEnd(); i!=rendi; --i)
      {
        coliterator ii=(*i).find(i.index());
        algmeta_bdsolve<I-1,relax>::bdsolve(*ii,v[i.index()],d[i.index()],w);
      }
    }
  };

  // recursion end ...
  template<>
  struct algmeta_bdsolve<0,withrelax> {
    template<class M, class X, class Y, class K>
    static void bdsolve (const M& A, X& v, const Y& d, const K& w)
    {
      A.solve(v,d);
      v *= w;
    }
  };
  template<>
  struct algmeta_bdsolve<0,norelax> {
    template<class M, class X, class Y, class K>
    static void bdsolve (const M& A, X& v, const Y& d, const K& /*w*/)
    {
      A.solve(v,d);
    }
  };

  // user calls

  // default block recursion level = 1

  //! block diagonal solve, no relaxation
  template<class M, class X, class Y>
  void bdsolve (const M& A, X& v, const Y& d)
  {
    typename X::field_type w=1;
    algmeta_bdsolve<1,norelax>::bdsolve(A,v,d,w);
  }
  //! block diagonal solve, with relaxation
  template<class M, class X, class Y, class K>
  void bdsolve (const M& A, X& v, const Y& d, const K& w)
  {
    algmeta_bdsolve<1,withrelax>::bdsolve(A,v,d,w);
  }

  // general block recursion level >= 0

  //! block diagonal solve, no relaxation
  template<class M, class X, class Y, int l>
  void bdsolve (const M& A, X& v, const Y& d, BL<l> /*bl*/)
  {
    typename X::field_type w=1;
    algmeta_bdsolve<l,norelax>::bdsolve(A,v,d,w);
  }
  //! block diagonal solve, with relaxation
  template<class M, class X, class Y, class K, int l>
  void bdsolve (const M& A, X& v, const Y& d, const K& w, BL<l> /*bl*/)
  {
    algmeta_bdsolve<l,withrelax>::bdsolve(A,v,d,w);
  }





  //============================================================
  // generic steps of iteration methods
  // Jacobi, Gauss-Seidel, SOR, SSOR
  // work directly on Ax=b, ie solve M(x^{i+1}-x^i) = w (b-Ax^i)
  // we can recurse over a fixed number of levels
  //============================================================


  // template meta program for iterative solver steps
  template<int I>
  struct algmeta_itsteps {

#if HAVE_DUNE_BOOST
#ifdef HAVE_BOOST_FUSION

    template<typename T11, typename T12, typename T13, typename T14,
        typename T15, typename T16, typename T17, typename T18,
        typename T19, typename T21, typename T22, typename T23,
        typename T24, typename T25, typename T26, typename T27,
        typename T28, typename T29, class K>
    static void dbgs (const MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19>& A,
                      MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& x,
                      const MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& b,
                      const K& w)
    {
      const int rowcount =
        boost::mpl::size<MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19> >::value;
      Dune::MultiTypeBlockMatrix_Solver<I,0,rowcount>::dbgs(A, x, b, w);
    }
#endif
#endif

    template<class M, class X, class Y, class K>
    static void dbgs (const M& A, X& x, const Y& b, const K& w)
    {
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;
      typedef typename Y::block_type bblock;
      bblock rhs;

      X xold(x);     // remember old x

      rowiterator endi=A.end();
      for (rowiterator i=A.begin(); i!=endi; ++i)
      {
        rhs = b[i.index()];           // rhs = b_i
        coliterator endj=(*i).end();
        coliterator j=(*i).begin();
        for (; j.index()<i.index(); ++j)           // iterate over a_ij with j < i
          (*j).mmv(x[j.index()],rhs);               // rhs -= sum_{j<i} a_ij * xnew_j
        coliterator diag=j++;           // *diag = a_ii and increment coliterator j from a_ii to a_i+1,i to skip diagonal
        for (; j != endj; ++j)           // iterate over a_ij with j > i
          (*j).mmv(x[j.index()],rhs);               // rhs -= sum_{j>i} a_ij * xold_j
        algmeta_itsteps<I-1>::dbgs(*diag,x[i.index()],rhs,w);           // if I==1: xnew_i = rhs/a_ii
      }
      // next two lines: xnew_i = w / a_ii * (b_i - sum_{j<i} a_ij * xnew_j - sum_{j>=i} a_ij * xold_j) + (1-w)*xold;
      x *= w;
      x.axpy(K(1)-w,xold);
    }

#if HAVE_DUNE_BOOST
#ifdef HAVE_BOOST_FUSION

    template<typename T11, typename T12, typename T13, typename T14,
        typename T15, typename T16, typename T17, typename T18,
        typename T19, typename T21, typename T22, typename T23,
        typename T24, typename T25, typename T26, typename T27,
        typename T28, typename T29, class K>
    static void bsorf (const MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19>& A,
                       MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& x,
                       const MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& b,
                       const K& w)
    {
      const int rowcount =
        boost::mpl::size<MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19> >::value;
      Dune::MultiTypeBlockMatrix_Solver<I,0,rowcount>::bsorf(A, x, b, w);
    }
#endif
#endif

    template<class M, class X, class Y, class K>
    static void bsorf (const M& A, X& x, const Y& b, const K& w)
    {
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;
      typedef typename Y::block_type bblock;
      typedef typename X::block_type xblock;
      bblock rhs;
      xblock v;

      // Initialize nested data structure if there are entries
      if(A.begin()!=A.end())
        v=x[0];

      rowiterator endi=A.end();
      for (rowiterator i=A.begin(); i!=endi; ++i)
      {
        rhs = b[i.index()];           // rhs = b_i
        coliterator endj=(*i).end();           // iterate over a_ij with j < i
        coliterator j=(*i).begin();
        for (; j.index()<i.index(); ++j)
          (*j).mmv(x[j.index()],rhs);               //  rhs -= sum_{j<i} a_ij * xnew_j
        coliterator diag=j;           // *diag = a_ii
        for (; j!=endj; ++j)
          (*j).mmv(x[j.index()],rhs);               // rhs -= sum_{j<i} a_ij * xnew_j
        algmeta_itsteps<I-1>::bsorf(*diag,v,rhs,w);           // if blocksize I==1: v = rhs/a_ii
        x[i.index()].axpy(w,v);           // x_i = w / a_ii * (b_i - sum_{j<i} a_ij * xnew_j - sum_{j>=i} a_ij * xold_j)
      }
    }

#if HAVE_DUNE_BOOST
#ifdef HAVE_BOOST_FUSION

    template<typename T11, typename T12, typename T13, typename T14,
        typename T15, typename T16, typename T17, typename T18,
        typename T19, typename T21, typename T22, typename T23,
        typename T24, typename T25, typename T26, typename T27,
        typename T28, typename T29, class K>
    static void bsorb (const MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19>& A,
                       MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& x,
                       const MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& b,
                       const K& w)
    {
      const int rowcount =
        mpl::size<MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19> >::value;
      Dune::MultiTypeBlockMatrix_Solver<I,rowcount-1,rowcount>::bsorb(A, x, b, w);
    }
#endif
#endif

    template<class M, class X, class Y, class K>
    static void bsorb (const M& A, X& x, const Y& b, const K& w)
    {
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;
      typedef typename Y::block_type bblock;
      typedef typename X::block_type xblock;
      bblock rhs;
      xblock v;

      // Initialize nested data structure if there are entries
      if(A.begin()!=A.end())
        v=x[0];

      rowiterator endi=A.beforeBegin();
      for (rowiterator i=A.beforeEnd(); i!=endi; --i)
      {
        rhs = b[i.index()];
        coliterator endj=(*i).end();
        coliterator j=(*i).begin();
        for (; j.index()<i.index(); ++j)
          (*j).mmv(x[j.index()],rhs);
        coliterator diag=j;
        for (; j!=endj; ++j)
          (*j).mmv(x[j.index()],rhs);
        algmeta_itsteps<I-1>::bsorb(*diag,v,rhs,w);
        x[i.index()].axpy(w,v);
      }
    }

#if HAVE_DUNE_BOOST
#ifdef HAVE_BOOST_FUSION

    template<typename T11, typename T12, typename T13, typename T14,
        typename T15, typename T16, typename T17, typename T18,
        typename T19, typename T21, typename T22, typename T23,
        typename T24, typename T25, typename T26, typename T27,
        typename T28, typename T29, class K>
    static void dbjac (const MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19>& A,
                       MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& x,
                       const MultiTypeBlockVector<T21,T22,T23,T24,T25,T26,T27,T28,T29>& b,
                       const K& w)
    {
      const int rowcount =
        boost::mpl::size<MultiTypeBlockMatrix<T11,T12,T13,T14,T15,T16,T17,T18,T19> >::value;
      Dune::MultiTypeBlockMatrix_Solver<I,0,rowcount >::dbjac(A, x, b, w);
    }
#endif
#endif

    template<class M, class X, class Y, class K>
    static void dbjac (const M& A, X& x, const Y& b, const K& w)
    {
      typedef typename M::ConstRowIterator rowiterator;
      typedef typename M::ConstColIterator coliterator;
      typedef typename Y::block_type bblock;
      bblock rhs;

      X v(x);     // allocate with same size

      rowiterator endi=A.end();
      for (rowiterator i=A.begin(); i!=endi; ++i)
      {
        rhs = b[i.index()];
        coliterator endj=(*i).end();
        coliterator j=(*i).begin();
        for (; j.index()<i.index(); ++j)
          (*j).mmv(x[j.index()],rhs);
        coliterator diag=j;
        for (; j!=endj; ++j)
          (*j).mmv(x[j.index()],rhs);
        algmeta_itsteps<I-1>::dbjac(*diag,v[i.index()],rhs,w);
      }
      x.axpy(w,v);
    }
  };
  // end of recursion
  template<>
  struct algmeta_itsteps<0> {
    template<class M, class X, class Y, class K>
    static void dbgs (const M& A, X& x, const Y& b, const K& /*w*/)
    {
      A.solve(x,b);
    }
    template<class M, class X, class Y, class K>
    static void bsorf (const M& A, X& x, const Y& b, const K& /*w*/)
    {
      A.solve(x,b);
    }
    template<class M, class X, class Y, class K>
    static void bsorb (const M& A, X& x, const Y& b, const K& /*w*/)
    {
      A.solve(x,b);
    }
    template<class M, class X, class Y, class K>
    static void dbjac (const M& A, X& x, const Y& b, const K& /*w*/)
    {
      A.solve(x,b);
    }
  };


  // user calls

  //! GS step
  template<class M, class X, class Y, class K>
  void dbgs (const M& A, X& x, const Y& b, const K& w)
  {
    algmeta_itsteps<1>::dbgs(A,x,b,w);
  }
  //! GS step
  template<class M, class X, class Y, class K, int l>
  void dbgs (const M& A, X& x, const Y& b, const K& w, BL<l> /*bl*/)
  {
    algmeta_itsteps<l>::dbgs(A,x,b,w);
  }
  //! SOR step
  template<class M, class X, class Y, class K>
  void bsorf (const M& A, X& x, const Y& b, const K& w)
  {
    algmeta_itsteps<1>::bsorf(A,x,b,w);
  }
  //! SOR step
  template<class M, class X, class Y, class K, int l>
  void bsorf (const M& A, X& x, const Y& b, const K& w, BL<l> /*bl*/)
  {
    algmeta_itsteps<l>::bsorf(A,x,b,w);
  }
  //! SSOR step
  template<class M, class X, class Y, class K>
  void bsorb (const M& A, X& x, const Y& b, const K& w)
  {
    algmeta_itsteps<1>::bsorb(A,x,b,w);
  }
  //! SSOR step
  template<class M, class X, class Y, class K, int l>
  void bsorb (const M& A, X& x, const Y& b, const K& w, BL<l> /*bl*/)
  {
    algmeta_itsteps<l>::bsorb(A,x,b,w);
  }
  //! Jacobi step
  template<class M, class X, class Y, class K>
  void dbjac (const M& A, X& x, const Y& b, const K& w)
  {
    algmeta_itsteps<1>::dbjac(A,x,b,w);
  }
  //! Jacobi step
  template<class M, class X, class Y, class K, int l>
  void dbjac (const M& A, X& x, const Y& b, const K& w, BL<l> /*bl*/)
  {
    algmeta_itsteps<l>::dbjac(A,x,b,w);
  }


  /** @} end documentation */

} // end namespace

#endif
