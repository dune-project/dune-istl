// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_OVERLAPPINGSCHWARZ_HH
#define DUNE_ISTL_OVERLAPPINGSCHWARZ_HH
#include <cassert>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <set>
#include <dune/common/dynmatrix.hh>
#include <dune/common/sllist.hh>

#include <dune/istl/bccsmatrixinitializer.hh>
#include "preconditioners.hh"
#include "superlu.hh"
#include "umfpack.hh"
#include "bvector.hh"
#include "bcrsmatrix.hh"
#include "ilusubdomainsolver.hh"
#include <dune/istl/solvertype.hh>

namespace Dune
{

  /**
   * @addtogroup ISTL_Prec
   *
   * @{
   */
  /**
   * @file
   * @author Markus Blatt
   * @brief Contains one level overlapping Schwarz preconditioners
   */

  template<class M, class X, class TM, class TD, class TA>
  class SeqOverlappingSchwarz;

  /**
   * @brief Initializer for SuperLU Matrices representing the subdomains.
   */
  template<class I, class S, class D>
  class OverlappingSchwarzInitializer
  {
  public:
    /** @brief The vector type containing the subdomain to row index mapping. */
    typedef D subdomain_vector;

    typedef I InitializerList;
    typedef typename InitializerList::value_type AtomInitializer;
    typedef typename AtomInitializer::Matrix Matrix;
    typedef typename Matrix::const_iterator Iter;
    typedef typename Matrix::row_type::const_iterator CIter;

    typedef S IndexSet;
    typedef typename IndexSet::size_type size_type;

    OverlappingSchwarzInitializer(InitializerList& il,
                                  const IndexSet& indices,
                                  const subdomain_vector& domains);


    void addRowNnz(const Iter& row);

    void allocate();

    void countEntries(const Iter& row, const CIter& col) const;

    void calcColstart() const;

    void copyValue(const Iter& row, const CIter& col) const;

    void createMatrix() const;
  private:
    class IndexMap
    {
    public:
      typedef typename S::size_type size_type;
      typedef std::map<size_type,size_type> Map;
      typedef typename Map::iterator iterator;
      typedef typename Map::const_iterator const_iterator;

      IndexMap();

      void insert(size_type grow);

      const_iterator find(size_type grow) const;

      iterator find(size_type grow);

      iterator begin();

      const_iterator begin() const;

      iterator end();

      const_iterator end() const;

    private:
      std::map<size_type,size_type> map_;
      size_type row;
    };


    typedef typename InitializerList::iterator InitIterator;
    typedef typename IndexSet::const_iterator IndexIteratur;
    InitializerList* initializers;
    const IndexSet *indices;
    mutable std::vector<IndexMap> indexMaps;
    const subdomain_vector& domains;
  };

  /**
   * @brief Tag that the tells the Schwarz method to be additive.
   */
  struct AdditiveSchwarzMode
  {};

  /**
   * @brief Tag that tells the Schwarz method to be multiplicative.
   */
  struct MultiplicativeSchwarzMode
  {};

  /**
   * @brief Tag that tells the Schwarz method to be multiplicative
   * and symmetric.
   */
  struct SymmetricMultiplicativeSchwarzMode
  {};

  /**
   * @brief Exact subdomain solver using Dune::DynamicMatrix<T>::solve
   * @tparam M The type of the matrix.
   */
  template<class M, class X, class Y>
  class DynamicMatrixSubdomainSolver;

  // Specialization for BCRSMatrix
  template<class K, class Al, class X, class Y>
  class DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >
  {
    typedef BCRSMatrix< K, Al> M;
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
    typedef typename X::field_type field_type;
    typedef typename std::remove_const<M>::type rilu_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    static constexpr size_t n = std::decay_t<decltype(Impl::asMatrix(std::declval<K>()))>::rows;

    /**
     * @brief Apply the subdomain solver.
     * @copydoc ILUSubdomainSolver::apply
     */
    void apply (DynamicVector<field_type>& v, DynamicVector<field_type>& d)
    {
      assert(v.size() > 0);
      assert(v.size() == d.size());
      assert(A.rows() <= v.size());
      assert(A.cols() <= v.size());
      size_t sz = A.rows();
      v.resize(sz);
      d.resize(sz);
      A.solve(v,d);
      v.resize(v.capacity());
      d.resize(d.capacity());
    }

    /**
     * @brief Set the data of the local problem.
     *
     * @param BCRS The global matrix.
     * @param rowset The global indices of the local problem.
     * @tparam S The type of the set with the indices.
     */
    template<class S>
    void setSubMatrix(const M& BCRS, S& rowset)
    {
      size_t sz = rowset.size();
      A.resize(sz*n,sz*n);
      typedef typename S::const_iterator SIter;
      size_t r = 0, c = 0;
      for(SIter rowIdx = rowset.begin(), rowEnd=rowset.end();
          rowIdx!= rowEnd; ++rowIdx, r++)
      {
        c = 0;
        for(SIter colIdx = rowset.begin(), colEnd=rowset.end();
            colIdx!= colEnd; ++colIdx, c++)
        {
          if (BCRS[*rowIdx].find(*colIdx) == BCRS[*rowIdx].end())
            continue;
          for (size_t i=0; i<n; i++)
          {
            for (size_t j=0; j<n; j++)
            {
              A[r*n+i][c*n+j] = Impl::asMatrix(BCRS[*rowIdx][*colIdx])[i][j];
            }
          }
        }
      }
    }
  private:
    DynamicMatrix<K> A;
  };

  template<typename T, bool tag>
  class OverlappingAssignerHelper
  {};

  template<typename T>
  using OverlappingAssigner = OverlappingAssignerHelper<T, Dune::StoresColumnCompressed<T>::value>;

  // specialization for DynamicMatrix
  template<class K, class Al, class X, class Y>
  class OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix<K, Al>, X, Y >,false>
  {
  public:
    typedef BCRSMatrix< K, Al> matrix_type;
    typedef typename X::field_type field_type;
    typedef Y range_type;
    typedef typename range_type::block_type block_type;
    typedef typename matrix_type::size_type size_type;
    static constexpr size_t n = std::decay_t<decltype(Impl::asMatrix(std::declval<K>()))>::rows;
    /**
     * @brief Constructor.
     * @param maxlength The maximum entries over all subdomains.
     * @param mat_ The global matrix.
     * @param b_ the global right hand side.
     * @param x_ the global left hand side.
     */
    OverlappingAssignerHelper(std::size_t maxlength, const BCRSMatrix<K, Al>& mat_, const X& b_, Y& x_);

    /**
     * @brief Deallocates memory of the local vector.
     */
    inline
    void deallocate();

    /**
     * @brief Resets the local index to zero.
     */
    inline
    void resetIndexForNextDomain();

    /**
     * @brief Get the local left hand side.
     * @return The local left hand side.
     */
    inline
    DynamicVector<field_type> & lhs();

    /**
     * @brief Get the local right hand side.
     * @return The local right hand side.
     */
    inline
    DynamicVector<field_type> & rhs();

    /**
     * @brief relax the result.
     * @param relax The relaxation parameter.
     */
    inline
    void relaxResult(field_type relax);

    /**
     * @brief calculate one entry of the local defect.
     * @param domainIndex One index of the domain.
     */
    void operator()(const size_type& domainIndex);

    /**
     * @brief Assigns the block to the current local
     * index.
     * At the same time the local defect is calculated
     * for the index and stored in the rhs.
     * Afterwards the is incremented for the next block.
     */
    inline
    void assignResult(block_type& res);

  private:
    /**
     * @brief The global matrix for the defect calculation.
     */
    const matrix_type* mat;
    /** @brief The local right hand side. */
    // we need a pointer, because we have to avoid deep copies
    DynamicVector<field_type> * rhs_;
    /** @brief The local left hand side. */
    // we need a pointer, because we have to avoid deep copies
    DynamicVector<field_type> * lhs_;
    /** @brief The global right hand side for the defect calculation. */
    const range_type* b;
    /** @brief The global right hand side for adding the local update to. */
    range_type* x;
    /** @brief The current local index. */
    std::size_t i;
    /** @brief The maximum entries over all subdomains. */
    std::size_t maxlength_;
  };

#if HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK
  template<template<class> class S, typename T, typename A>
  struct OverlappingAssignerHelper<S<BCRSMatrix<T, A>>, true>
  {
    typedef BCRSMatrix<T, A> matrix_type;
    typedef typename S<BCRSMatrix<T, A>>::range_type range_type;
    typedef typename range_type::field_type field_type;
    typedef typename range_type::block_type block_type;

    typedef typename matrix_type::size_type size_type;

    static constexpr size_t n = std::decay_t<decltype(Impl::asMatrix(std::declval<T>()))>::rows;
    static constexpr size_t m = std::decay_t<decltype(Impl::asMatrix(std::declval<T>()))>::cols;
    /**
     * @brief Constructor.
     * @param maxlength The maximum entries over all subdomains.
     * @param mat The global matrix.
     * @param b the global right hand side.
     * @param x the global left hand side.
     */
    OverlappingAssignerHelper(std::size_t maxlength, const matrix_type& mat,
                        const range_type& b, range_type& x);
    /**
     * @brief Deallocates memory of the local vector.
     * @warning memory is released by the destructor as this Functor
     * is copied and the copy needs to still have the data.
     */
    void deallocate();

    /*
     * @brief Resets the local index to zero.
     */
    void resetIndexForNextDomain();

    /**
     * @brief Get the local left hand side.
     * @return The local left hand side.
     */
    field_type* lhs();

    /**
     * @brief Get the local right hand side.
     * @return The local right hand side.
     */
    field_type* rhs();

    /**
     * @brief relax the result.
     * @param relax The relaxation parameter.
     */
    void relaxResult(field_type relax);

    /**
     * @brief calculate one entry of the local defect.
     * @param domain One index of the domain.
     */
    void operator()(const size_type& domain);

    /**
     * @brief Assigns the block to the current local
     * index.
     * At the same time the local defect is calculated
     * for the index and stored in the rhs.
     * Afterwards the is incremented for the next block.
     */
    void assignResult(block_type& res);

  private:
    /**
     * @brief The global matrix for the defect calculation.
     */
    const matrix_type* mat;
    /** @brief The local right hand side. */
    field_type* rhs_;
    /** @brief The local left hand side. */
    field_type* lhs_;
    /** @brief The global right hand side for the defect calculation. */
    const range_type* b;
    /** @brief The global right hand side for adding the local update to. */
    range_type* x;
    /** @brief The current local index. */
    std::size_t i;
    /** @brief The maximum entries over all subdomains. */
    std::size_t maxlength_;
  };

#endif // HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK

  template<class M, class X, class Y>
  class OverlappingAssignerILUBase
  {
  public:
    typedef M matrix_type;

    typedef typename Y::field_type field_type;

    typedef typename Y::block_type block_type;

    typedef typename matrix_type::size_type size_type;
    /**
     * @brief Constructor.
     * @param maxlength The maximum entries over all subdomains.
     * @param mat The global matrix.
     * @param b the global right hand side.
     * @param x the global left hand side.
     */
    OverlappingAssignerILUBase(std::size_t maxlength, const M& mat,
                               const Y& b, X& x);
    /**
     * @brief Deallocates memory of the local vector.
     * @warning memory is released by the destructor as this Functor
     * is copied and the copy needs to still have the data.
     */
    void deallocate();

    /**
     * @brief Resets the local index to zero.
     */
    void resetIndexForNextDomain();

    /**
     * @brief Get the local left hand side.
     * @return The local left hand side.
     */
    X& lhs();

    /**
     * @brief Get the local right hand side.
     * @return The local right hand side.
     */
    Y& rhs();

    /**
     * @brief relax the result.
     * @param relax The relaxation parameter.
     */
    void relaxResult(field_type relax);

    /**
     * @brief calculate one entry of the local defect.
     * @param domain One index of the domain.
     */
    void operator()(const size_type& domain);

    /**
     * @brief Assigns the block to the current local
     * index.
     * At the same time the local defect is calculated
     * for the index and stored in the rhs.
     * Afterwards the is incremented for the next block.
     */
    void assignResult(block_type& res);

  private:
    /**
     * @brief The global matrix for the defect calculation.
     */
    const M* mat;
    /** @brief The local left hand side. */
    X* lhs_;
    /** @brief The local right hand side. */
    Y* rhs_;
    /** @brief The global right hand side for the defect calculation. */
    const Y* b;
    /** @brief The global left hand side for adding the local update to. */
    X* x;
    /** @brief The maximum entries over all subdomains. */
    size_type i;
  };

  // specialization for ILU0
  template<class M, class X, class Y>
  class OverlappingAssignerHelper<ILU0SubdomainSolver<M,X,Y>, false>
    : public OverlappingAssignerILUBase<M,X,Y>
  {
  public:
    /**
     * @brief Constructor.
     * @param maxlength The maximum entries over all subdomains.
     * @param mat The global matrix.
     * @param b the global right hand side.
     * @param x the global left hand side.
     */
    OverlappingAssignerHelper(std::size_t maxlength, const M& mat,
                        const Y& b, X& x)
      : OverlappingAssignerILUBase<M,X,Y>(maxlength, mat,b,x)
    {}
  };

  // specialization for ILUN
  template<class M, class X, class Y>
  class OverlappingAssignerHelper<ILUNSubdomainSolver<M,X,Y>,false>
    : public OverlappingAssignerILUBase<M,X,Y>
  {
  public:
    /**
     * @brief Constructor.
     * @param maxlength The maximum entries over all subdomains.
     * @param mat The global matrix.
     * @param b the global right hand side.
     * @param x the global left hand side.
     */
    OverlappingAssignerHelper(std::size_t maxlength, const M& mat,
                        const Y& b, X& x)
      : OverlappingAssignerILUBase<M,X,Y>(maxlength, mat,b,x)
    {}
  };

  template<typename S, typename T>
  struct AdditiveAdder
  {};

  template<typename S, typename T, typename A>
  struct AdditiveAdder<S, BlockVector<T,A> >
  {
    typedef typename A::size_type size_type;
    typedef typename std::decay_t<decltype(Impl::asVector(std::declval<T>()))>::field_type field_type;
    AdditiveAdder(BlockVector<T,A>& v, BlockVector<T,A>& x,
                  OverlappingAssigner<S>& assigner, const field_type& relax_);
    void operator()(const size_type& domain);
    void axpy();
    static constexpr size_t n = std::decay_t<decltype(Impl::asVector(std::declval<T>()))>::dimension;

  private:
    BlockVector<T,A>* v;
    BlockVector<T,A>* x;
    OverlappingAssigner<S>* assigner;
    field_type relax;
  };

  template<typename S,typename T>
  struct MultiplicativeAdder
  {};

  template<typename S, typename T, typename A>
  struct MultiplicativeAdder<S, BlockVector<T,A> >
  {
    typedef typename A::size_type size_type;
    typedef typename std::decay_t<decltype(Impl::asVector(std::declval<T>()))>::field_type field_type;
    MultiplicativeAdder(BlockVector<T,A>& v, BlockVector<T,A>& x,
                        OverlappingAssigner<S>& assigner_, const field_type& relax_);
    void operator()(const size_type& domain);
    void axpy();
    static constexpr size_t n = std::decay_t<decltype(Impl::asVector(std::declval<T>()))>::dimension;

  private:
    BlockVector<T,A>* x;
    OverlappingAssigner<S>* assigner;
    field_type relax;
  };

  /**
   * @brief template meta program for choosing  how to add the correction.
   *
   * There are specialization for the additive, the multiplicative, and the symmetric multiplicative mode.
   *
   * \tparam T The Schwarz mode (either AdditiveSchwarzMode or MuliplicativeSchwarzMode or
   * SymmetricMultiplicativeSchwarzMode)
   * \tparam X The vector field type
   */
  template<typename T, class X, class S>
  struct AdderSelector
  {};

  template<class X, class S>
  struct AdderSelector<AdditiveSchwarzMode,X,S>
  {
    typedef AdditiveAdder<S,X> Adder;
  };

  template<class X, class S>
  struct AdderSelector<MultiplicativeSchwarzMode,X,S>
  {
    typedef MultiplicativeAdder<S,X> Adder;
  };

  template<class X, class S>
  struct AdderSelector<SymmetricMultiplicativeSchwarzMode,X,S>
  {
    typedef MultiplicativeAdder<S,X> Adder;
  };

  /**
   * @brief Helper template meta program for application of overlapping Schwarz.
   *
   * The is needed because when using the multiplicative Schwarz version one
   * might still want to make multigrid symmetric, i.e. forward sweep when pre-
   * and backward sweep when post-smoothing.
   *
   * @tparam T1 type of the vector with the subdomain solvers.
   * @tparam T2 type of the vector with the subdomain vector fields.
   * @tparam forward If true apply in a forward sweep.
   */
  template<typename T1, typename T2, bool forward>
  struct IteratorDirectionSelector
  {
    typedef T1 solver_vector;
    typedef typename solver_vector::iterator solver_iterator;
    typedef T2 subdomain_vector;
    typedef typename subdomain_vector::const_iterator domain_iterator;

    static solver_iterator begin(solver_vector& sv)
    {
      return sv.begin();
    }

    static solver_iterator end(solver_vector& sv)
    {
      return sv.end();
    }
    static domain_iterator begin(const subdomain_vector& sv)
    {
      return sv.begin();
    }

    static domain_iterator end(const subdomain_vector& sv)
    {
      return sv.end();
    }
  };

  template<typename T1, typename T2>
  struct IteratorDirectionSelector<T1,T2,false>
  {
    typedef T1 solver_vector;
    typedef typename solver_vector::reverse_iterator solver_iterator;
    typedef T2 subdomain_vector;
    typedef typename subdomain_vector::const_reverse_iterator domain_iterator;

    static solver_iterator begin(solver_vector& sv)
    {
      return sv.rbegin();
    }

    static solver_iterator end(solver_vector& sv)
    {
      return sv.rend();
    }
    static domain_iterator begin(const subdomain_vector& sv)
    {
      return sv.rbegin();
    }

    static domain_iterator end(const subdomain_vector& sv)
    {
      return sv.rend();
    }
  };

  /**
   * @brief Helper template meta program for application of overlapping Schwarz.
   *
   * The is needed because when using the multiplicative Schwarz version one
   * might still want to make multigrid symmetric, i.e. forward sweep when pre-
   * and backward sweep when post-smoothing.
   * @tparam T The smoother to apply.
   */
  template<class T>
  struct SeqOverlappingSchwarzApplier
  {
    typedef T smoother;
    typedef typename smoother::range_type range_type;

    static void apply(smoother& sm, range_type& v, const range_type& b)
    {
      sm.template apply<true>(v, b);
    }
  };

  template<class M, class X, class TD, class TA>
  struct SeqOverlappingSchwarzApplier<SeqOverlappingSchwarz<M,X,SymmetricMultiplicativeSchwarzMode,TD,TA> >
  {
    typedef SeqOverlappingSchwarz<M,X,SymmetricMultiplicativeSchwarzMode,TD,TA> smoother;
    typedef typename smoother::range_type range_type;

    static void apply(smoother& sm, range_type& v, const range_type& b)
    {
      sm.template apply<true>(v, b);
      sm.template apply<false>(v, b);
    }
  };

  template<class T, bool tag>
  struct SeqOverlappingSchwarzAssemblerHelper
  {};

  template<class T>
  using SeqOverlappingSchwarzAssembler = SeqOverlappingSchwarzAssemblerHelper<T,Dune::StoresColumnCompressed<T>::value>;

  template<class K, class Al, class X, class Y>
  struct SeqOverlappingSchwarzAssemblerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  {
    typedef BCRSMatrix< K, Al> matrix_type;
    static constexpr size_t n = std::decay_t<decltype(Impl::asMatrix(std::declval<K>()))>::rows;
    template<class RowToDomain, class Solvers, class SubDomains>
    static std::size_t assembleLocalProblems(const RowToDomain& rowToDomain, const matrix_type& mat,
                                             Solvers& solvers, const SubDomains& domains,
                                             bool onTheFly);
  };

  template<template<class> class S, typename T, typename A>
  struct SeqOverlappingSchwarzAssemblerHelper<S<BCRSMatrix<T,A>>,true>
  {
    typedef BCRSMatrix<T,A> matrix_type;
    static constexpr size_t n = std::decay_t<decltype(Impl::asMatrix(std::declval<T>()))>::rows;
    template<class RowToDomain, class Solvers, class SubDomains>
    static std::size_t assembleLocalProblems(const RowToDomain& rowToDomain, const matrix_type& mat,
                                             Solvers& solvers, const SubDomains& domains,
                                             bool onTheFly);
  };

  template<class M,class X, class Y>
  struct SeqOverlappingSchwarzAssemblerILUBase
  {
    typedef M matrix_type;
    template<class RowToDomain, class Solvers, class SubDomains>
    static std::size_t assembleLocalProblems(const RowToDomain& rowToDomain, const matrix_type& mat,
                                             Solvers& solvers, const SubDomains& domains,
                                             bool onTheFly);
  };

  template<class M,class X, class Y>
  struct SeqOverlappingSchwarzAssemblerHelper<ILU0SubdomainSolver<M,X,Y>,false>
    : public SeqOverlappingSchwarzAssemblerILUBase<M,X,Y>
  {};

  template<class M,class X, class Y>
  struct SeqOverlappingSchwarzAssemblerHelper<ILUNSubdomainSolver<M,X,Y>,false>
    : public SeqOverlappingSchwarzAssemblerILUBase<M,X,Y>
  {};

  /**
   * @brief Sequential overlapping Schwarz preconditioner
   *
   * @tparam M The matrix type.
   * @tparam X The range and domain type.
   * @tparam TM The Schwarz mode. Currently supported modes are AdditiveSchwarzMode,
   * MultiplicativeSchwarzMode, and SymmetricMultiplicativeSchwarzMode. (Default values is AdditiveSchwarzMode)
   * @tparam TD The type of the local subdomain solver to be used.
   * @tparam TA The type of the allocator to use.
   */
  template<class M, class X, class TM=AdditiveSchwarzMode,
      class TD=ILU0SubdomainSolver<M,X,X>, class TA=std::allocator<X> >
  class SeqOverlappingSchwarz
    : public Preconditioner<X,X>
  {
  public:
    /**
     * @brief The type of the matrix to precondition.
     */
    typedef M matrix_type;

    /**
     * @brief The domain type of the preconditioner
     */
    typedef X domain_type;

    /**
     * @brief The range type of the preconditioner.
     */
    typedef X range_type;

    /**
     * @brief The mode (additive or multiplicative) of the Schwarz
     * method.
     *
     * Either AdditiveSchwarzMode or MultiplicativeSchwarzMode
     */
    typedef TM Mode;

    /**
     * @brief The field type of the preconditioner.
     */
    typedef typename X::field_type field_type;

    /** @brief The return type of the size method. */
    typedef typename matrix_type::size_type size_type;

    /** @brief The allocator to use. */
    typedef TA allocator;

    /** @brief The type for the subdomain to row index mapping. */
    typedef std::set<size_type, std::less<size_type>,
        typename std::allocator_traits<TA>::template rebind_alloc<size_type> >
    subdomain_type;

    /** @brief The vector type containing the subdomain to row index mapping. */
    typedef std::vector<subdomain_type, typename std::allocator_traits<TA>::template rebind_alloc<subdomain_type> > subdomain_vector;

    /** @brief The type for the row to subdomain mapping. */
    typedef SLList<size_type, typename std::allocator_traits<TA>::template rebind_alloc<size_type> > subdomain_list;

    /** @brief The vector type containing the row index to subdomain mapping. */
    typedef std::vector<subdomain_list, typename std::allocator_traits<TA>::template rebind_alloc<subdomain_list> > rowtodomain_vector;

    /** @brief The type for the subdomain solver in use. */
    typedef TD slu;

    /** @brief The vector type containing subdomain solvers. */
    typedef std::vector<slu, typename std::allocator_traits<TA>::template rebind_alloc<slu> > slu_vector;

    /**
     * @brief Construct the overlapping Schwarz method.
     * @param mat The matrix to precondition.
     * @param subDomains Array of sets of rowindices belonging to an overlapping
     * subdomain
     * @param relaxationFactor relaxation factor
     * @param onTheFly_ If true the decomposition of the exact local solvers is
     * computed on the fly for each subdomain and
     * iteration step. If false all decompositions are computed in pre and
     * only forward and backward substitution takes place
     * in the iteration steps.
     * @warning Each rowindex should be part of at least one subdomain!
     */
    SeqOverlappingSchwarz(const matrix_type& mat, const subdomain_vector& subDomains,
                          field_type relaxationFactor=1, bool onTheFly_=true);

    /**
     * Construct the overlapping Schwarz method
     * @param mat The matrix to precondition.
     * @param rowToDomain The mapping of the rows onto the domains.
     * @param relaxationFactor relaxation factor
     * @param onTheFly_ If true the decomposition of the exact local solvers is
     * computed on the fly for each subdomain and
     * iteration step. If false all decompositions are computed in pre and
     * only forward and backward substitution takes place
     * in the iteration steps.
     */
    SeqOverlappingSchwarz(const matrix_type& mat, const rowtodomain_vector& rowToDomain,
                          field_type relaxationFactor=1, bool onTheFly_=true);

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre ([[maybe_unused]] X& x, [[maybe_unused]] X& b)
    {}

    /*!
       \brief Apply the precondtioner

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const X& d);

    /*!
       \brief Postprocess the preconditioner.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post ([[maybe_unused]] X& x)
    {}

    template<bool forward>
    void apply(X& v, const X& d);

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual SolverCategory::Category category() const
    {
      return SolverCategory::sequential;
    }

  private:
    const M& mat;
    slu_vector solvers;
    subdomain_vector subDomains;
    field_type relax;

    typename M::size_type maxlength;

    bool onTheFly;
  };



  template<class I, class S, class D>
  OverlappingSchwarzInitializer<I,S,D>::OverlappingSchwarzInitializer(InitializerList& il,
                                                                      const IndexSet& idx,
                                                                      const subdomain_vector& domains_)
    : initializers(&il), indices(&idx), indexMaps(il.size()), domains(domains_)
  {}


  template<class I, class S, class D>
  void OverlappingSchwarzInitializer<I,S,D>::addRowNnz(const Iter& row)
  {
    typedef typename IndexSet::value_type::const_iterator iterator;
    for(iterator domain=(*indices)[row.index()].begin(); domain != (*indices)[row.index()].end(); ++domain) {
      (*initializers)[*domain].addRowNnz(row, domains[*domain]);
      indexMaps[*domain].insert(row.index());
    }
  }

  template<class I, class S, class D>
  void OverlappingSchwarzInitializer<I,S,D>::allocate()
  {
    for(auto&& i: *initializers)
      i.allocateMatrixStorage();
    for(auto&& i: *initializers)
      i.allocateMarker();
  }

  template<class I, class S, class D>
  void OverlappingSchwarzInitializer<I,S,D>::countEntries(const Iter& row, const CIter& col) const
  {
    typedef typename IndexSet::value_type::const_iterator iterator;
    for(iterator domain=(*indices)[row.index()].begin(); domain != (*indices)[row.index()].end(); ++domain) {
      typename std::map<size_type,size_type>::const_iterator v = indexMaps[*domain].find(col.index());
      if(v!= indexMaps[*domain].end()) {
        (*initializers)[*domain].countEntries(indexMaps[*domain].find(col.index())->second);
      }
    }
  }

  template<class I, class S, class D>
  void OverlappingSchwarzInitializer<I,S,D>::calcColstart() const
  {
    for(auto&& i : *initializers)
      i.calcColstart();
  }

  template<class I, class S, class D>
  void OverlappingSchwarzInitializer<I,S,D>::copyValue(const Iter& row, const CIter& col) const
  {
    typedef typename IndexSet::value_type::const_iterator iterator;
    for(iterator domain=(*indices)[row.index()].begin(); domain!= (*indices)[row.index()].end(); ++domain) {
      typename std::map<size_type,size_type>::const_iterator v = indexMaps[*domain].find(col.index());
      if(v!= indexMaps[*domain].end()) {
        assert(indexMaps[*domain].end()!=indexMaps[*domain].find(row.index()));
        (*initializers)[*domain].copyValue(col, indexMaps[*domain].find(row.index())->second,
                                           v->second);
      }
    }
  }

  template<class I, class S, class D>
  void OverlappingSchwarzInitializer<I,S,D>::createMatrix() const
  {
    std::vector<IndexMap>().swap(indexMaps);
    for(auto&& i: *initializers)
      i.createMatrix();
  }

  template<class I, class S, class D>
  OverlappingSchwarzInitializer<I,S,D>::IndexMap::IndexMap()
    : row(0)
  {}

  template<class I, class S, class D>
  void OverlappingSchwarzInitializer<I,S,D>::IndexMap::insert(size_type grow)
  {
    assert(map_.find(grow)==map_.end());
    map_.insert(std::make_pair(grow, row++));
  }

  template<class I, class S, class D>
  typename OverlappingSchwarzInitializer<I,S,D>::IndexMap::const_iterator
  OverlappingSchwarzInitializer<I,S,D>::IndexMap::find(size_type grow) const
  {
    return map_.find(grow);
  }

  template<class I, class S, class D>
  typename OverlappingSchwarzInitializer<I,S,D>::IndexMap::iterator
  OverlappingSchwarzInitializer<I,S,D>::IndexMap::find(size_type grow)
  {
    return map_.find(grow);
  }

  template<class I, class S, class D>
  typename OverlappingSchwarzInitializer<I,S,D>::IndexMap::const_iterator
  OverlappingSchwarzInitializer<I,S,D>::IndexMap::end() const
  {
    return map_.end();
  }

  template<class I, class S, class D>
  typename OverlappingSchwarzInitializer<I,S,D>::IndexMap::iterator
  OverlappingSchwarzInitializer<I,S,D>::IndexMap::end()
  {
    return map_.end();
  }

  template<class I, class S, class D>
  typename OverlappingSchwarzInitializer<I,S,D>::IndexMap::const_iterator
  OverlappingSchwarzInitializer<I,S,D>::IndexMap::begin() const
  {
    return map_.begin();
  }

  template<class I, class S, class D>
  typename OverlappingSchwarzInitializer<I,S,D>::IndexMap::iterator
  OverlappingSchwarzInitializer<I,S,D>::IndexMap::begin()
  {
    return map_.begin();
  }

  template<class M, class X, class TM, class TD, class TA>
  SeqOverlappingSchwarz<M,X,TM,TD,TA>::SeqOverlappingSchwarz(const matrix_type& mat_, const rowtodomain_vector& rowToDomain,
                                                             field_type relaxationFactor, bool fly)
    : mat(mat_), relax(relaxationFactor), onTheFly(fly)
  {
    typedef typename rowtodomain_vector::const_iterator RowDomainIterator;
    typedef typename subdomain_list::const_iterator DomainIterator;
#ifdef DUNE_ISTL_WITH_CHECKING
    assert(rowToDomain.size()==mat.N());
    assert(rowToDomain.size()==mat.M());

    for(RowDomainIterator iter=rowToDomain.begin(); iter != rowToDomain.end(); ++iter)
      assert(iter->size()>0);

#endif
    // calculate the number of domains
    size_type domains=0;
    for(RowDomainIterator iter=rowToDomain.begin(); iter != rowToDomain.end(); ++iter)
      for(DomainIterator d=iter->begin(); d != iter->end(); ++d)
        domains=std::max(domains, *d);
    ++domains;

    solvers.resize(domains);
    subDomains.resize(domains);

    // initialize subdomains to row mapping from row to subdomain mapping
    size_type row=0;
    for(RowDomainIterator iter=rowToDomain.begin(); iter != rowToDomain.end(); ++iter, ++row)
      for(DomainIterator d=iter->begin(); d != iter->end(); ++d)
        subDomains[*d].insert(row);

#ifdef DUNE_ISTL_WITH_CHECKING
    size_type i=0;
    typedef typename subdomain_vector::const_iterator iterator;
    for(iterator iter=subDomains.begin(); iter != subDomains.end(); ++iter) {
      typedef typename subdomain_type::const_iterator entry_iterator;
      Dune::dvverb<<"domain "<<i++<<":";
      for(entry_iterator entry = iter->begin(); entry != iter->end(); ++entry) {
        Dune::dvverb<<" "<<*entry;
      }
      Dune::dvverb<<std::endl;
    }
#endif
    maxlength = SeqOverlappingSchwarzAssembler<slu>
                ::assembleLocalProblems(rowToDomain, mat, solvers, subDomains, onTheFly);
  }

  template<class M, class X, class TM, class TD, class TA>
  SeqOverlappingSchwarz<M,X,TM,TD,TA>::SeqOverlappingSchwarz(const matrix_type& mat_,
                                                             const subdomain_vector& sd,
                                                             field_type relaxationFactor,
                                                             bool fly)
    :  mat(mat_), solvers(sd.size()), subDomains(sd), relax(relaxationFactor),
      onTheFly(fly)
  {
    typedef typename subdomain_vector::const_iterator DomainIterator;

#ifdef DUNE_ISTL_WITH_CHECKING
    size_type i=0;

    for(DomainIterator d=sd.begin(); d != sd.end(); ++d,++i) {
      //std::cout<<i<<": "<<d->size()<<std::endl;
      assert(d->size()>0);
      typedef typename DomainIterator::value_type::const_iterator entry_iterator;
      Dune::dvverb<<"domain "<<i<<":";
      for(entry_iterator entry = d->begin(); entry != d->end(); ++entry) {
        Dune::dvverb<<" "<<*entry;
      }
      Dune::dvverb<<std::endl;
    }

#endif

    // Create a row to subdomain mapping
    rowtodomain_vector rowToDomain(mat.N());

    size_type domainId=0;

    for(DomainIterator domain=sd.begin(); domain != sd.end(); ++domain, ++domainId) {
      typedef typename subdomain_type::const_iterator iterator;
      for(iterator row=domain->begin(); row != domain->end(); ++row)
        rowToDomain[*row].push_back(domainId);
    }

    maxlength = SeqOverlappingSchwarzAssembler<slu>
                ::assembleLocalProblems(rowToDomain, mat, solvers, subDomains, onTheFly);
  }

  /**
     template helper struct to determine the size of a domain for the
     SeqOverlappingSchwarz solver

     only implemented for BCRSMatrix<T>
   */
  template<class M>
  struct SeqOverlappingSchwarzDomainSize {};

  template<typename T, typename A>
  struct SeqOverlappingSchwarzDomainSize<BCRSMatrix<T,A > >
  {
    static constexpr size_t n = std::decay_t<decltype(Impl::asMatrix(std::declval<T>()))>::rows;
    static constexpr size_t m = std::decay_t<decltype(Impl::asMatrix(std::declval<T>()))>::cols;
    template<class Domain>
    static int size(const Domain & d)
    {
      assert(n==m);
      return m*d.size();
    }
  };

  template<class K, class Al, class X, class Y>
  template<class RowToDomain, class Solvers, class SubDomains>
  std::size_t
  SeqOverlappingSchwarzAssemblerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>::
  assembleLocalProblems([[maybe_unused]] const RowToDomain& rowToDomain,
                        [[maybe_unused]] const matrix_type& mat,
                        [[maybe_unused]] Solvers& solvers,
                        const SubDomains& subDomains,
                        [[maybe_unused]] bool onTheFly)
  {
    typedef typename SubDomains::const_iterator DomainIterator;
    std::size_t maxlength = 0;

    assert(onTheFly);

    for(DomainIterator domain=subDomains.begin(); domain!=subDomains.end(); ++domain)
      maxlength=std::max(maxlength, domain->size());
    maxlength*=n;

    return maxlength;
  }

#if HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK
  template<template<class> class S, typename T, typename A>
  template<class RowToDomain, class Solvers, class SubDomains>
  std::size_t SeqOverlappingSchwarzAssemblerHelper<S<BCRSMatrix<T,A>>,true>::assembleLocalProblems(const RowToDomain& rowToDomain,
                                                                                                   const matrix_type& mat,
                                                                                                   Solvers& solvers,
                                                                                                   const SubDomains& subDomains,
                                                                                                   bool onTheFly)
  {
    typedef typename S<BCRSMatrix<T,A>>::MatrixInitializer MatrixInitializer;
    typedef typename std::vector<MatrixInitializer>::iterator InitializerIterator;
    typedef typename SubDomains::const_iterator DomainIterator;
    typedef typename Solvers::iterator SolverIterator;
    std::size_t maxlength = 0;

    if(onTheFly) {
      for(DomainIterator domain=subDomains.begin(); domain!=subDomains.end(); ++domain)
        maxlength=std::max(maxlength, domain->size());
      maxlength*=Impl::asMatrix(*mat[0].begin()).N();
    }else{
      // initialize the initializers
      DomainIterator domain=subDomains.begin();

      // Create the initializers list.
      std::vector<MatrixInitializer> initializers(subDomains.size());

      SolverIterator solver=solvers.begin();
      for(InitializerIterator initializer=initializers.begin(); initializer!=initializers.end();
          ++initializer, ++solver, ++domain) {
        solver->getInternalMatrix().N_=SeqOverlappingSchwarzDomainSize<matrix_type>::size(*domain);
        solver->getInternalMatrix().M_=SeqOverlappingSchwarzDomainSize<matrix_type>::size(*domain);
        //solver->setVerbosity(true);
        *initializer=MatrixInitializer(solver->getInternalMatrix());
      }

      // Set up the supermatrices according to the subdomains
      typedef OverlappingSchwarzInitializer<std::vector<MatrixInitializer>,
          RowToDomain, SubDomains> Initializer;

      Initializer initializer(initializers, rowToDomain, subDomains);
      copyToBCCSMatrix(initializer, mat);

      // Calculate the LU decompositions
      for(auto&& s: solvers)
        s.decompose();
      for (SolverIterator solverIt = solvers.begin(); solverIt != solvers.end(); ++solverIt)
      {
        assert(solverIt->getInternalMatrix().N() == solverIt->getInternalMatrix().M());
        maxlength = std::max(maxlength, solverIt->getInternalMatrix().N());
      }
    }
    return maxlength;
  }

#endif // HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK

  template<class M,class X,class Y>
  template<class RowToDomain, class Solvers, class SubDomains>
  std::size_t SeqOverlappingSchwarzAssemblerILUBase<M,X,Y>::assembleLocalProblems([[maybe_unused]] const RowToDomain& rowToDomain,
                                                                                  const matrix_type& mat,
                                                                                  Solvers& solvers,
                                                                                  const SubDomains& subDomains,
                                                                                  bool onTheFly)
  {
    typedef typename SubDomains::const_iterator DomainIterator;
    typedef typename Solvers::iterator SolverIterator;
    std::size_t maxlength = 0;

    if(onTheFly) {
      for(DomainIterator domain=subDomains.begin(); domain!=subDomains.end(); ++domain)
        maxlength=std::max(maxlength, domain->size());
    }else{
      // initialize the solvers of the local prolems.
      SolverIterator solver=solvers.begin();
      for(DomainIterator domain=subDomains.begin(); domain!=subDomains.end();
          ++domain, ++solver) {
        solver->setSubMatrix(mat, *domain);
        maxlength=std::max(maxlength, domain->size());
      }
    }

    return maxlength;

  }


  template<class M, class X, class TM, class TD, class TA>
  void SeqOverlappingSchwarz<M,X,TM,TD,TA>::apply(X& x, const X& b)
  {
    SeqOverlappingSchwarzApplier<SeqOverlappingSchwarz>::apply(*this, x, b);
  }

  template<class M, class X, class TM, class TD, class TA>
  template<bool forward>
  void SeqOverlappingSchwarz<M,X,TM,TD,TA>::apply(X& x, const X& b)
  {
    typedef slu_vector solver_vector;
    typedef typename IteratorDirectionSelector<solver_vector,subdomain_vector,forward>::solver_iterator iterator;
    typedef typename IteratorDirectionSelector<solver_vector,subdomain_vector,forward>::domain_iterator
    domain_iterator;

    OverlappingAssigner<TD> assigner(maxlength, mat, b, x);

    domain_iterator domain=IteratorDirectionSelector<solver_vector,subdomain_vector,forward>::begin(subDomains);
    iterator solver = IteratorDirectionSelector<solver_vector,subdomain_vector,forward>::begin(solvers);
    X v(x); // temporary for the update
    v=0;

    typedef typename AdderSelector<TM,X,TD >::Adder Adder;
    Adder adder(v, x, assigner, relax);

    for(; domain != IteratorDirectionSelector<solver_vector,subdomain_vector,forward>::end(subDomains); ++domain) {
      //Copy rhs to C-array for SuperLU
      std::for_each(domain->begin(), domain->end(), assigner);
      assigner.resetIndexForNextDomain();
      if(onTheFly) {
        // Create the subdomain solver
        slu sdsolver;
        sdsolver.setSubMatrix(mat, *domain);
        // Apply
        sdsolver.apply(assigner.lhs(), assigner.rhs());
      }else{
        solver->apply(assigner.lhs(), assigner.rhs());
        ++solver;
      }

      //Add relaxed correction to from SuperLU to v
      std::for_each(domain->begin(), domain->end(), adder);
      assigner.resetIndexForNextDomain();

    }

    adder.axpy();
    assigner.deallocate();
  }

  template<class K, class Al, class X, class Y>
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::OverlappingAssignerHelper(std::size_t maxlength, const BCRSMatrix<K, Al>& mat_,
                        const X& b_, Y& x_) :
    mat(&mat_),
    rhs_( new DynamicVector<field_type>(maxlength, 42) ),
    lhs_( new DynamicVector<field_type>(maxlength, -42) ),
    b(&b_),
    x(&x_),
    i(0),
    maxlength_(maxlength)
  {}

  template<class K, class Al, class X, class Y>
  void
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::deallocate()
  {
    delete rhs_;
    delete lhs_;
  }

  template<class K, class Al, class X, class Y>
  void
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::resetIndexForNextDomain()
  {
    i=0;
  }

  template<class K, class Al, class X, class Y>
  DynamicVector<typename X::field_type> &
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::lhs()
  {
    return *lhs_;
  }

  template<class K, class Al, class X, class Y>
  DynamicVector<typename X::field_type> &
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::rhs()
  {
    return *rhs_;
  }

  template<class K, class Al, class X, class Y>
  void
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::relaxResult(field_type relax)
  {
    lhs() *= relax;
  }

  template<class K, class Al, class X, class Y>
  void
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::operator()(const size_type& domainIndex)
  {
    lhs() = 0.0;
#if 0
    //assign right hand side of current domainindex block
    for(size_type j=0; j<n; ++j, ++i) {
      assert(i<maxlength_);
      rhs()[i]=(*b)[domainIndex][j];
    }

    // loop over all Matrix row entries and calculate defect.
    typedef typename matrix_type::ConstColIterator col_iterator;

    // calculate defect for current row index block
    for(col_iterator col=(*mat)[domainIndex].begin(); col!=(*mat)[domainIndex].end(); ++col) {
      block_type tmp(0.0);
      (*col).mv((*x)[col.index()], tmp);
      i-=n;
      for(size_type j=0; j<n; ++j, ++i) {
        assert(i<maxlength_);
        rhs()[i]-=tmp[j];
      }
    }
#else
    //assign right hand side of current domainindex block
    for(size_type j=0; j<n; ++j, ++i) {
      assert(i<maxlength_);
      rhs()[i]=Impl::asVector((*b)[domainIndex])[j];

      // loop over all Matrix row entries and calculate defect.
      typedef typename matrix_type::ConstColIterator col_iterator;

      // calculate defect for current row index block
      for(col_iterator col=(*mat)[domainIndex].begin(); col!=(*mat)[domainIndex].end(); ++col) {
        for(size_type k=0; k<n; ++k) {
          rhs()[i]-=Impl::asMatrix(*col)[j][k] * Impl::asVector((*x)[col.index()])[k];
        }
      }
    }
#endif
  }

  template<class K, class Al, class X, class Y>
  void
  OverlappingAssignerHelper< DynamicMatrixSubdomainSolver< BCRSMatrix< K, Al>, X, Y >,false>
  ::assignResult(block_type& res)
  {
    // assign the result of the local solve to the global vector
    for(size_type j=0; j<n; ++j, ++i) {
      assert(i<maxlength_);
      Impl::asVector(res)[j]+=lhs()[i];
    }
  }

#if HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK

  template<template<class> class S, typename T, typename A>
  OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>
  ::OverlappingAssignerHelper(std::size_t maxlength,
                        const BCRSMatrix<T,A>& mat_,
                        const range_type& b_,
                        range_type& x_)
    : mat(&mat_),
      b(&b_),
      x(&x_), i(0), maxlength_(maxlength)
  {
    rhs_ = new field_type[maxlength];
    lhs_ = new field_type[maxlength];

  }

  template<template<class> class S, typename T, typename A>
  void OverlappingAssignerHelper<S<BCRSMatrix<T,A> >,true>::deallocate()
  {
    delete[] rhs_;
    delete[] lhs_;
  }

  template<template<class> class S, typename T, typename A>
  void OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::operator()(const size_type& domainIndex)
  {
    //assign right hand side of current domainindex block
    // rhs is an array of doubles!
    // rhs[starti] = b[domainindex]
    for(size_type j=0; j<n; ++j, ++i) {
      assert(i<maxlength_);
      rhs_[i]=Impl::asVector((*b)[domainIndex])[j];
    }


    // loop over all Matrix row entries and calculate defect.
    typedef typename matrix_type::ConstColIterator col_iterator;

    // calculate defect for current row index block
    for(col_iterator col=(*mat)[domainIndex].begin(); col!=(*mat)[domainIndex].end(); ++col) {
      block_type tmp;
      Impl::asMatrix(*col).mv((*x)[col.index()], tmp);
      i-=n;
      for(size_type j=0; j<n; ++j, ++i) {
        assert(i<maxlength_);
        rhs_[i]-=Impl::asVector(tmp)[j];
      }

    }

  }

  template<template<class> class S, typename T, typename A>
  void OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::relaxResult(field_type relax)
  {
    for(size_type j=i+n; i<j; ++i) {
      assert(i<maxlength_);
      lhs_[i]*=relax;
    }
    i-=n;
  }

  template<template<class> class S, typename T, typename A>
  void OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::assignResult(block_type& res)
  {
    // assign the result of the local solve to the global vector
    for(size_type j=0; j<n; ++j, ++i) {
      assert(i<maxlength_);
      Impl::asVector(res)[j]+=lhs_[i];
    }
  }

  template<template<class> class S, typename T, typename A>
  void OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::resetIndexForNextDomain()
  {
    i=0;
  }

  template<template<class> class S, typename T, typename A>
  typename OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::field_type*
  OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::lhs()
  {
    return lhs_;
  }

  template<template<class> class S, typename T, typename A>
  typename OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::field_type*
  OverlappingAssignerHelper<S<BCRSMatrix<T,A>>,true>::rhs()
  {
    return rhs_;
  }

#endif // HAVE_SUPERLU || HAVE_SUITESPARSE_UMFPACK

  template<class M, class X, class Y>
  OverlappingAssignerILUBase<M,X,Y>::OverlappingAssignerILUBase(std::size_t maxlength,
                                                                const M& mat_,
                                                                const Y& b_,
                                                                X& x_)
    : mat(&mat_),
      b(&b_),
      x(&x_), i(0)
  {
    rhs_= new Y(maxlength);
    lhs_ = new X(maxlength);
  }

  template<class M, class X, class Y>
  void OverlappingAssignerILUBase<M,X,Y>::deallocate()
  {
    delete rhs_;
    delete lhs_;
  }

  template<class M, class X, class Y>
  void OverlappingAssignerILUBase<M,X,Y>::operator()(const size_type& domainIndex)
  {
    (*rhs_)[i]=(*b)[domainIndex];

    // loop over all Matrix row entries and calculate defect.
    typedef typename matrix_type::ConstColIterator col_iterator;

    // calculate defect for current row index block
    for(col_iterator col=(*mat)[domainIndex].begin(); col!=(*mat)[domainIndex].end(); ++col) {
      Impl::asMatrix(*col).mmv((*x)[col.index()], (*rhs_)[i]);
    }
    // Goto next local index
    ++i;
  }

  template<class M, class X, class Y>
  void OverlappingAssignerILUBase<M,X,Y>::relaxResult(field_type relax)
  {
    (*lhs_)[i]*=relax;
  }

  template<class M, class X, class Y>
  void OverlappingAssignerILUBase<M,X,Y>::assignResult(block_type& res)
  {
    res+=(*lhs_)[i++];
  }

  template<class M, class X, class Y>
  X& OverlappingAssignerILUBase<M,X,Y>::lhs()
  {
    return *lhs_;
  }

  template<class M, class X, class Y>
  Y& OverlappingAssignerILUBase<M,X,Y>::rhs()
  {
    return *rhs_;
  }

  template<class M, class X, class Y>
  void OverlappingAssignerILUBase<M,X,Y>::resetIndexForNextDomain()
  {
    i=0;
  }

  template<typename S, typename T, typename A>
  AdditiveAdder<S,BlockVector<T,A> >::AdditiveAdder(BlockVector<T,A>& v_,
                                                    BlockVector<T,A>& x_,
                                                    OverlappingAssigner<S>& assigner_,
                                                    const field_type& relax_)
    : v(&v_), x(&x_), assigner(&assigner_), relax(relax_)
  {}

  template<typename S, typename T, typename A>
  void AdditiveAdder<S,BlockVector<T,A> >::operator()(const size_type& domainIndex)
  {
    // add the result of the local solve to the current update
    assigner->assignResult((*v)[domainIndex]);
  }


  template<typename S, typename T, typename A>
  void AdditiveAdder<S,BlockVector<T,A> >::axpy()
  {
    // relax the update and add it to the current guess.
    x->axpy(relax,*v);
  }


  template<typename S, typename T, typename A>
  MultiplicativeAdder<S,BlockVector<T,A> >
  ::MultiplicativeAdder([[maybe_unused]] BlockVector<T,A>& v_,
                        BlockVector<T,A>& x_,
                        OverlappingAssigner<S>& assigner_, const field_type& relax_)
    : x(&x_), assigner(&assigner_), relax(relax_)
  {}


  template<typename S,typename T, typename A>
  void MultiplicativeAdder<S,BlockVector<T,A> >::operator()(const size_type& domainIndex)
  {
    // add the result of the local solve to the current guess
    assigner->relaxResult(relax);
    assigner->assignResult((*x)[domainIndex]);
  }


  template<typename S,typename T, typename A>
  void MultiplicativeAdder<S,BlockVector<T,A> >::axpy()
  {
    // nothing to do, as the corrections already relaxed and added in operator()
  }


  /** @} */
}

#endif
