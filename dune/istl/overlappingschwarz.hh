// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_OVERLAPPINGSCHWARZ_HH
#define DUNE_OVERLAPPINGSCHWARZ_HH
#include <cassert>
#include <algorithm>
#include <functional>
#include <vector>
#include <set>
#include <dune/common/sllist.hh>
#include "preconditioners.hh"
#include "superlu.hh"
#include "bvector.hh"
#include "bcrsmatrix.hh"
#include "ilusubdomainsolver.hh"

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
#if HAVE_SUPERLU

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
   * @brief Tag that the tells the schwarz method to be additive.
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



  template<typename T>
  struct OverlappingAssigner
  {};

  template<typename T, typename A, int n, int m>
  struct OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >
  {
    typedef BCRSMatrix<FieldMatrix<T,n,m>,A> matrix_type;
    typedef typename SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> >::range_type range_type;
    typedef typename range_type::field_type field_type;
    typedef typename range_type::block_type block_type;

    typedef typename matrix_type::size_type size_type;

    /**
     * @brief Constructor.
     * @param mat The global matrix.
     * @param rhs storage for the local defect.
     * @param b the global right hand side.
     * @param x the global left hand side.
     */
    OverlappingAssigner(std::size_t maxlength, const BCRSMatrix<FieldMatrix<T,n,m>,A>& mat,
                        const range_type& b, range_type& x);

    void deallocate();

    void resetIndexForNextDomain();

    field_type* lhs();

    field_type* rhs();

    void relaxResult(field_type relax);

    /**
     * @brief calculate one entry of the local defect.
     * @param domain One index of the domain.
     */
    void operator()(const size_type& domain);

    void assignResult(block_type& res);

  private:
    const matrix_type* mat;
    field_type* rhs_;
    field_type* lhs_;
    const range_type* b;
    range_type* x;
    std::size_t i;
    std::size_t maxlength_;
  };

  template<class M, class X, class Y>
  class OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >
  {
  public:
    typedef M matrix_type;

    typedef typename M::field_type field_type;

    typedef typename Y::block_type block_type;

    typedef typename matrix_type::size_type size_type;
    /**
     * @brief Constructor.
     * @param mat The global matrix.
     * @param rhs storage for the local defect.
     * @param b the global right hand side.
     * @param x the global left hand side.
     */
    OverlappingAssigner(std::size_t maxlength, const M& mat,
                        const Y& b, X& x);

    void deallocate();


    void resetIndexForNextDomain();

    X& lhs();

    Y& rhs();

    void relaxResult(field_type relax);

    /**
     * @brief calculate one entry of the local defect.
     * @param domain One index of the domain.
     */
    void operator()(const size_type& domain);

    void assignResult(block_type& res);
  private:
    const M* mat;
    X* lhs_;
    Y* rhs_;
    const Y* b;
    X* x;
    size_type i;
  };


  template<typename S, typename T>
  struct AdditiveAdder
  {};

  template<typename S, typename T, typename A, int n>
  struct AdditiveAdder<S, BlockVector<FieldVector<T,n>,A> >
  {
    typedef typename A::size_type size_type;
    AdditiveAdder(BlockVector<FieldVector<T,n>,A>& v, BlockVector<FieldVector<T,n>,A>& x,
                  OverlappingAssigner<S>& assigner, const T& relax_);
    void operator()(const size_type& domain);
    void axpy();

  private:
    BlockVector<FieldVector<T,n>,A>* v;
    BlockVector<FieldVector<T,n>,A>* x;
    OverlappingAssigner<S>* assigner;
    T relax;
  };

  template<typename S,typename T>
  struct MultiplicativeAdder
  {};

  template<typename S, typename T, typename A, int n>
  struct MultiplicativeAdder<S, BlockVector<FieldVector<T,n>,A> >
  {
    typedef typename A::size_type size_type;
    MultiplicativeAdder(BlockVector<FieldVector<T,n>,A>& v, BlockVector<FieldVector<T,n>,A>& x,
                        OverlappingAssigner<S>& assigner_, const T& relax_);
    void operator()(const size_type& domain);
    void axpy();

  private:
    BlockVector<FieldVector<T,n>,A>* v;
    BlockVector<FieldVector<T,n>,A>* x;
    OverlappingAssigner<S>* assigner;
    T relax;
  };

  /**
   * @brief template meta program for choosing  how to add the correction.
   *
   * There are specialization for the additive, the multiplicative, and the symmetric multiplicative mode.
   *
   * \tparam The Schwarz mode (either AdditiveSchwarzMode or MuliplicativeSchwarzMode or
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
   * @brief Helper template meta program for application of overlapping schwarz.
   *
   * The is needed because when using the multiplicative schwarz version one
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
   * @brief Helper template meta program for application of overlapping schwarz.
   *
   * The is needed because when using the multiplicative schwarz version one
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

  template<class T>
  struct SeqOverlappingSchwarzAssembler
  {};

  template<class T>
  struct SeqOverlappingSchwarzAssembler<SuperLU<T> >
  {
    typedef T matrix_type;
    template<class RowToDomain, class Solvers, class SubDomains>
    static std::size_t assembleLocalProblems(const RowToDomain& rowToDomain, const matrix_type& mat,
                                             Solvers& solvers, const SubDomains& domains,
                                             bool onTheFly);
  };

  template<class M,class X, class Y>
  struct SeqOverlappingSchwarzAssembler<ILU0SubdomainSolver<M,X,Y> >
  {
    typedef M matrix_type;
    template<class RowToDomain, class Solvers, class SubDomains>
    static std::size_t assembleLocalProblems(const RowToDomain& rowToDomain, const matrix_type& mat,
                                             Solvers& solvers, const SubDomains& domains,
                                             bool onTheFly);
  };

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
  template<class M, class X, class TM=AdditiveSchwarzMode, class TD=SuperLU<M>, class TA=std::allocator<X> >
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
        typename TA::template rebind<size_type>::other>
    subdomain_type;

    /** @brief The vector type containing the subdomain to row index mapping. */
    typedef std::vector<subdomain_type, typename TA::template rebind<subdomain_type>::other> subdomain_vector;

    /** @brief The type for the row to subdomain mapping. */
    typedef SLList<size_type, typename TA::template rebind<size_type>::other> subdomain_list;

    /** @brief The vector type containing the row index to subdomain mapping. */
    typedef std::vector<subdomain_list, typename TA::template rebind<subdomain_list>::other > rowtodomain_vector;

    /** @brief The type for the subdomain solver in use. */
    typedef TD slu;

    /** @brief The vector type containing subdomain solvers. */
    typedef std::vector<slu, typename TA::template rebind<slu>::other> slu_vector;

    enum {
      //! \brief The category the precondtioner is part of.
      category = SolverCategory::sequential
    };

    /**
     * @brief Construct the overlapping Schwarz method.
     * @param mat The matrix to precondition.
     * @param subDomains Array of sets of rowindices belonging to an overlapping
     * subdomain
     * @param relaxationFactor relaxation factor
     * @param onTheFly If true the decomposition of the exact local solvers is
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
     * @param onTheFly If true the decomposition of the exact local solvers is
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
    virtual void pre (X& x, X& b) {}

    /*!
       \brief Apply the precondtioner

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const X& d);

    template<bool forward>
    void apply(X& v, const X& d);

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x) {
      std::cout<<" avg nnz over subdomain is "<<nnz<<std::endl;
    }

  private:
    const M& mat;
    slu_vector solvers;
    subdomain_vector subDomains;
    field_type relax;

    typename M::size_type maxlength;
    std::size_t nnz;

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
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::allocateMatrixStorage));
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::allocateMarker));
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
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::calcColstart));
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
    indexMaps.clear();
    indexMaps.swap(std::vector<IndexMap>(indexMaps));
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::createMatrix));
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
      std::cout<<"domain "<<i++<<":";
      for(entry_iterator entry = iter->begin(); entry != iter->end(); ++entry) {
        std::cout<<" "<<*entry;
      }
      std::cout<<std::endl;
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
      std::cout<<"domain "<<i<<":";
      for(entry_iterator entry = d->begin(); entry != d->end(); ++entry) {
        std::cout<<" "<<*entry;
      }
      std::cout<<std::endl;
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

     only implemented for BCRSMatrix<FieldMatrix<T,n,m>
   */
  template<class M>
  struct SeqOverlappingSchwarzDomainSize {};

  template<typename T, typename A, int n, int m>
  struct SeqOverlappingSchwarzDomainSize<BCRSMatrix<FieldMatrix<T,n,m>,A > >
  {
    template<class Domain>
    static int size(const Domain & d)
    {
      assert(n==m);
      return m*d.size();
    }
  };

  template<class T>
  template<class RowToDomain, class Solvers, class SubDomains>
  std::size_t SeqOverlappingSchwarzAssembler<SuperLU<T> >::assembleLocalProblems(const RowToDomain& rowToDomain,
                                                                                 const matrix_type& mat,
                                                                                 Solvers& solvers,
                                                                                 const SubDomains& subDomains,
                                                                                 bool onTheFly)
  {
    typedef typename std::vector<SuperMatrixInitializer<matrix_type> >::iterator InitializerIterator;
    typedef typename SubDomains::const_iterator DomainIterator;
    typedef typename Solvers::iterator SolverIterator;
    std::size_t maxlength = 0;

    if(onTheFly) {
      for(DomainIterator domain=subDomains.begin(); domain!=subDomains.end(); ++domain)
        maxlength=std::max(maxlength, domain->size());
      maxlength*=mat[0].begin()->N();
    }else{
      // initialize the initializers
      DomainIterator domain=subDomains.begin();

      // Create the initializers list.
      std::vector<SuperMatrixInitializer<matrix_type> > initializers(subDomains.size());

      SolverIterator solver=solvers.begin();
      for(InitializerIterator initializer=initializers.begin(); initializer!=initializers.end();
          ++initializer, ++solver, ++domain) {
        solver->mat.N_=SeqOverlappingSchwarzDomainSize<matrix_type>::size(*domain);
        solver->mat.M_=SeqOverlappingSchwarzDomainSize<matrix_type>::size(*domain);
        //solver->setVerbosity(true);
        *initializer=SuperMatrixInitializer<matrix_type>(solver->mat);
      }

      // Set up the supermatrices according to the subdomains
      typedef OverlappingSchwarzInitializer<std::vector<SuperMatrixInitializer<matrix_type> >,
          RowToDomain, SubDomains> Initializer;

      Initializer initializer(initializers, rowToDomain, subDomains);
      copyToSuperMatrix(initializer, mat);
      if(solvers.size()==1)
        assert(solvers[0].mat==mat);

      /*    for(SolverIterator solver=solvers.begin(); solver!=solvers.end(); ++solver)
            dPrint_CompCol_Matrix("superlu", &static_cast<SuperMatrix&>(solver->mat)); */

      // Calculate the LU decompositions
      std::for_each(solvers.begin(), solvers.end(), std::mem_fun_ref(&SuperLU<matrix_type>::decompose));
      for(SolverIterator solver=solvers.begin(); solver!=solvers.end(); ++solver) {
        assert(solver->mat.N()==solver->mat.M());
        maxlength=std::max(maxlength, solver->mat.N());
        //writeCompColMatrixToMatlab(static_cast<SuperLUMatrix<M>&>(solver->mat), std::cout);
      }
    }
    return maxlength;
  }

  template<class M,class X,class Y>
  template<class RowToDomain, class Solvers, class SubDomains>
  std::size_t SeqOverlappingSchwarzAssembler<ILU0SubdomainSolver<M,X,Y> >::assembleLocalProblems(const RowToDomain& rowToDomain,
                                                                                                 const matrix_type& mat,
                                                                                                 Solvers& solvers,
                                                                                                 const SubDomains& subDomains,
                                                                                                 bool onTheFly)
  {
    typedef typename std::vector<SuperMatrixInitializer<matrix_type> >::iterator InitializerIterator;
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
    typedef typename X::block_type block;
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

    nnz=0;
    std::size_t no=0;
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
        //nnz+=sdsolver.nnz();
      }else{
        solver->apply(assigner.lhs(), assigner.rhs());
        //nnz+=solver->nnz();
        ++solver;
      }
      ++no;
      //Add relaxed correction to from SuperLU to v
      std::for_each(domain->begin(), domain->end(), adder);
      assigner.resetIndexForNextDomain();

    }
    nnz/=no;

    adder.axpy();
    assigner.deallocate();
  }

  template<typename T, typename A, int n, int m>
  OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >
  ::OverlappingAssigner(std::size_t maxlength,
                        const BCRSMatrix<FieldMatrix<T,n,m>,A>& mat_,
                        const range_type& b_,
                        range_type& x_)
    : mat(&mat_),
      b(&b_),
      x(&x_), i(0), maxlength_(maxlength)
  {
    rhs_ = new field_type[maxlength];
    lhs_ = new field_type[maxlength];

  }

  template<typename T, typename A, int n, int m>
  void OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::deallocate()
  {
    delete[] rhs_;
    delete[] lhs_;
  }

  template<typename T, typename A, int n, int m>
  void OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::operator()(const size_type& domainIndex)
  {
    //assign right hand side of current domainindex block
    // rhs is an array of doubles!
    // rhs[starti] = b[domainindex]
    for(size_type j=0; j<n; ++j, ++i) {
      assert(i<maxlength_);
      rhs_[i]=(*b)[domainIndex][j];
    }


    // loop over all Matrix row entries and calculate defect.
    typedef typename matrix_type::ConstColIterator col_iterator;

    // calculate defect for current row index block
    for(col_iterator col=(*mat)[domainIndex].begin(); col!=(*mat)[domainIndex].end(); ++col) {
      block_type tmp;
      (*col).mv((*x)[col.index()], tmp);
      i-=n;
      for(size_type j=0; j<n; ++j, ++i) {
        assert(i<maxlength_);
        rhs_[i]-=tmp[j];
      }

    }

  }
  template<typename T, typename A, int n, int m>
  void OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::relaxResult(field_type relax)
  {
    for(size_type j=i+n; i<j; ++i) {
      assert(i<maxlength_);
      lhs_[i]*=relax;
    }
    i-=n;
  }

  template<typename T, typename A, int n, int m>
  void OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::assignResult(block_type& res)
  {
    // assign the result of the local solve to the global vector
    for(size_type j=0; j<n; ++j, ++i) {
      assert(i<maxlength_);
      res[j]+=lhs_[i];
    }
  }

  template<typename T, typename A, int n, int m>
  void OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::resetIndexForNextDomain()
  {
    i=0;
  }

  template<typename T, typename A, int n, int m>
  typename OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::field_type*
  OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::lhs()
  {
    return lhs_;
  }

  template<typename T, typename A, int n, int m>
  typename OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::field_type*
  OverlappingAssigner<SuperLU<BCRSMatrix<FieldMatrix<T,n,m>,A> > >::rhs()
  {
    return rhs_;
  }

  template<class M, class X, class Y>
  OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::OverlappingAssigner(std::size_t maxlength,
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
  void OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::deallocate()
  {
    delete rhs_;
    delete lhs_;
  }

  template<class M, class X, class Y>
  void OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::operator()(const size_type& domainIndex)
  {
    (*rhs_)[i]=(*b)[domainIndex];

    // loop over all Matrix row entries and calculate defect.
    typedef typename matrix_type::ConstColIterator col_iterator;

    // calculate defect for current row index block
    for(col_iterator col=(*mat)[domainIndex].begin(); col!=(*mat)[domainIndex].end(); ++col) {
      (*col).mmv((*x)[col.index()], (*rhs_)[i]);
    }
    // Goto next local index
    ++i;
  }

  template<class M, class X, class Y>
  void OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::relaxResult(field_type relax)
  {
    (*lhs_)[i]*=relax;
  }

  template<class M, class X, class Y>
  void OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::assignResult(block_type& res)
  {
    res=(*lhs_)[i++];
  }

  template<class M, class X, class Y>
  X& OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::lhs()
  {
    return *lhs_;
  }

  template<class M, class X, class Y>
  Y& OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::rhs()
  {
    return *rhs_;
  }

  template<class M, class X, class Y>
  void OverlappingAssigner<ILU0SubdomainSolver<M,X,Y> >::resetIndexForNextDomain()
  {
    i=0;
  }

  template<typename S, typename T, typename A, int n>
  AdditiveAdder<S,BlockVector<FieldVector<T,n>,A> >::AdditiveAdder(BlockVector<FieldVector<T,n>,A>& v_,
                                                                   BlockVector<FieldVector<T,n>,A>& x_,
                                                                   OverlappingAssigner<S>& assigner_,
                                                                   const T& relax_)
    : v(&v_), x(&x_), assigner(&assigner_), relax(relax_)
  {}

  template<typename S, typename T, typename A, int n>
  void AdditiveAdder<S,BlockVector<FieldVector<T,n>,A> >::operator()(const size_type& domainIndex)
  {
    // add the result of the local solve to the current update
    assigner->assignResult((*v)[domainIndex]);
  }


  template<typename S, typename T, typename A, int n>
  void AdditiveAdder<S,BlockVector<FieldVector<T,n>,A> >::axpy()
  {
    // relax the update and add it to the current guess.
    x->axpy(relax,*v);
  }


  template<typename S, typename T, typename A, int n>
  MultiplicativeAdder<S,BlockVector<FieldVector<T,n>,A> >
  ::MultiplicativeAdder(BlockVector<FieldVector<T,n>,A>& v_,
                        BlockVector<FieldVector<T,n>,A>& x_,
                        OverlappingAssigner<S>& assigner_, const T& relax_)
    : v(&v_), x(&x_), assigner(&assigner_), relax(relax_)
  {}


  template<typename S,typename T, typename A, int n>
  void MultiplicativeAdder<S,BlockVector<FieldVector<T,n>,A> >::operator()(const size_type& domainIndex)
  {
    // add the result of the local solve to the current guess
    assigner->relaxResult(relax);
    assigner->assignResult((*x)[domainIndex]);
  }


  template<typename S,typename T, typename A, int n>
  void MultiplicativeAdder<S,BlockVector<FieldVector<T,n>,A> >::axpy()
  {
    // nothing to do, as the corrections already relaxed and added in operator()
  }


  /** @} */
#endif
}

#endif
