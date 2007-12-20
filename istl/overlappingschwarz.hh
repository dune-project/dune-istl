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

namespace Dune
{

  /**
   * @addtogroup ISTL
   *
   * @{
   */
  /**
   * @file
   * @author Markus Blatt
   * @brief Contains one level overlapping Schwarz preconditioners
   */
#ifdef HAVE_SUPERLU

  /**
   * @brief Initializer for SuperLU Matrices representing the subdomains.
   */
  template<class I, class S>
  class OverlappingSchwarzInitializer
  {
  public:
    typedef I InitializerList;
    typedef typename InitializerList::value_type AtomInitializer;
    typedef typename AtomInitializer::Matrix Matrix;
    typedef typename Matrix::const_iterator Iter;
    typedef typename Matrix::row_type::const_iterator CIter;

    typedef S IndexSet;

    OverlappingSchwarzInitializer(InitializerList& il,
                                  const IndexSet& indices);


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
      typedef std::size_t size_type;
      typedef std::map<size_type,std::size_t> Map;
      typedef typename Map::iterator iterator;
      typedef typename Map::const_iterator const_iterator;

      IndexMap();

      void insert(size_type grow);

      const_iterator find(size_type grow) const;

      iterator find(size_type grow);

      iterator end();

      const_iterator end() const;

    private:
      std::map<size_type,std::size_t> map_;
      std::size_t row;
    };


    typedef typename InitializerList::iterator InitIterator;
    typedef typename IndexSet::const_iterator IndexIteratur;
    InitializerList* initializers;
    const IndexSet *indices;
    std::vector<IndexMap> indexMaps;
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

  struct SymmetricMultiplicativeSchwarzMode
  {};

  namespace
  {
    template<typename T>
    struct AdditiveAdder
    {};

    template<typename T, typename A, int n>
    struct AdditiveAdder<BlockVector<FieldVector<T,n>,A> >
    {
      AdditiveAdder(BlockVector<FieldVector<T,n>,A>& v, BlockVector<FieldVector<T,n>,A>& x, const T* t, const T& relax_);
      void operator()(const std::size_t& domain);
      void axpy();

    private:
      const T* t;
      BlockVector<FieldVector<T,n>,A>* v;
      BlockVector<FieldVector<T,n>,A>* x;
      T relax;
      int i;
    };

    template<typename T>
    struct MultiplicativeAdder
    {};

    template<typename T, typename A, int n>
    struct MultiplicativeAdder<BlockVector<FieldVector<T,n>,A> >
    {
      MultiplicativeAdder(BlockVector<FieldVector<T,n>,A>& v, BlockVector<FieldVector<T,n>,A>& x, const T* t, const T& relax_);
      void operator()(const std::size_t& domain);
      void axpy();

    private:
      const T* t;
      BlockVector<FieldVector<T,n>,A>* v;
      BlockVector<FieldVector<T,n>,A>* x;
      T relax;
      int i;
    };


    template<typename T, class X>
    struct AdderSelector
    {};

    template<class X>
    struct AdderSelector<AdditiveSchwarzMode,X>
    {
      typedef AdditiveAdder<X> Adder;
    };

    template<class X>
    struct AdderSelector<MultiplicativeSchwarzMode,X>
    {
      typedef MultiplicativeAdder<X> Adder;
    };

    template<class X>
    struct AdderSelector<SymmetricMultiplicativeSchwarzMode,X>
    {
      typedef MultiplicativeAdder<X> Adder;
    };

    template<typename T1, typename T2, bool forward>
    struct ApplyHelper
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
    struct ApplyHelper<T1,T2,false>
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

    template<class T>
    struct Applier
    {
      typedef T smoother;
      typedef typename smoother::range_type range_type;

      static void apply(smoother& sm, range_type& v, const range_type& b)
      {
        sm.template apply<true>(v, b);
      }
    };

    template<class M, class X, class TA>
    struct Applier<SeqOverlappingSchwarz<M,X,SymmetricMultiplicativeSchwarzMode, TA> >
    {
      typedef SeqOverlappingSchwarz<M,X,SymmetricMultiplicativeSchwarzMode, TA> smoother;
      typedef typename smoother::range_type range_type;

      static void apply(smoother& sm, range_type& v, const range_type& b)
      {
        sm.template apply<true>(v, b);
        sm.template apply<false>(v, b);
      }
    };
  }

  /**
   * @brief Sequential overlapping Schwarz preconditioner
   */
  template<class M, class X, class TM=AdditiveSchwarzMode, class TA=std::allocator<X> >
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
    typedef std::set<size_type, std::less<size_type>, typename TA::template rebind< std::less<size_type> >::other> subdomain_type;

    /** @brief The vector type containing the subdomain to row index mapping. */
    typedef std::vector<subdomain_type, typename TA::template rebind<subdomain_type>::other> subdomain_vector;

    /** @brief The type for the row to subdomain mapping. */
    typedef SLList<size_type, typename TA::template rebind<size_type>::other> subdomain_list;

    /** @brief The vector type containing the row index to subdomain mapping. */
    typedef std::vector<subdomain_list, typename TA::template rebind<subdomain_list>::other > rowtodomain_vector;

    /** @brief The type for the SuperLU solver in use. */
    typedef SuperLU<matrix_type> slu;

    /** @brief The vector type containing SuperLU solvers. */
    typedef std::vector<slu, typename TA::template rebind<slu>::other> slu_vector;

    enum {
      //! \brief The category the precondtioner is part of.
      category = SolverCategory::sequential
    };

    /**
     * @brief Construct the overlapping Schwarz method.
     * @param mat The matrix to precondition.
     * @param subdomains Array of sets of rowindices belonging to an overlapping
     * subdomain
     * @warning Each rowindex should be part of at least one subdomain!
     */
    SeqOverlappingSchwarz(const matrix_type& mat, const subdomain_vector& subDomains,
                          field_type relaxationFactor=1);

    /**
     * Construct the overlapping Schwarz method
     * @param mat The matrix to precondition.
     * @param rowToDomain The mapping of the rows onto the domains.
     */
    SeqOverlappingSchwarz(const matrix_type& mat, const rowtodomain_vector& rowToDomain,
                          field_type relaxationFactor=1);

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
    virtual void post (X& x) {}

  private:
    const M& mat;
    slu_vector solvers;
    subdomain_vector subDomains;
    field_type relax;

    int maxlength;

    void initialize(const rowtodomain_vector& rowToDomain, const matrix_type& mat);

    template<typename T>
    struct Assigner
    {};

    template<typename T, typename A, int n>
    struct Assigner<BlockVector<FieldVector<T,n>,A> >
    {
      Assigner(const M& mat, T* x, const BlockVector<FieldVector<T,n>,A>& b, const BlockVector<FieldVector<T,n>,A>& x);
      void operator()(const std::size_t& domain);
    private:
      const M* mat;
      T* rhs;
      const BlockVector<FieldVector<T,n>,A>* b;
      const BlockVector<FieldVector<T,n>,A>* x;
      int i;
    };

  };


  template<class I, class S>
  OverlappingSchwarzInitializer<I,S>::OverlappingSchwarzInitializer(InitializerList& il,
                                                                    const IndexSet& idx)
    : initializers(&il), indices(&idx), indexMaps(il.size())
  {}


  template<class I, class S>
  void OverlappingSchwarzInitializer<I,S>::addRowNnz(const Iter& row)
  {
    typedef typename IndexSet::value_type::const_iterator iterator;
    for(iterator domain=(*indices)[row.index()].begin(); domain != (*indices)[row.index()].end(); ++domain) {
      (*initializers)[*domain].addRowNnz(row);
      indexMaps[*domain].insert(row.index());
    }
  }

  template<class I, class S>
  void OverlappingSchwarzInitializer<I,S>::allocate()
  {
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::allocateMatrixStorage));
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::allocateMarker));
  }

  template<class I, class S>
  void OverlappingSchwarzInitializer<I,S>::countEntries(const Iter& row, const CIter& col) const
  {
    typedef typename IndexSet::value_type::const_iterator iterator;
    for(iterator domain=(*indices)[row.index()].begin(); domain != (*indices)[row.index()].end(); ++domain) {
      typename std::map<std::size_t,std::size_t>::const_iterator v = indexMaps[*domain].find(col.index());
      if(v!= indexMaps[*domain].end()) {
        (*initializers)[*domain].countEntries(indexMaps[*domain].find(col.index())->second);
      }
    }
  }

  template<class I, class S>
  void OverlappingSchwarzInitializer<I,S>::calcColstart() const
  {
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::calcColstart));
  }

  template<class I, class S>
  void OverlappingSchwarzInitializer<I,S>::copyValue(const Iter& row, const CIter& col) const
  {
    typedef typename IndexSet::value_type::const_iterator iterator;
    for(iterator domain=(*indices)[row.index()].begin(); domain!= (*indices)[row.index()].end(); ++domain) {
      typename std::map<std::size_t,std::size_t>::const_iterator v = indexMaps[*domain].find(col.index());
      if(v!= indexMaps[*domain].end()) {
        (*initializers)[*domain].copyValue(col, indexMaps[*domain].find(row.index())->second,
                                           indexMaps[*domain].find(col.index())->second);
      }
    }
  }

  template<class I, class S>
  void OverlappingSchwarzInitializer<I,S>::createMatrix() const
  {
    std::for_each(initializers->begin(), initializers->end(),
                  std::mem_fun_ref(&AtomInitializer::createMatrix));
  }

  template<class I, class S>
  OverlappingSchwarzInitializer<I,S>::IndexMap::IndexMap()
    : row(0)
  {}

  template<class I, class S>
  void OverlappingSchwarzInitializer<I,S>::IndexMap::insert(size_type grow)
  {
    assert(map_.find(grow)==map_.end());
    map_.insert(std::make_pair(grow, row++));
  }

  template<class I, class S>
  typename OverlappingSchwarzInitializer<I,S>::IndexMap::const_iterator
  OverlappingSchwarzInitializer<I,S>::IndexMap::find(size_type grow) const
  {
    return map_.find(grow);
  }

  template<class I, class S>
  typename OverlappingSchwarzInitializer<I,S>::IndexMap::iterator
  OverlappingSchwarzInitializer<I,S>::IndexMap::find(size_type grow)
  {
    return map_.find(grow);
  }

  template<class I, class S>
  typename OverlappingSchwarzInitializer<I,S>::IndexMap::const_iterator
  OverlappingSchwarzInitializer<I,S>::IndexMap::end() const
  {
    return map_.end();
  }

  template<class I, class S>
  typename OverlappingSchwarzInitializer<I,S>::IndexMap::iterator
  OverlappingSchwarzInitializer<I,S>::IndexMap::end()
  {
    return map_.end();
  }

  template<class M, class X, class TM, class TA>
  SeqOverlappingSchwarz<M,X,TM,TA>::SeqOverlappingSchwarz(const matrix_type& mat_, const rowtodomain_vector& rowToDomain,
                                                          field_type relaxationFactor)
    : mat(mat_), relax(relaxationFactor)
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
    int domains=0;
    for(RowDomainIterator iter=rowToDomain.begin(); iter != rowToDomain.end(); ++iter)
      for(DomainIterator d=iter->begin(); d != iter->end(); ++d)
        domains=std::max(domains, *d);
    ++domains;

    solvers.resize(domains);
    subDomains.resize(domains);

    // initialize subdomains to row mapping from row to subdomain mapping
    int row=0;
    for(RowDomainIterator iter=rowToDomain.begin(); iter != rowToDomain.end(); ++iter, ++row)
      for(DomainIterator d=iter->begin(); d != iter->end(); ++d)
        subDomains[*d].insert(row);

#ifdef DUNE_ISTL_WITH_CHECKING
    int i=0;
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
    initialize(rowToDomain, mat);
  }

  template<class M, class X, class TM, class TA>
  SeqOverlappingSchwarz<M,X,TM,TA>::SeqOverlappingSchwarz(const matrix_type& mat_,
                                                          const subdomain_vector& sd,
                                                          field_type relaxationFactor)
    :  mat(mat_), solvers(sd.size()), subDomains(sd), relax(relaxationFactor)
  {
    typedef typename subdomain_vector::const_iterator DomainIterator;

#ifdef DUNE_ISTL_WITH_CHECKING
    int i=0;

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

    int domainId=0;

    for(DomainIterator domain=sd.begin(); domain != sd.end(); ++domain, ++domainId) {
      typedef typename subdomain_type::const_iterator iterator;
      for(iterator row=domain->begin(); row != domain->end(); ++row)
        rowToDomain[*row].push_back(domainId);
    }

    initialize(rowToDomain, mat);
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

  template<class M, class X, class TM, class TA>
  void SeqOverlappingSchwarz<M,X,TM,TA>::initialize(const rowtodomain_vector& rowToDomain, const matrix_type& mat)
  {
    typedef typename std::vector<SuperMatrixInitializer<matrix_type> >::iterator InitializerIterator;
    typedef typename subdomain_vector::const_iterator DomainIterator;
    typedef typename slu_vector::iterator SolverIterator;
    // initialize the initializers
    DomainIterator domain=subDomains.begin();

    // Create the initializers list.
    std::vector<SuperMatrixInitializer<matrix_type> > initializers(subDomains.size());

    SolverIterator solver=solvers.begin();
    for(InitializerIterator initializer=initializers.begin(); initializer!=initializers.end();
        ++initializer, ++solver, ++domain) {
      solver->mat.N_=SeqOverlappingSchwarzDomainSize<M>::size(*domain);
      solver->mat.M_=SeqOverlappingSchwarzDomainSize<M>::size(*domain);
      //solver->setVerbosity(true);
      *initializer=SuperMatrixInitializer<matrix_type>(solver->mat);
    }

    // Set up the supermatrices according to the subdomains
    typedef OverlappingSchwarzInitializer<std::vector<SuperMatrixInitializer<matrix_type> >,
        rowtodomain_vector > Initializer;

    Initializer initializer(initializers, rowToDomain);
    copyToSuperMatrix(initializer, mat);
    if(solvers.size()==1)
      assert(solvers[0].mat==mat);

    /*    for(SolverIterator solver=solvers.begin(); solver!=solvers.end(); ++solver)
          dPrint_CompCol_Matrix("superlu", &static_cast<SuperMatrix&>(solver->mat)); */

    // Calculate the LU decompositions
    std::for_each(solvers.begin(), solvers.end(), std::mem_fun_ref(&slu::decompose));
    maxlength=0;
    for(SolverIterator solver=solvers.begin(); solver!=solvers.end(); ++solver) {
      assert(solver->mat.N()==solver->mat.M());
      maxlength=std::max(maxlength, solver->mat.N());
      //writeCompColMatrixToMatlab(static_cast<SuperLUMatrix<M>&>(solver->mat), std::cout);
    }

  }

  template<class M, class X, class TM, class TA>
  void SeqOverlappingSchwarz<M,X,TM,TA>::apply(X& x, const X& b)
  {
    Applier<SeqOverlappingSchwarz>::apply(*this, x, b);
  }

  template<class M, class X, class TM, class TA>
  template<bool forward>
  void SeqOverlappingSchwarz<M,X,TM,TA>::apply(X& x, const X& b)
  {
    typedef typename X::block_type block;
    typedef slu_vector solver_vector;
    typedef typename ApplyHelper<solver_vector,subdomain_vector,forward>::solver_iterator iterator;
    typedef typename ApplyHelper<solver_vector,subdomain_vector,forward>::domain_iterator
    domain_iterator;

    field_type *lhs=new field_type[maxlength];
    field_type *rhs=new field_type[maxlength];
    for(int i=0; i<maxlength; ++i)
      lhs[i]=0;


    domain_iterator domain=ApplyHelper<solver_vector,subdomain_vector,forward>::begin(subDomains);

    X v(x); // temporary for the update
    v=0;

    typedef typename AdderSelector<TM,X>::Adder Adder;
    for(iterator solver=ApplyHelper<solver_vector,subdomain_vector,forward>::begin(solvers);
        solver != ApplyHelper<solver_vector,subdomain_vector,forward>::end(solvers); ++solver, ++domain) {
      //Copy rhs to C-array for SuperLU
      std::for_each(domain->begin(), domain->end(), Assigner<X>(mat, rhs, b, x));

      solver->apply(lhs, rhs);
      //Add relaxed correction to from SuperLU to v
      std::for_each(domain->begin(), domain->end(), Adder(v, x, lhs, relax));
    }

    Adder adder(v, x, lhs, relax);
    adder.axpy();
    delete[] lhs;
    delete[] rhs;

  }

  template<class M, class X, class TM, class TA>
  template<typename T, typename A, int n>
  SeqOverlappingSchwarz<M,X,TM,TA>::Assigner<BlockVector<FieldVector<T,n>,A> >::Assigner(const M& mat_, T* rhs_, const BlockVector<FieldVector<T,n>,A>& b_,
                                                                                         const BlockVector<FieldVector<T,n>,A>& x_)
    : mat(&mat_), rhs(rhs_), b(&b_), x(&x_), i(0)
  {}

  template<class M, class X, class TM, class TA>
  template<typename T, typename A, int n>
  void SeqOverlappingSchwarz<M,X,TM,TA>::Assigner<BlockVector<FieldVector<T,n>,A> >::operator()(const std::size_t& domainIndex)
  {
    int starti;
    starti = i;

    for(int j=0; j<n; ++j, ++i)
      rhs[i]=(*b)[domainIndex][j];

    // loop over all Matrix row entries and calculate defect.
    typedef typename M::ConstColIterator col_iterator;
    typedef typename subdomain_type::const_iterator domain_iterator;

    for(col_iterator col=(*mat)[domainIndex].begin(); col!=(*mat)[domainIndex].end(); ++col) {
      typename X::block_type tmp;
      (*col).mv((*x)[col.index()], tmp);
      i-=n;
      for(int j=0; j<n; ++j, ++i)
        rhs[i]-=tmp[j];
    }
    assert(starti+n==i);
  }
  namespace
  {

    template<typename T, typename A, int n>
    AdditiveAdder<BlockVector<FieldVector<T,n>,A> >::AdditiveAdder(BlockVector<FieldVector<T,n>,A>& v_,
                                                                   BlockVector<FieldVector<T,n>,A>& x_, const T* t_, const T& relax_)
      : t(t_), v(&v_), x(&x_), relax(relax_), i(0)
    {}

    template<typename T, typename A, int n>
    void AdditiveAdder<BlockVector<FieldVector<T,n>,A> >::operator()(const std::size_t& domainIndex)
    {
      for(int j=0; j<n; ++j, ++i)
        (*v)[domainIndex][j]+=t[i];
    }


    template<typename T, typename A, int n>
    void AdditiveAdder<BlockVector<FieldVector<T,n>,A> >::axpy()
    {
      x->axpy(relax,*v);
    }


    template<typename T, typename A, int n>
    MultiplicativeAdder<BlockVector<FieldVector<T,n>,A> >
    ::MultiplicativeAdder(BlockVector<FieldVector<T,n>,A>& v_,
                          BlockVector<FieldVector<T,n>,A>& x_, const T* t_, const T& relax_)
      : t(t_), v(&v_), x(&x_), relax(relax_), i(0)
    {}


    template<typename T, typename A, int n>
    void MultiplicativeAdder<BlockVector<FieldVector<T,n>,A> >::operator()(const std::size_t& domainIndex)
    {
      for(int j=0; j<n; ++j, ++i)
        (*x)[domainIndex][j]+=relax*t[i];
    }


    template<typename T, typename A, int n>
    void MultiplicativeAdder<BlockVector<FieldVector<T,n>,A> >::axpy()
    {}
  }

  /** @} */
#endif
}

#endif
