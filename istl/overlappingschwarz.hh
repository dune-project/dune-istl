// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_OVERLAPPINGSCHWARZ_HH
#define DUNE_OVERLAPPINGSCHWARZ_HH

#include <algorithm>
#include <functional>
#include <vector>
#include <set>
#include <dune/common/sllist.hh>
#include "preconditioners.hh"
#include "superlu.hh"
#include "bvector.hh"
namespace Dune
{

#ifdef HAVE_SUPERLU

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

      IndexMap()
        : row(0)
      {}
      void insert(size_type grow)
      {
        assert(map_.find(grow)==map_.end());
        map_.insert(std::make_pair(grow, row++));
      }
      const_iterator find(size_type grow) const
      {
        return map_.find(grow);
      }
      iterator find(size_type grow)
      {
        return map_.find(grow);
      }
      iterator end()
      {
        return map_.end();
      }
      const_iterator end() const
      {
        return map_.end();
      }
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

  template<class M, class X, class TA=std::allocator<X> >
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
     * @brief The field type of the preconditioner.
     */
    typedef typename X::field_type field_type;

    typedef typename matrix_type::size_type size_type;

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
    SeqOverlappingSchwarz(const matrix_type& mat, const std::vector<std::set<size_type,std::less<size_type> > >& subDomains,
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

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x) {}

  private:
    std::vector<SuperLU<matrix_type>,TA> solvers;
    std::vector<std::set<size_type> > subDomains;
    field_type relax;

    int maxlength;

    template<typename T>
    struct Assigner
    {};

    template<typename T, typename A, int n>
    struct Assigner<BlockVector<FieldVector<T,n>,A> >
    {
      Assigner(T* x, const BlockVector<FieldVector<T,n>,A>& y);
      void operator()(const std::size_t& domain);
    private:
      T* x;
      const BlockVector<FieldVector<T,n>,A>* y;
      int i;
    };

    template<typename T>
    struct Adder
    {};

    template<typename T, typename A, int n>
    struct Adder<BlockVector<FieldVector<T,n>,A> >
    {
      Adder(BlockVector<FieldVector<T,n>,A>& y, const T* x, T relax);
      void operator()(const std::size_t& domain);
    private:
      const T* x;
      BlockVector<FieldVector<T,n>,A>* y;
      int i;
      T relax;
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

  template<class M, class X, class TA>
  SeqOverlappingSchwarz<M,X,TA>::SeqOverlappingSchwarz(const matrix_type& mat,
                                                       const std::vector<std::set<size_type> >& sd,
                                                       field_type relaxationFactor)
    :  solvers(sd.size()), subDomains(sd), relax(relaxationFactor)
  {

    // Create the initializers list.
    std::vector<SuperMatrixInitializer<matrix_type> > initializers(subDomains.size());

    typedef typename std::vector<SuperMatrixInitializer<matrix_type> >::iterator InitializerIterator;
    typedef typename std::vector<SuperLU<matrix_type>,TA>::iterator SolverIterator;
    typedef typename std::vector<std::set<size_type> >::const_iterator DomainIterator;
    DomainIterator domain=subDomains.begin();

    SolverIterator solver=solvers.begin();
    for(InitializerIterator initializer=initializers.begin(); initializer!=initializers.end();
        ++initializer, ++solver, ++domain) {
      solver->mat.N_=domain->size();
      solver->mat.M_=domain->size();
      *initializer=SuperMatrixInitializer<matrix_type>(solver->mat);
    }

    // Create a row to subdomain mapping
    std::vector<SLList<size_type,TA>,TA> rowToDomain(mat.N());

    int domainId=0;

    for(DomainIterator domain=sd.begin(); domain != sd.end(); ++domain, ++domainId) {
      typedef typename std::set<size_type>::const_iterator iterator;
      for(iterator row=domain->begin(); row != domain->end(); ++row)
        rowToDomain[*row].push_back(domainId);
    }


    // Set up the supermatrices according to the subdomains
    typedef OverlappingSchwarzInitializer<std::vector<SuperMatrixInitializer<matrix_type> >,
        std::vector<SLList<size_type,TA>,TA> > Initializer;

    Initializer initializer(initializers, rowToDomain);
    copyToSuperMatrix(initializer, mat);
    if(solvers.size()==1)
      assert(solvers[0].mat==mat);

    // Calculate the LU decompositions
    std::for_each(solvers.begin(), solvers.end(), std::mem_fun_ref(&SuperLU<matrix_type>::decompose));
    maxlength=0;
    for(SolverIterator solver=solvers.begin(); solver!=solvers.end(); ++solver) {
      assert(solver->mat.N()==solver->mat.M());
      maxlength=std::max(maxlength, solver->mat.N());
    }
  }

  template<class M, class X, class TA>
  void SeqOverlappingSchwarz<M,X,TA>::apply(X& v, const X& d)
  {
    typedef typename std::vector<SuperLU<matrix_type>,TA>::iterator iterator;
    typedef typename std::vector<std::set<size_type> >::const_iterator domain_iterator;
    typedef typename std::set<size_type,std::less<size_type>,TA>::const_iterator index_iterator;

    field_type *lhs=new field_type[maxlength];
    field_type *rhs=new field_type[maxlength];
    for(int i=0; i<maxlength; ++i)
      lhs[i]=0;
    domain_iterator domain=subDomains.begin();

    for(iterator solver=solvers.begin(); solver != solvers.end(); ++solver, ++domain) {
      //InverseOperatoResult res;
      //Copy defect to C-array for SuperLU
      std::for_each(domain->begin(), domain->end(), Assigner<X>(rhs, d));
      solver->apply(lhs, rhs);
      //Add relaxed correction to form SuperLU to v
      std::for_each(domain->begin(), domain->end(), Adder<X>(v, lhs, relax));
    }

    delete[] lhs;
    delete[] rhs;
  }

  template<class M, class X, class TA>
  template<typename T, typename A, int n>
  SeqOverlappingSchwarz<M,X,TA>::Assigner<BlockVector<FieldVector<T,n>,A> >::Assigner(T* x_, const BlockVector<FieldVector<T,n>,A>& y_)
    : x(x_), y(&y_), i(0)
  {}

  template<class M, class X, class TA>
  template<typename T, typename A, int n>
  void SeqOverlappingSchwarz<M,X,TA>::Assigner<BlockVector<FieldVector<T,n>,A> >::operator()(const std::size_t& domainIndex)
  {
    for(int j=0; j<n; ++j, ++i)
      x[i]=(*y)[domainIndex][j];
  }

  template<class M, class X, class TA>
  template<typename T, typename A, int n>
  SeqOverlappingSchwarz<M,X,TA>::Adder<BlockVector<FieldVector<T,n>,A> >::Adder(BlockVector<FieldVector<T,n>,A>& y_, const T* x_,
                                                                                T r)
    : x(x_), y(&y_), i(0), relax(r)
  {}

  template<class M, class X, class TA>
  template<typename T, typename A, int n>
  void SeqOverlappingSchwarz<M,X,TA>::Adder<BlockVector<FieldVector<T,n>,A> >::operator()(const std::size_t& domainIndex)
  {
    for(int j=0; j<n; ++j, ++i)
      (*y)[domainIndex][j]+=relax*x[i];
  }

#endif
};

#endif
