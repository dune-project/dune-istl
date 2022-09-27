// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_AGGREGATES_HH
#define DUNE_AMG_AGGREGATES_HH


#include "parameters.hh"
#include "graph.hh"
#include "properties.hh"
#include "combinedfunctor.hh"

#include <dune/common/timer.hh>
#include <dune/common/stdstreams.hh>
#include <dune/common/poolallocator.hh>
#include <dune/common/sllist.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/scalarmatrixview.hh>

#include <utility>
#include <set>
#include <algorithm>
#include <complex>
#include <limits>
#include <ostream>
#include <tuple>

namespace Dune
{
  namespace Amg
  {

    /**
     * @addtogroup ISTL_PAAMG
     *
     * @{
     */
    /** @file
     * @author Markus Blatt
     * @brief Provides classes for the Coloring process of AMG
     */

    /**
     * @brief Base class of all aggregation criterions.
     */
    template<class T>
    class AggregationCriterion : public T
    {

    public:
      /**
       * @brief The policy for calculating the dependency graph.
       */
      typedef T DependencyPolicy;

      /**
       * @brief Constructor.
       *
       * The parameters will be initialized with default values suitable
       * for 2D isotropic problems.
       *
       * If that does not fit your needs either use setDefaultValuesIsotropic
       * setDefaultValuesAnisotropic or setup the values by hand
       */
      AggregationCriterion()
        : T()
      {}

      AggregationCriterion(const Parameters& parms)
        : T(parms)
      {}
      /**
       * @brief Sets reasonable default values for an isotropic problem.
       *
       * Reasonable means that we should end up with cube aggregates of
       * diameter 2.
       *
       * @param dim The dimension of the problem.
       * @param diameter The preferred diameter for the aggregation.
       */
      void setDefaultValuesIsotropic(std::size_t dim, std::size_t diameter=2)
      {
        this->setMaxDistance(diameter-1);
        std::size_t csize=1;

        for(; dim>0; dim--) {
          csize*=diameter;
          this->setMaxDistance(this->maxDistance()+diameter-1);
        }
        this->setMinAggregateSize(csize);
        this->setMaxAggregateSize(static_cast<std::size_t>(csize*1.5));
      }

      /**
       * @brief Sets reasonable default values for an aisotropic problem.
       *
       * Reasonable means that we should end up with cube aggregates with
       * sides of diameter 2 and sides in one dimension that are longer
       * (e.g. for 3D: 2x2x3).
       *
       * @param dim The dimension of the problem.
       * @param diameter The preferred diameter for the aggregation.
       */
      void setDefaultValuesAnisotropic(std::size_t dim,std::size_t diameter=2)
      {
        setDefaultValuesIsotropic(dim, diameter);
        this->setMaxDistance(this->maxDistance()+dim-1);
      }
    };

    template<class T>
    std::ostream& operator<<(std::ostream& os, const AggregationCriterion<T>& criterion)
    {
      os<<"{ maxdistance="<<criterion.maxDistance()<<" minAggregateSize="
      <<criterion.minAggregateSize()<< " maxAggregateSize="<<criterion.maxAggregateSize()
      <<" connectivity="<<criterion.maxConnectivity()<<" debugLevel="<<criterion.debugLevel()<<"}";
      return os;
    }

    /**
     * @brief Dependency policy for symmetric matrices.
     *
     * We assume that not only the sparsity pattern is symmetric
     * but also the entries (a_ij=aji). If that is not the case
     * the resulting dependency graph might be unsymmetric.
     *
     * \tparam M The type of the matrix
     * \tparam N The type of the metric that turns matrix blocks into
     * field values
     */
    template<class M, class N>
    class SymmetricMatrixDependency : public Dune::Amg::Parameters
    {
    public:
      /**
       * @brief The matrix type we build the dependency of.
       */
      typedef M Matrix;

      /**
       * @brief The norm to use for examining the matrix entries.
       */
      typedef N Norm;

      /**
       * @brief Constant Row iterator of the matrix.
       */
      typedef typename Matrix::row_type Row;

      /**
       * @brief Constant column iterator of the matrix.
       */
      typedef typename Matrix::ConstColIterator ColIter;

      void init(const Matrix* matrix);

      void initRow(const Row& row, int index);

      void examine(const ColIter& col);

      template<class G>
      void examine(G& graph, const typename G::EdgeIterator& edge, const ColIter& col);

      bool isIsolated();


      SymmetricMatrixDependency(const Parameters& parms)
        : Parameters(parms)
      {}
      SymmetricMatrixDependency()
        : Parameters()
      {}

    protected:
      /** @brief The matrix we work on. */
      const Matrix* matrix_;
      /** @brief The current max value.*/
      typedef typename Matrix::field_type field_type;
      typedef typename FieldTraits<field_type>::real_type real_type;
      real_type maxValue_;
      /** @brief The functor for calculating the norm. */
      Norm norm_;
      /** @brief index of the currently evaluated row. */
      int row_;
      /** @brief The norm of the current diagonal. */
      real_type diagonal_;
      std::vector<real_type> vals_;
      typename std::vector<real_type>::iterator valIter_;

    };


    template<class M, class N>
    inline void SymmetricMatrixDependency<M,N>::init(const Matrix* matrix)
    {
      matrix_ = matrix;
    }

    template<class M, class N>
    inline void SymmetricMatrixDependency<M,N>::initRow(const Row& row, int index)
    {
      using std::min;
      vals_.assign(row.size(), 0.0);
      assert(vals_.size()==row.size());
      valIter_=vals_.begin();

      maxValue_ = min(- std::numeric_limits<real_type>::max(), std::numeric_limits<real_type>::min());
      diagonal_=norm_(row[index]);
      row_ = index;
    }

    template<class M, class N>
    inline void SymmetricMatrixDependency<M,N>::examine(const ColIter& col)
    {
      using std::max;
      // skip positive offdiagonals if norm preserves sign of them.
      real_type eij = norm_(*col);
      if(!N::is_sign_preserving || eij<0)  // || eji<0)
      {
        *valIter_ = eij/diagonal_*eij/norm_(matrix_->operator[](col.index())[col.index()]);
        maxValue_ = max(maxValue_, *valIter_);
      }else
        *valIter_ =0;
      ++valIter_;
    }

    template<class M, class N>
    template<class G>
    inline void SymmetricMatrixDependency<M,N>::examine(G&, const typename G::EdgeIterator& edge, const ColIter&)
    {
      if(*valIter_ > alpha() * maxValue_) {
        edge.properties().setDepends();
        edge.properties().setInfluences();
      }
      ++valIter_;
    }

    template<class M, class N>
    inline bool SymmetricMatrixDependency<M,N>::isIsolated()
    {
      if(diagonal_==0)
        DUNE_THROW(Dune::ISTLError, "No diagonal entry for row "<<row_<<".");
      valIter_=vals_.begin();
      return maxValue_  < beta();
    }

    /**
     * @brief Dependency policy for symmetric matrices.
     */
    template<class M, class N>
    class Dependency : public Parameters
    {
    public:
      /**
       * @brief The matrix type we build the dependency of.
       */
      typedef M Matrix;

      /**
       * @brief The norm to use for examining the matrix entries.
       */
      typedef N Norm;

      /**
       * @brief Constant Row iterator of the matrix.
       */
      typedef typename Matrix::row_type Row;

      /**
       * @brief Constant column iterator of the matrix.
       */
      typedef typename Matrix::ConstColIterator ColIter;

      void init(const Matrix* matrix);

      void initRow(const Row& row, int index);

      void examine(const ColIter& col);

      template<class G>
      void examine(G& graph, const typename G::EdgeIterator& edge, const ColIter& col);

      bool isIsolated();

      Dependency(const Parameters& parms)
        : Parameters(parms)
      {}

      Dependency()
        : Parameters()
      {}

    protected:
      /** @brief The matrix we work on. */
      const Matrix* matrix_;
      /** @brief The current max value.*/
      typedef typename Matrix::field_type field_type;
      typedef typename FieldTraits<field_type>::real_type real_type;
      real_type maxValue_;
      /** @brief The functor for calculating the norm. */
      Norm norm_;
      /** @brief index of the currently evaluated row. */
      int row_;
      /** @brief The norm of the current diagonal. */
      real_type diagonal_;
    };

    /**
     * @brief Dependency policy for symmetric matrices.
     */
    template<class M, class N>
    class SymmetricDependency : public Parameters
    {
    public:
      /**
       * @brief The matrix type we build the dependency of.
       */
      typedef M Matrix;

      /**
       * @brief The norm to use for examining the matrix entries.
       */
      typedef N Norm;

      /**
       * @brief Constant Row iterator of the matrix.
       */
      typedef typename Matrix::row_type Row;

      /**
       * @brief Constant column iterator of the matrix.
       */
      typedef typename Matrix::ConstColIterator ColIter;

      void init(const Matrix* matrix);

      void initRow(const Row& row, int index);

      void examine(const ColIter& col);

      template<class G>
      void examine(G& graph, const typename G::EdgeIterator& edge, const ColIter& col);

      bool isIsolated();


      SymmetricDependency(const Parameters& parms)
        : Parameters(parms)
      {}
      SymmetricDependency()
        : Parameters()
      {}

    protected:
      /** @brief The matrix we work on. */
      const Matrix* matrix_;
      /** @brief The current max value.*/
      typedef typename Matrix::field_type field_type;
      typedef typename FieldTraits<field_type>::real_type real_type;
      real_type maxValue_;
      /** @brief The functor for calculating the norm. */
      Norm norm_;
      /** @brief index of the currently evaluated row. */
      int row_;
      /** @brief The norm of the current diagonal. */
      real_type diagonal_;
    private:
      void initRow(const Row& row, int index, const std::true_type&);
      void initRow(const Row& row, int index, const std::false_type&);
    };

    /**
     * @brief Norm that uses only the [N][N] entry of the block to determine couplings.
     *
     */
    template<int N>
    class Diagonal
    {
    public:
      enum { /* @brief We preserve the sign.*/
        is_sign_preserving = true
      };

      /**
       * @brief compute the norm of a matrix.
       * @param m The matrix to compute the norm of
       */
      template<class M>
      typename FieldTraits<typename M::field_type>::real_type operator()(const M& m,
                                                                         [[maybe_unused]] typename std::enable_if_t<!Dune::IsNumber<M>::value>* sfinae = nullptr) const
      {
        typedef typename M::field_type field_type;
        typedef typename FieldTraits<field_type>::real_type real_type;
        static_assert( std::is_convertible<field_type, real_type >::value,
                  "use of diagonal norm in AMG not implemented for complex field_type");
        return m[N][N];
        // possible implementation for complex types: return signed_abs(m[N][N]);
      }

      /**
       * @brief Compute the norm of a scalar
       * @param m The scalar to compute the norm of
       */
      template<class M>
      auto operator()(const M& m,
                      typename std::enable_if_t<Dune::IsNumber<M>::value>* sfinae = nullptr) const
      {
        typedef typename FieldTraits<M>::real_type real_type;
        static_assert( std::is_convertible<M, real_type >::value,
                  "use of diagonal norm in AMG not implemented for complex field_type");
        return m;
        // possible implementation for complex types: return signed_abs(m[N][N]);
      }

    private:

      //! return sign * abs_value; for real numbers this is just v
      template<typename T>
      static T signed_abs(const T & v)
      {
        return v;
      }

      //! return sign * abs_value; for complex numbers this is csgn(v) * abs(v)
      template<typename T>
      static T signed_abs(const std::complex<T> & v)
      {
        // return sign * abs_value
        // in case of complex numbers this extends to using the csgn function to determine the sign
        return csgn(v) * std::abs(v);
      }

      //! sign function for complex numbers; for real numbers we assume imag(v) = 0
      template<typename T>
      static T csgn(const T & v)
      {
        return (T(0) < v) - (v < T(0));
      }

      //! sign function for complex numbers
      template<typename T>
      static T csgn(std::complex<T> a)
      {
        return csgn(a.real())+(a.real() == 0.0)*csgn(a.imag());
      }

    };

    /**
     * @brief Norm that uses only the [0][0] entry of the block to determine couplings.
     *
     */
    class FirstDiagonal : public Diagonal<0>
    {};

    /**
     * @brief Functor using the row sum (infinity) norm to determine strong couplings.
     *
     * The is proposed by several people for elasticity problems.
     */
    struct RowSum
    {

      enum { /* @brief We preserve the sign.*/
        is_sign_preserving = false
      };
      /**
       * @brief compute the norm of a matrix.
       * @param m The matrix row to compute the norm of.
       */
      template<class M>
      typename FieldTraits<typename M::field_type>::real_type operator()(const M& m) const
      {
        return m.infinity_norm();
      }
    };

    struct FrobeniusNorm
    {

      enum { /* @brief We preserve the sign.*/
        is_sign_preserving = false
      };
      /**
       * @brief compute the norm of a matrix.
       * @param m The matrix row to compute the norm of.
       */
      template<class M>
      typename FieldTraits<typename M::field_type>::real_type operator()(const M& m) const
      {
        return m.frobenius_norm();
      }
    };
    struct AlwaysOneNorm
    {

      enum { /* @brief We preserve the sign.*/
        is_sign_preserving = false
      };
      /**
       * @brief compute the norm of a matrix.
       * @param m The matrix row to compute the norm of.
       */
      template<class M>
      typename FieldTraits<typename M::field_type>::real_type operator()(const M& m) const
      {
        return 1;
      }
    };
    /**
     * @brief Criterion taking advantage of symmetric matrices.
     *
     * \tparam M The type of the matrix the amg coarsening works on, e.g. BCRSMatrix
     * \tparam Norm The norm to use to determine the strong couplings between the nodes, e.g. FirstDiagonal or RowSum.
     */
    template<class M, class Norm>
    class SymmetricCriterion : public AggregationCriterion<SymmetricDependency<M,Norm> >
    {
    public:
      SymmetricCriterion(const Parameters& parms)
        : AggregationCriterion<SymmetricDependency<M,Norm> >(parms)
      {}
      SymmetricCriterion()
      {}
    };


    /**
     * @brief Criterion suitable for unsymmetric matrices.
     *
     * Nevertheless the sparsity pattern has to be symmetric.
     *
     * \tparam M The type of the matrix the amg coarsening works on, e.g. BCRSMatrix
     * \tparam Norm The norm to use to determine the strong couplings between the nodes, e.g. FirstDiagonal or RowSum.
     */
    template<class M, class Norm>
    class UnSymmetricCriterion : public AggregationCriterion<Dependency<M,Norm> >
    {
    public:
      UnSymmetricCriterion(const Parameters& parms)
        : AggregationCriterion<Dependency<M,Norm> >(parms)
      {}
      UnSymmetricCriterion()
      {}
    };
    // forward declaration
    template<class G> class Aggregator;


    /**
     * @brief Class providing information about the mapping of
     * the vertices onto aggregates.
     *
     * It is assumed that the vertices are consecutively numbered
     * from 0 to the maximum vertex number.
     */
    template<class V>
    class AggregatesMap
    {
    public:

      /**
       * @brief Identifier of not yet aggregated vertices.
       */
      static const V UNAGGREGATED;

      /**
       * @brief Identifier of isolated vertices.
       */
      static const V ISOLATED;
      /**
       * @brief The vertex descriptor type.
       */
      typedef V VertexDescriptor;

      /**
       * @brief The aggregate descriptor type.
       */
      typedef V AggregateDescriptor;

      /**
       * @brief The allocator we use for our lists and the
       * set.
       */
      typedef PoolAllocator<VertexDescriptor,100> Allocator;

      /**
       * @brief The type of a single linked list of vertex
       * descriptors.
       */
      typedef SLList<VertexDescriptor,Allocator> VertexList;

      /**
       * @brief A Dummy visitor that does nothing for each visited edge.
       */
      class DummyEdgeVisitor
      {
      public:
        template<class EdgeIterator>
        void operator()([[maybe_unused]] const EdgeIterator& edge) const
        {}
      };


      /**
       * @brief Constructs without allocating memory.
       */
      AggregatesMap();

      /**
       * @brief Constructs with allocating memory.
       * @param noVertices The number of vertices we will hold information
       * for.
       */
      AggregatesMap(std::size_t noVertices);

      /**
       * @brief Destructor.
       */
      ~AggregatesMap();

      /**
       * @brief Build the aggregates.
       * @param matrix The matrix describing the dependency.
       * @param graph The graph corresponding to the matrix.
       * @param criterion The aggregation criterion.
       * @param finestLevel Whether this the finest level. In that case rows representing
       * Dirichlet boundaries will be detected and ignored during aggregation.
       * @return A tuple of the total number of aggregates, the number of isolated aggregates, the
       *         number of isolated aggregates, the number of aggregates consisting only of one vertex, and
       *         the number of skipped aggregates built.
       */
      template<class M, class G, class C>
      std::tuple<int,int,int,int> buildAggregates(const M& matrix, G& graph, const C& criterion,
                                                  bool finestLevel);

      /**
       * @brief Breadth first search within an aggregate
       *
       * \tparam reset If true the visited flags of the vertices
       *  will be reset after the search
       * \tparam G The type of the graph we perform the search on
       * \tparam F The type of the visitor to operate on the vertices
       *
       * @param start The vertex where the search should start
       * from. This does not need to belong to the aggregate.
       * @param aggregate The aggregate id.
       * @param graph The matrix graph to perform the search on.
       * @param visitedMap A map to mark the already visited vertices
       * @param aggregateVisitor A functor that is called with
       * each G::ConstEdgeIterator with an edge pointing to the
       * aggregate. Use DummyVisitor if these are of no interest.
       */
      template<bool reset, class G, class F, class VM>
      std::size_t breadthFirstSearch(const VertexDescriptor& start,
                                     const AggregateDescriptor& aggregate,
                                     const G& graph,
                                     F& aggregateVisitor,
                                     VM& visitedMap) const;

      /**
       * @brief Breadth first search within an aggregate
       *
       * \tparam L A container type providing push_back(Vertex), and
       * pop_front() in case remove is true
       * \tparam remove If true the entries in the visited list
       * will be removed.
       * \tparam reset If true the visited flag will be reset after
       * the search
       *
       * @param start The vertex where the search should start
       * from. This does not need to belong to the aggregate.
       * @param aggregate The aggregate id.
       * @param graph The matrix graph to perform the search on.
       * @param visited A list to store the visited vertices in.
       * @param aggregateVisitor A functor that is called with
       * each G::ConstEdgeIterator with an edge pointing to the
       * aggregate. Use DummyVisitor these are of no interest.
       * @param nonAggregateVisitor A functor that is called with
       * each G::ConstEdgeIterator with an edge pointing to another
       * aggregate. Use DummyVisitor these are of no interest.
       * @param visitedMap A map to mark the already visited vertices
       */
      template<bool remove, bool reset, class G, class L, class F1, class F2, class VM>
      std::size_t breadthFirstSearch(const VertexDescriptor& start,
                                     const AggregateDescriptor& aggregate,
                                     const G& graph, L& visited, F1& aggregateVisitor,
                                     F2& nonAggregateVisitor,
                                     VM& visitedMap) const;

      /**
       * @brief Allocate memory for holding the information.
       * @param noVertices The total number of vertices to be
       * mapped.
       */
      void allocate(std::size_t noVertices);

      /**
       * @brief Get the number of vertices.
       */
      std::size_t noVertices() const;

      /**
       * @brief Free the allocated memory.
       */
      void free();

      /**
       * @brief Get the aggregate a vertex belongs to.
       * @param v The vertex we want to know the aggregate of.
       * @return The aggregate the vertex is mapped to.
       */
      AggregateDescriptor& operator[](const VertexDescriptor& v);

      /**
       * @brief Get the aggregate a vertex belongs to.
       * @param v The vertex we want to know the aggregate of.
       * @return The aggregate the vertex is mapped to.
       */
      const AggregateDescriptor& operator[](const VertexDescriptor& v) const;

      typedef const AggregateDescriptor* const_iterator;

      const_iterator begin() const
      {
        return aggregates_;
      }

      const_iterator end() const
      {
        return aggregates_+noVertices();
      }

      typedef AggregateDescriptor* iterator;

      iterator begin()
      {
        return aggregates_;
      }

      iterator end()
      {
        return aggregates_+noVertices();
      }
    private:
      /** @brief Prevent copying. */
      AggregatesMap(const AggregatesMap<V>&) = delete;
      /** @brief Prevent assingment. */
      AggregatesMap<V>& operator=(const AggregatesMap<V>&) = delete;

      /**
       * @brief The aggregates the vertices belong to.
       */
      AggregateDescriptor* aggregates_;

      /**
       * @brief The number of vertices in the map.
       */
      std::size_t noVertices_;
    };

    /**
     * @brief Build the dependency of the matrix graph.
     */
    template<class G, class C>
    void buildDependency(G& graph,
                         const typename C::Matrix& matrix,
                         C criterion,
                         bool finestLevel);

    /**
     * @brief A class for temporarily storing the vertices of an
     * aggregate in.
     */
    template<class G, class S>
    class Aggregate
    {

    public:

      /***
       * @brief The type of the matrix graph we work with.
       */
      typedef G MatrixGraph;
      /**
       * @brief The vertex descriptor type.
       */
      typedef typename MatrixGraph::VertexDescriptor Vertex;

      /**
       * @brief The allocator we use for our lists and the
       * set.
       */
      typedef PoolAllocator<Vertex,100> Allocator;

      /**
       * @brief The type of a single linked list of vertex
       * descriptors.
       */
      typedef S VertexSet;

      /** @brief Const iterator over a vertex list. */
      typedef typename VertexSet::const_iterator const_iterator;

      /**
       * @brief Type of the mapping of aggregate members onto distance spheres.
       */
      typedef std::size_t* SphereMap;

      /**
       * @brief Constructor.
       * @param graph The matrix graph we work on.
       * @param aggregates The mapping of vertices onto aggregates.
       * @param connectivity The set of vertices connected to the aggregate.
       * distance spheres.
       * @param front_ The vertices of the current aggregate front.
       */
      Aggregate(MatrixGraph& graph, AggregatesMap<Vertex>& aggregates,
                VertexSet& connectivity, std::vector<Vertex>& front_);

      void invalidate()
      {
        --id_;
      }

      /**
       * @brief Reconstruct the aggregat from an seed node.
       *
       * Will determine all vertices of the same agggregate
       * and reference those.
       */
      void reconstruct(const Vertex& vertex);

      /**
       * @brief Initialize the aggregate with one vertex.
       */
      void seed(const Vertex& vertex);

      /**
       * @brief Add a vertex to the aggregate.
       */
      void add(const Vertex& vertex);

      void add(std::vector<Vertex>& vertex);
      /**
       * @brief Clear the aggregate.
       */
      void clear();

      /**
       * @brief Get the size of the aggregate.
       */
      typename VertexSet::size_type size();
      /**
       * @brief Get tne number of connections to other aggregates.
       */
      typename VertexSet::size_type connectSize();

      /**
       * @brief Get the id identifying the aggregate.
       */
      int id();

      /** @brief get an iterator over the vertices of the aggregate. */
      const_iterator begin() const;

      /** @brief get an iterator over the vertices of the aggregate. */
      const_iterator end() const;

    private:
      /**
       * @brief The vertices of the aggregate.
       */
      VertexSet vertices_;

      /**
       * @brief The number of the currently referenced
       * aggregate.
       */
      int id_;

      /**
       * @brief The matrix graph the aggregates live on.
       */
      MatrixGraph& graph_;

      /**
       * @brief The aggregate mapping we build.
       */
      AggregatesMap<Vertex>& aggregates_;

      /**
       * @brief The connections to other aggregates.
       */
      VertexSet& connected_;

      /**
       * @brief The vertices of the current aggregate front.
       */
      std::vector<Vertex>& front_;
    };

    /**
     * @brief Class for building the aggregates.
     */
    template<class G>
    class Aggregator
    {
    public:

      /**
       * @brief The matrix graph type used.
       */
      typedef G MatrixGraph;

      /**
       * @brief The vertex identifier
       */
      typedef typename MatrixGraph::VertexDescriptor Vertex;

      /** @brief The type of the aggregate descriptor. */
      typedef typename MatrixGraph::VertexDescriptor AggregateDescriptor;

      /**
       * @brief Constructor.
       */
      Aggregator();

      /**
       * @brief Destructor.
       */
      ~Aggregator();

      /**
       * @brief Build the aggregates.
       *
       * \tparam C The type of the coarsening Criterion to use
       *
       * @param m The matrix to build the aggregates accordingly.
       * @param graph A (sub) graph of the matrix.
       * @param aggregates Aggregate map we will build. All entries should be initialized
       * to UNAGGREGATED!
       * @param c The coarsening criterion to use.
       * @param finestLevel Whether this the finest level. In that case rows representing
       * Dirichlet boundaries will be detected and ignored during aggregation.
       * @return A tuple of the total number of aggregates, the number of isolated aggregates, the
       *         number of isolated aggregates, the number of aggregates consisting only of one vertex, and
       *         the number of skipped aggregates built.
       */
      template<class M, class C>
      std::tuple<int,int,int,int> build(const M& m, G& graph,
                                        AggregatesMap<Vertex>& aggregates, const C& c,
                                        bool finestLevel);
    private:
      /**
       * @brief The allocator we use for our lists and the
       * set.
       */
      typedef PoolAllocator<Vertex,100> Allocator;

      /**
       * @brief The single linked list we use.
       */
      typedef SLList<Vertex,Allocator> VertexList;

      /**
       * @brief The set of vertices we use.
       */
      typedef std::set<Vertex,std::less<Vertex>,Allocator> VertexSet;

      /**
       * @brief The type of mapping of aggregate members to spheres.
       */
      typedef std::size_t* SphereMap;

      /**
       * @brief The graph we aggregate for.
       */
      MatrixGraph* graph_;

      /**
       * @brief The vertices of the current aggregate-
       */
      Aggregate<MatrixGraph,VertexSet>* aggregate_;

      /**
       * @brief The vertices of the current aggregate front.
       */
      std::vector<Vertex> front_;

      /**
       * @brief The set of connected vertices.
       */
      VertexSet connected_;

      /**
       * @brief Number of vertices mapped.
       */
      int size_;

      /**
       * @brief Stack.
       */
      class Stack
      {
      public:
        static const Vertex NullEntry;

        Stack(const MatrixGraph& graph,
              const Aggregator<G>& aggregatesBuilder,
              const AggregatesMap<Vertex>& aggregates);
        ~Stack();
        Vertex pop();
      private:
        enum { N = 1300000 };

        /** @brief The graph we work on. */
        const MatrixGraph& graph_;
        /** @brief The aggregates builder. */
        const Aggregator<G>& aggregatesBuilder_;
        /** @brief The aggregates information. */
        const AggregatesMap<Vertex>& aggregates_;
        /** @brief The current size. */
        int size_;
        Vertex maxSize_;
        /** @brief The index of the top element. */
        typename MatrixGraph::ConstVertexIterator begin_;
        typename MatrixGraph::ConstVertexIterator end_;

        /** @brief The values on the stack. */
        Vertex* vals_;

      };

      friend class Stack;

      /**
       * @brief Visits all neighbours of vertex belonging to a
       * specific aggregate.
       *
       * @param vertex The vertex whose neighbours we want to
       * visit.
       * @param aggregate The id of the aggregate.
       * @param visitor The visitor evaluated for each EdgeIterator
       * (by its method operator()(ConstEdgeIterator edge)
       */
      template<class V>
      void visitAggregateNeighbours(const Vertex& vertex, const AggregateDescriptor& aggregate,
                                    const AggregatesMap<Vertex>& aggregates,
                                    V& visitor) const;

      /**
       * @brief An Adaptor for vsitors that only
       * evaluates edges pointing to a specific aggregate.
       */
      template<class V>
      class AggregateVisitor
      {
      public:
        /**
         * @brief The type of the adapted visitor
         */
        typedef V Visitor;
        /**
         * @brief Constructor.
         * @param aggregates The aggregate numbers of the
         * vertices.
         * @param  aggregate The id of the aggregate to visit.
         * @param visitor The visitor.
         */
        AggregateVisitor(const AggregatesMap<Vertex>& aggregates, const AggregateDescriptor& aggregate,
                         Visitor& visitor);

        /**
         * @brief Examine an edge.
         *
         * The edge will be examined by the adapted visitor if
         * it belongs to the right aggregate.
         */
        void operator()(const typename MatrixGraph::ConstEdgeIterator& edge);

      private:
        /** @brief Mapping of vertices to aggregates. */
        const AggregatesMap<Vertex>& aggregates_;
        /** @brief The aggregate id we want to visit. */
        AggregateDescriptor aggregate_;
        /** @brief The visitor to use on the aggregate. */
        Visitor* visitor_;
      };

      /**
       * @brief A simple counter functor.
       */
      class Counter
      {
      public:
        /** @brief Constructor */
        Counter();
        /** @brief Access the current count. */
        int value();

      protected:
        /** @brief Increment counter */
        void increment();
        /** @brief Decrement counter */
        void decrement();

      private:
        int count_;
      };


      /**
       * @brief Counts the number of edges to vertices belonging
       * to the aggregate front.
       */
      class FrontNeighbourCounter : public Counter
      {
      public:
        /**
         * @brief Constructor.
         * @param front The vertices of the front.
         */
        FrontNeighbourCounter(const MatrixGraph& front);

        void operator()(const typename MatrixGraph::ConstEdgeIterator& edge);

      private:
        const MatrixGraph& graph_;
      };

      /**
       * @brief Count the number of neighbours of a vertex that belong
       * to the aggregate front.
       */
      int noFrontNeighbours(const Vertex& vertex) const;

      /**
       * @brief Counter of TwoWayConnections.
       */
      class TwoWayCounter : public Counter
      {
      public:
        void operator()(const typename MatrixGraph::ConstEdgeIterator& edge);
      };

      /**
       * @brief Count the number of twoway connection from
       * a vertex to an aggregate.
       *
       * @param vertex The vertex whose connections are counted.
       * @param aggregate The id of the aggregate the connections
       * should point to.
       * @param aggregates The mapping of the vertices onto aggregates.
       * @return The number of one way connections from the vertex to
       * the aggregate.
       */
      int twoWayConnections(const Vertex&, const AggregateDescriptor& aggregate,
                            const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief Counter of OneWayConnections.
       */
      class OneWayCounter : public Counter
      {
      public:
        void operator()(const typename MatrixGraph::ConstEdgeIterator& edge);
      };

      /**
       * @brief Count the number of oneway connection from
       * a vertex to an aggregate.
       *
       * @param vertex The vertex whose connections are counted.
       * @param aggregate The id of the aggregate the connections
       * should point to.
       * @param aggregates The mapping of the vertices onto aggregates.
       * @return The number of one way connections from the vertex to
       * the aggregate.
       */
      int oneWayConnections(const Vertex&, const AggregateDescriptor& aggregate,
                            const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief Connectivity counter
       *
       * Increments count if the neighbour is already known as
       * connected or is not yet aggregated.
       */
      class ConnectivityCounter : public Counter
      {
      public:
        /**
         * @brief Constructor.
         * @param connected The set of connected aggregates.
         * @param aggregates Mapping of the vertices onto the aggregates.
         * @param aggregates The mapping of aggregates to vertices.
         */
        ConnectivityCounter(const VertexSet& connected, const AggregatesMap<Vertex>& aggregates);

        void operator()(const typename MatrixGraph::ConstEdgeIterator& edge);

      private:
        /** @brief The connected aggregates. */
        const VertexSet& connected_;
        /** @brief The mapping of vertices to aggregates. */
        const AggregatesMap<Vertex>& aggregates_;

      };

      /**
       * @brief Get the connectivity of a vertex.
       *
       * For each unaggregated neighbour or neighbour of an aggregate
       * that is already known as connected the count is increased by
       * one. In all other cases by two.
       *
       * @param vertex The vertex whose connectivity we want.
       * @param aggregates The mapping of the vertices onto the aggregates.
       * @return The value of the connectivity.
       */
      double connectivity(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates) const;
      /**
       * @brief Test whether the vertex is connected to the aggregate.
       * @param vertex The vertex descriptor.
       * @param aggregate The aggregate descriptor.
       * @param aggregates The mapping of the vertices onto the aggregates.
       * @return True if there is a connection to the aggregate.
       */
      bool connected(const Vertex& vertex, const AggregateDescriptor& aggregate,
                     const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief Test whether the vertex is connected to an aggregate of a list.
       * @param vertex The vertex descriptor.
       * @param aggregateList The list of aggregate descriptors.
       * @param aggregates The mapping of the vertices onto the aggregates.
       * @return True if there is a connection to the aggregate.
       */
      bool connected(const Vertex& vertex, const SLList<AggregateDescriptor>& aggregateList,
                     const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief Counts the edges depending on the dependency.
       *
       * If the inluence flag of the edge is set the counter is
       * increased and/or if the depends flag is set it is
       * incremented, too.
       */
      class DependencyCounter : public Counter
      {
      public:
        /**
         * @brief Constructor.
         */
        DependencyCounter();

        void operator()(const typename MatrixGraph::ConstEdgeIterator& edge);
      };

      /**
       * @brief Adds the targets of each edge to
       * the list of front vertices.
       *
       * Vertices already marked as front nodes will not get added.
       */
      class FrontMarker
      {
      public:
        /**
         * @brief Constructor.
         *
         * @param front The list to store the front vertices in.
         * @param graph The matrix graph we work on.
         */
        FrontMarker(std::vector<Vertex>& front, MatrixGraph& graph);

        void operator()(const typename MatrixGraph::ConstEdgeIterator& edge);

      private:
        /** @brief The list of front vertices. */
        std::vector<Vertex>& front_;
        /** @brief The matrix graph we work on. */
        MatrixGraph& graph_;
      };

      /**
       * @brief Unmarks all front vertices.
       */
      void unmarkFront();

      /**
       * @brief counts the dependency between a vertex and unaggregated
       * neighbours.
       *
       * If the inluence flag of the edge is set the counter is
       * increased and/or if the depends flag is set it is
       * incremented, too.
       *
       * @param vertex The vertex whose neighbours we count.
       * @param aggregates The mapping of the vertices onto the aggregates.
       * @return The sum of the number of unaggregated
       * neighbours the vertex depends on and the number of unaggregated
       * neighbours the vertex influences.
       */
      int unusedNeighbours(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief Count connections to neighbours.
       *
       * Counts the number of strong connections of a vertex to vertices
       * that are not yet aggregated
       * and the ones that belong to specific aggregate.
       *
       * @param vertex The vertex that we count the neighbours of.
       * @param aggregates The mapping of the vertices into aggregates.
       * @param aggregate The descriptor of the aggregate.
       * @return The pair of number of connections to unaggregate vertices
       * and number of connections to vertices of the specific aggregate.
       */
      std::pair<int,int> neighbours(const Vertex& vertex,
                                    const AggregateDescriptor& aggregate,
                                    const AggregatesMap<Vertex>& aggregates) const;
      /**
       * @brief Counts the number of neighbours belonging to an aggregate.
       *
       *
       * If the inluence flag of the edge is set the counter is
       * increased and/or if the depends flag is set it is
       * incremented, too.
       *
       * @param vertex The vertex whose neighbours we count.
       * @param aggregate The aggregate id.
       * @param aggregates The mapping of the vertices onto the aggregates.
       * @return The sum of the number of
       * neighbours belonging to the aggregate
       * the vertex depends on and the number of
       * neighbours of the aggregate the vertex influences.
       */
      int aggregateNeighbours(const Vertex& vertex, const AggregateDescriptor& aggregate, const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief Checks wether a vertex is admisible to be added to an aggregate.
       *
       * @param vertex The vertex whose admissibility id to be checked.
       * @param aggregate The id of the aggregate.
       * @param aggregates The mapping of the vertices onto aggregates.
       */
      bool admissible(const Vertex& vertex, const AggregateDescriptor& aggregate, const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief The maximum distance of the vertex to any vertex in the
       * current aggregate.
       *
       * @return The maximum of all shortest paths from the vertex to any
       * vertex of the aggregate.
       */
      std::size_t distance(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates);

      /**
       * @brief Find a strongly connected cluster of a vertex.
       *
       * @param vertex The vertex whose neighbouring aggregate we search.
       * @param aggregates The mapping of the vertices onto aggregates.
       * @return A vertex of neighbouring aggregate the vertex is allowed to
       * be added to.
       */
      Vertex mergeNeighbour(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates) const;

      /**
       * @brief Find a nonisolated connected aggregate.
       *
       * @param vertex The vertex whose neighbouring aggregate we search.
       * @param aggregates The mapping of the vertices onto aggregates.
       * @param[out] list to store the vertices of neighbouring aggregates the vertex is allowed to
       * be added to.
       */
      void nonisoNeighbourAggregate(const Vertex& vertex,
                                    const AggregatesMap<Vertex>& aggregates,
                                    SLList<Vertex>& neighbours) const;

      /**
       * @brief Grows the aggregate from a seed.
       *
       * @param seed The first vertex of the aggregate.
       * @param aggregates The mapping of he vertices onto the aggregates.
       * @param c The coarsen criterium.
       */
      template<class C>
      void growAggregate(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates, const C& c);
      template<class C>
      void growIsolatedAggregate(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates, const C& c);
    };

#ifndef DOXYGEN

    template<class M, class N>
    inline void SymmetricDependency<M,N>::init(const Matrix* matrix)
    {
      matrix_ = matrix;
    }

    template<class M, class N>
    inline void SymmetricDependency<M,N>::initRow(const Row& row, int index)
    {
      initRow(row, index, std::is_convertible<field_type, real_type>());
    }

    template<class M, class N>
    inline void SymmetricDependency<M,N>::initRow(const Row& row, int index, const std::false_type&)
    {
      DUNE_THROW(InvalidStateException, "field_type needs to convertible to real_type");
    }

    template<class M, class N>
    inline void SymmetricDependency<M,N>::initRow([[maybe_unused]] const Row& row, int index, const std::true_type&)
    {
      using std::min;
      maxValue_ = min(- std::numeric_limits<typename Matrix::field_type>::max(), std::numeric_limits<typename Matrix::field_type>::min());
      row_ = index;
      diagonal_ = norm_(matrix_->operator[](row_)[row_]);
    }

    template<class M, class N>
    inline void SymmetricDependency<M,N>::examine(const ColIter& col)
    {
      using std::max;
      real_type eij = norm_(*col);
      typename Matrix::ConstColIterator opposite_entry =
        matrix_->operator[](col.index()).find(row_);
      if ( opposite_entry == matrix_->operator[](col.index()).end() )
      {
        // Consider this a weak connection we disregard.
        return;
      }
      real_type eji = norm_(*opposite_entry);

      // skip positive offdiagonals if norm preserves sign of them.
      if(!N::is_sign_preserving || eij<0 || eji<0)
        maxValue_ = max(maxValue_,
                             eij /diagonal_ * eji/
                             norm_(matrix_->operator[](col.index())[col.index()]));
    }

    template<class M, class N>
    template<class G>
    inline void SymmetricDependency<M,N>::examine(G& graph, const typename G::EdgeIterator& edge, const ColIter& col)
    {
      real_type eij = norm_(*col);
      typename Matrix::ConstColIterator opposite_entry =
        matrix_->operator[](col.index()).find(row_);

      if ( opposite_entry == matrix_->operator[](col.index()).end() )
      {
        // Consider this as a weak connection we disregard.
        return;
      }
      real_type eji = norm_(*opposite_entry);
      // skip positve offdiagonals if norm preserves sign of them.
      if(!N::is_sign_preserving || (eij<0 || eji<0))
        if(eji / norm_(matrix_->operator[](edge.target())[edge.target()]) *
           eij/ diagonal_ > alpha() * maxValue_) {
          edge.properties().setDepends();
          edge.properties().setInfluences();
          typename G::EdgeProperties& other = graph.getEdgeProperties(edge.target(), edge.source());
          other.setInfluences();
          other.setDepends();
        }
    }

    template<class M, class N>
    inline bool SymmetricDependency<M,N>::isIsolated()
    {
      return maxValue_  < beta();
    }


    template<class M, class N>
    inline void Dependency<M,N>::init(const Matrix* matrix)
    {
      matrix_ = matrix;
    }

    template<class M, class N>
    inline void Dependency<M,N>::initRow([[maybe_unused]] const Row& row, int index)
    {
      using std::min;
      maxValue_ = min(- std::numeric_limits<real_type>::max(), std::numeric_limits<real_type>::min());
      row_ = index;
      diagonal_ = norm_(matrix_->operator[](row_)[row_]);
    }

    template<class M, class N>
    inline void Dependency<M,N>::examine(const ColIter& col)
    {
      using std::max;
      maxValue_ = max(maxValue_, -norm_(*col));
    }

    template<class M, class N>
    template<class G>
    inline void Dependency<M,N>::examine(G& graph, const typename G::EdgeIterator& edge, const ColIter& col)
    {
      if(-norm_(*col) >= maxValue_ * alpha()) {
        edge.properties().setDepends();
        typedef typename G::EdgeDescriptor ED;
        ED e= graph.findEdge(edge.target(), edge.source());
        if(e!=std::numeric_limits<ED>::max())
        {
          typename G::EdgeProperties& other = graph.getEdgeProperties(e);
          other.setInfluences();
        }
      }
    }

    template<class M, class N>
    inline bool Dependency<M,N>::isIsolated()
    {
      return maxValue_  < beta() * diagonal_;
    }

    template<class G,class S>
    Aggregate<G,S>::Aggregate(MatrixGraph& graph, AggregatesMap<Vertex>& aggregates,
                              VertexSet& connected, std::vector<Vertex>& front)
      : vertices_(), id_(-1), graph_(graph), aggregates_(aggregates),
        connected_(connected), front_(front)
    {}

    template<class G,class S>
    void Aggregate<G,S>::reconstruct(const Vertex& vertex)
    {
      /*
         vertices_.push_back(vertex);
         typedef typename VertexList::const_iterator iterator;
         iterator begin = vertices_.begin();
         iterator end   = vertices_.end();*/
      throw "Not yet implemented";

      //      while(begin!=end){
      //for();
      //      }

    }

    template<class G,class S>
    inline void Aggregate<G,S>::seed(const Vertex& vertex)
    {
      dvverb<<"Connected cleared"<<std::endl;
      connected_.clear();
      vertices_.clear();
      connected_.insert(vertex);
      dvverb << " Inserting "<<vertex<<" size="<<connected_.size();
      ++id_ ;
      add(vertex);
    }


    template<class G,class S>
    inline void Aggregate<G,S>::add(const Vertex& vertex)
    {
      vertices_.insert(vertex);
      aggregates_[vertex]=id_;
      if(front_.size())
        front_.erase(std::lower_bound(front_.begin(), front_.end(), vertex));


      typedef typename MatrixGraph::ConstEdgeIterator iterator;
      const iterator end = graph_.endEdges(vertex);
      for(iterator edge = graph_.beginEdges(vertex); edge != end; ++edge) {
        dvverb << " Inserting "<<aggregates_[edge.target()];
        connected_.insert(aggregates_[edge.target()]);
        dvverb <<" size="<<connected_.size();
        if(aggregates_[edge.target()]==AggregatesMap<Vertex>::UNAGGREGATED &&
           !graph_.getVertexProperties(edge.target()).front())
        {
          front_.push_back(edge.target());
          graph_.getVertexProperties(edge.target()).setFront();
        }
      }
      dvverb <<std::endl;
      std::sort(front_.begin(), front_.end());
    }

    template<class G,class S>
    inline void Aggregate<G,S>::add(std::vector<Vertex>& vertices)
    {
#ifndef NDEBUG
      std::size_t oldsize = vertices_.size();
#endif
      typedef typename std::vector<Vertex>::iterator Iterator;

      typedef typename VertexSet::iterator SIterator;

      SIterator pos=vertices_.begin();
      std::vector<Vertex> newFront;
      newFront.reserve(front_.capacity());

      std::set_difference(front_.begin(), front_.end(), vertices.begin(), vertices.end(),
                          std::back_inserter(newFront));
      front_=newFront;

      for(Iterator vertex=vertices.begin(); vertex != vertices.end(); ++vertex)
      {
        pos=vertices_.insert(pos,*vertex);
        vertices_.insert(*vertex);
        graph_.getVertexProperties(*vertex).resetFront(); // Not a front node any more.
        aggregates_[*vertex]=id_;

        typedef typename MatrixGraph::ConstEdgeIterator iterator;
        const iterator end = graph_.endEdges(*vertex);
        for(iterator edge = graph_.beginEdges(*vertex); edge != end; ++edge) {
          dvverb << " Inserting "<<aggregates_[edge.target()];
          connected_.insert(aggregates_[edge.target()]);
          if(aggregates_[edge.target()]==AggregatesMap<Vertex>::UNAGGREGATED &&
             !graph_.getVertexProperties(edge.target()).front())
          {
            front_.push_back(edge.target());
            graph_.getVertexProperties(edge.target()).setFront();
          }
          dvverb <<" size="<<connected_.size();
        }
        dvverb <<std::endl;
      }
      std::sort(front_.begin(), front_.end());
      assert(oldsize+vertices.size()==vertices_.size());
    }
    template<class G,class S>
    inline void Aggregate<G,S>::clear()
    {
      vertices_.clear();
      connected_.clear();
      id_=-1;
    }

    template<class G,class S>
    inline typename Aggregate<G,S>::VertexSet::size_type
    Aggregate<G,S>::size()
    {
      return vertices_.size();
    }

    template<class G,class S>
    inline typename Aggregate<G,S>::VertexSet::size_type
    Aggregate<G,S>::connectSize()
    {
      return connected_.size();
    }

    template<class G,class S>
    inline int Aggregate<G,S>::id()
    {
      return id_;
    }

    template<class G,class S>
    inline typename Aggregate<G,S>::const_iterator Aggregate<G,S>::begin() const
    {
      return vertices_.begin();
    }

    template<class G,class S>
    inline typename Aggregate<G,S>::const_iterator Aggregate<G,S>::end() const
    {
      return vertices_.end();
    }

    template<class V>
    const V AggregatesMap<V>::UNAGGREGATED = std::numeric_limits<V>::max();

    template<class V>
    const V AggregatesMap<V>::ISOLATED = std::numeric_limits<V>::max()-1;

    template<class V>
    AggregatesMap<V>::AggregatesMap()
      : aggregates_(0)
    {}

    template<class V>
    AggregatesMap<V>::~AggregatesMap()
    {
      if(aggregates_!=0)
        delete[] aggregates_;
    }


    template<class V>
    inline AggregatesMap<V>::AggregatesMap(std::size_t noVertices)
    {
      allocate(noVertices);
    }

    template<class V>
    inline std::size_t AggregatesMap<V>::noVertices() const
    {
      return noVertices_;
    }

    template<class V>
    inline void AggregatesMap<V>::allocate(std::size_t noVertices)
    {
      aggregates_ = new AggregateDescriptor[noVertices];
      noVertices_ = noVertices;

      for(std::size_t i=0; i < noVertices; i++)
        aggregates_[i]=UNAGGREGATED;
    }

    template<class V>
    inline void AggregatesMap<V>::free()
    {
      assert(aggregates_ != 0);
      delete[] aggregates_;
      aggregates_=0;
    }

    template<class V>
    inline typename AggregatesMap<V>::AggregateDescriptor&
    AggregatesMap<V>::operator[](const VertexDescriptor& v)
    {
      return aggregates_[v];
    }

    template<class V>
    inline const typename AggregatesMap<V>::AggregateDescriptor&
    AggregatesMap<V>::operator[](const VertexDescriptor& v) const
    {
      return aggregates_[v];
    }

    template<class V>
    template<bool reset, class G, class F,class VM>
    inline std::size_t AggregatesMap<V>::breadthFirstSearch(const V& start,
                                                            const AggregateDescriptor& aggregate,
                                                            const G& graph, F& aggregateVisitor,
                                                            VM& visitedMap) const
    {
      VertexList vlist;

      DummyEdgeVisitor dummy;
      return breadthFirstSearch<true,reset>(start, aggregate, graph, vlist, aggregateVisitor, dummy, visitedMap);
    }

    template<class V>
    template<bool remove, bool reset, class G, class L, class F1, class F2, class VM>
    std::size_t AggregatesMap<V>::breadthFirstSearch(const V& start,
                                                     const AggregateDescriptor& aggregate,
                                                     const G& graph,
                                                     L& visited,
                                                     F1& aggregateVisitor,
                                                     F2& nonAggregateVisitor,
                                                     VM& visitedMap) const
    {
      typedef typename L::const_iterator ListIterator;
      int visitedSpheres = 0;

      visited.push_back(start);
      put(visitedMap, start, true);

      ListIterator current = visited.begin();
      ListIterator end = visited.end();
      std::size_t i=0, size=visited.size();

      // visit the neighbours of all vertices of the
      // current sphere.
      while(current != end) {

        for(; i<size; ++current, ++i) {
          typedef typename G::ConstEdgeIterator EdgeIterator;
          const EdgeIterator endEdge = graph.endEdges(*current);

          for(EdgeIterator edge = graph.beginEdges(*current);
              edge != endEdge; ++edge) {

            if(aggregates_[edge.target()]==aggregate) {
              if(!get(visitedMap, edge.target())) {
                put(visitedMap, edge.target(), true);
                visited.push_back(edge.target());
                aggregateVisitor(edge);
              }
            }else
              nonAggregateVisitor(edge);
          }
        }
        end = visited.end();
        size = visited.size();
        if(current != end)
          visitedSpheres++;
      }

      if(reset)
        for(current = visited.begin(); current != end; ++current)
          put(visitedMap, *current, false);


      if(remove)
        visited.clear();

      return visitedSpheres;
    }

    template<class G>
    Aggregator<G>::Aggregator()
      : graph_(0), aggregate_(0), front_(), connected_(), size_(-1)
    {}

    template<class G>
    Aggregator<G>::~Aggregator()
    {
      size_=-1;
    }

    template<class G, class C>
    void buildDependency(G& graph,
                         const typename C::Matrix& matrix,
                         C criterion, bool firstlevel)
    {
      //      assert(graph.isBuilt());
      typedef typename C::Matrix Matrix;
      typedef typename G::VertexIterator VertexIterator;

      criterion.init(&matrix);

      for(VertexIterator vertex = graph.begin(); vertex != graph.end(); ++vertex) {
        typedef typename Matrix::row_type Row;

        const Row& row = matrix[*vertex];

        // Tell the criterion what row we will examine now
        // This might for example be used for calculating the
        // maximum offdiagonal value
        criterion.initRow(row, *vertex);

        // On a first path all columns are examined. After this
        // the calculator should know whether the vertex is isolated.
        typedef typename Matrix::ConstColIterator ColIterator;
        ColIterator end = row.end();
        typename FieldTraits<typename Matrix::field_type>::real_type absoffdiag=0.;

        using std::max;
        if(firstlevel) {
          for(ColIterator col = row.begin(); col != end; ++col)
            if(col.index()!=*vertex) {
              criterion.examine(col);
              absoffdiag = max(absoffdiag, Impl::asMatrix(*col).frobenius_norm());
            }

          if(absoffdiag==0)
            vertex.properties().setExcludedBorder();
        }
        else
          for(ColIterator col = row.begin(); col != end; ++col)
            if(col.index()!=*vertex)
              criterion.examine(col);

        // reset the vertex properties
        //vertex.properties().reset();

        // Check whether the vertex is isolated.
        if(criterion.isIsolated()) {
          //std::cout<<"ISOLATED: "<<*vertex<<std::endl;
          vertex.properties().setIsolated();
        }else{
          // Examine all the edges beginning at this vertex.
          auto eEnd = vertex.end();
          auto col = matrix[*vertex].begin();

          for(auto edge = vertex.begin(); edge!= eEnd; ++edge, ++col) {
            // Move to the right column.
            while(col.index()!=edge.target())
              ++col;
            criterion.examine(graph, edge, col);
          }
        }

      }
    }


    template<class G>
    template<class V>
    inline Aggregator<G>::AggregateVisitor<V>::AggregateVisitor(const AggregatesMap<Vertex>& aggregates,
                                                                const AggregateDescriptor& aggregate, V& visitor)
      : aggregates_(aggregates), aggregate_(aggregate), visitor_(&visitor)
    {}

    template<class G>
    template<class V>
    inline void Aggregator<G>::AggregateVisitor<V>::operator()(const typename MatrixGraph::ConstEdgeIterator& edge)
    {
      if(aggregates_[edge.target()]==aggregate_)
        visitor_->operator()(edge);
    }

    template<class G>
    template<class V>
    inline void Aggregator<G>::visitAggregateNeighbours(const Vertex& vertex,
                                                        const AggregateDescriptor& aggregate,
                                                        const AggregatesMap<Vertex>& aggregates,
                                                        V& visitor) const
    {
      // Only evaluates for edge pointing to the aggregate
      AggregateVisitor<V> v(aggregates, aggregate, visitor);
      visitNeighbours(*graph_, vertex, v);
    }


    template<class G>
    inline Aggregator<G>::Counter::Counter()
      : count_(0)
    {}

    template<class G>
    inline void Aggregator<G>::Counter::increment()
    {
      ++count_;
    }

    template<class G>
    inline void Aggregator<G>::Counter::decrement()
    {
      --count_;
    }
    template<class G>
    inline int Aggregator<G>::Counter::value()
    {
      return count_;
    }

    template<class G>
    inline void Aggregator<G>::TwoWayCounter::operator()(const typename MatrixGraph::ConstEdgeIterator& edge)
    {
      if(edge.properties().isTwoWay())
        Counter::increment();
    }

    template<class G>
    int Aggregator<G>::twoWayConnections(const Vertex& vertex, const AggregateDescriptor& aggregate,
                                         const AggregatesMap<Vertex>& aggregates) const
    {
      TwoWayCounter counter;
      visitAggregateNeighbours(vertex, aggregate, aggregates, counter);
      return counter.value();
    }

    template<class G>
    int Aggregator<G>::oneWayConnections(const Vertex& vertex, const AggregateDescriptor& aggregate,
                                         const AggregatesMap<Vertex>& aggregates) const
    {
      OneWayCounter counter;
      visitAggregateNeighbours(vertex, aggregate, aggregates, counter);
      return counter.value();
    }

    template<class G>
    inline void Aggregator<G>::OneWayCounter::operator()(const typename MatrixGraph::ConstEdgeIterator& edge)
    {
      if(edge.properties().isOneWay())
        Counter::increment();
    }

    template<class G>
    inline Aggregator<G>::ConnectivityCounter::ConnectivityCounter(const VertexSet& connected,
                                                                   const AggregatesMap<Vertex>& aggregates)
      : Counter(), connected_(connected), aggregates_(aggregates)
    {}


    template<class G>
    inline void Aggregator<G>::ConnectivityCounter::operator()(const typename MatrixGraph::ConstEdgeIterator& edge)
    {
      if(connected_.find(aggregates_[edge.target()]) == connected_.end() || aggregates_[edge.target()]==AggregatesMap<Vertex>::UNAGGREGATED)
        // Would be a new connection
        Counter::increment();
      else{
        Counter::increment();
        Counter::increment();
      }
    }

    template<class G>
    inline double Aggregator<G>::connectivity(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates) const
    {
      ConnectivityCounter counter(connected_, aggregates);
      double noNeighbours=visitNeighbours(*graph_, vertex, counter);
      return (double)counter.value()/noNeighbours;
    }

    template<class G>
    inline Aggregator<G>::DependencyCounter::DependencyCounter()
      : Counter()
    {}

    template<class G>
    inline void Aggregator<G>::DependencyCounter::operator()(const typename MatrixGraph::ConstEdgeIterator& edge)
    {
      if(edge.properties().depends())
        Counter::increment();
      if(edge.properties().influences())
        Counter::increment();
    }

    template<class G>
    int Aggregator<G>::unusedNeighbours(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates) const
    {
      return aggregateNeighbours(vertex, AggregatesMap<Vertex>::UNAGGREGATED, aggregates);
    }

    template<class G>
    std::pair<int,int> Aggregator<G>::neighbours(const Vertex& vertex,
                                                 const AggregateDescriptor& aggregate,
                                                 const AggregatesMap<Vertex>& aggregates) const
    {
      DependencyCounter unused, aggregated;
      typedef AggregateVisitor<DependencyCounter> CounterT;
      typedef std::tuple<CounterT,CounterT> CounterTuple;
      CombinedFunctor<CounterTuple> visitors(CounterTuple(CounterT(aggregates, AggregatesMap<Vertex>::UNAGGREGATED, unused), CounterT(aggregates, aggregate, aggregated)));
      visitNeighbours(*graph_, vertex, visitors);
      return std::make_pair(unused.value(), aggregated.value());
    }


    template<class G>
    int Aggregator<G>::aggregateNeighbours(const Vertex& vertex, const AggregateDescriptor& aggregate, const AggregatesMap<Vertex>& aggregates) const
    {
      DependencyCounter counter;
      visitAggregateNeighbours(vertex, aggregate, aggregates, counter);
      return counter.value();
    }

    template<class G>
    std::size_t Aggregator<G>::distance(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates)
    {
      return 0;
      typename PropertyMapTypeSelector<VertexVisitedTag,G>::Type visitedMap = get(VertexVisitedTag(), *graph_);
      VertexList vlist;
      typename AggregatesMap<Vertex>::DummyEdgeVisitor dummy;
      return aggregates.template breadthFirstSearch<true,true>(vertex,
                                                               aggregate_->id(), *graph_,
                                                               vlist, dummy, dummy, visitedMap);
    }

    template<class G>
    inline Aggregator<G>::FrontMarker::FrontMarker(std::vector<Vertex>& front, MatrixGraph& graph)
      : front_(front), graph_(graph)
    {}

    template<class G>
    inline void Aggregator<G>::FrontMarker::operator()(const typename MatrixGraph::ConstEdgeIterator& edge)
    {
      Vertex target = edge.target();

      if(!graph_.getVertexProperties(target).front()) {
        front_.push_back(target);
        graph_.getVertexProperties(target).setFront();
      }
    }

    template<class G>
    inline bool Aggregator<G>::admissible(const Vertex& vertex, const AggregateDescriptor& aggregate, const AggregatesMap<Vertex>& aggregates) const
    {
      // Todo
      Dune::dvverb<<" Admissible not yet implemented!"<<std::endl;
      return true;
      //Situation 1: front node depends on two nodes. Then these
      // have to be strongly connected to each other

      // Iterate over all all neighbours of front node
      typedef typename MatrixGraph::ConstEdgeIterator Iterator;
      Iterator vend = graph_->endEdges(vertex);
      for(Iterator edge = graph_->beginEdges(vertex); edge != vend; ++edge) {
        // if(edge.properties().depends() && !edge.properties().influences()
        if(edge.properties().isStrong()
           && aggregates[edge.target()]==aggregate)
        {
          // Search for another link to the aggregate
          Iterator edge1 = edge;
          for(++edge1; edge1 != vend; ++edge1) {
            //if(edge1.properties().depends() && !edge1.properties().influences()
            if(edge1.properties().isStrong()
               && aggregates[edge.target()]==aggregate)
            {
              //Search for an edge connecting the two vertices that is
              //strong
              bool found=false;
              Iterator v2end = graph_->endEdges(edge.target());
              for(Iterator edge2 = graph_->beginEdges(edge.target()); edge2 != v2end; ++edge2) {
                if(edge2.target()==edge1.target() &&
                   edge2.properties().isStrong()) {
                  found =true;
                  break;
                }
              }
              if(found)
              {
                return true;
              }
            }
          }
        }
      }

      // Situation 2: cluster node depends on front node and other cluster node
      /// Iterate over all all neighbours of front node
      vend = graph_->endEdges(vertex);
      for(Iterator edge = graph_->beginEdges(vertex); edge != vend; ++edge) {
        //if(!edge.properties().depends() && edge.properties().influences()
        if(edge.properties().isStrong()
           && aggregates[edge.target()]==aggregate)
        {
          // Search for a link from target that stays within the aggregate
          Iterator v1end = graph_->endEdges(edge.target());

          for(Iterator edge1=graph_->beginEdges(edge.target()); edge1 != v1end; ++edge1) {
            //if(edge1.properties().depends() && !edge1.properties().influences()
            if(edge1.properties().isStrong()
               && aggregates[edge1.target()]==aggregate)
            {
              bool found=false;
              // Check if front node is also connected to this one
              Iterator v2end = graph_->endEdges(vertex);
              for(Iterator edge2 = graph_->beginEdges(vertex); edge2 != v2end; ++edge2) {
                if(edge2.target()==edge1.target()) {
                  if(edge2.properties().isStrong())
                    found=true;
                  break;
                }
              }
              if(found)
              {
                return true;
              }
            }
          }
        }
      }
      return false;
    }

    template<class G>
    void Aggregator<G>::unmarkFront()
    {
      typedef typename std::vector<Vertex>::const_iterator Iterator;

      for(Iterator vertex=front_.begin(); vertex != front_.end(); ++vertex)
        graph_->getVertexProperties(*vertex).resetFront();

      front_.clear();
    }

    template<class G>
    inline void
    Aggregator<G>::nonisoNeighbourAggregate(const Vertex& vertex,
                                            const AggregatesMap<Vertex>& aggregates,
                                            SLList<Vertex>& neighbours) const
    {
      typedef typename MatrixGraph::ConstEdgeIterator Iterator;
      Iterator end=graph_->beginEdges(vertex);
      neighbours.clear();

      for(Iterator edge=graph_->beginEdges(vertex); edge!=end; ++edge)
      {
        if(aggregates[edge.target()]!=AggregatesMap<Vertex>::UNAGGREGATED && graph_->getVertexProperties(edge.target()).isolated())
          neighbours.push_back(aggregates[edge.target()]);
      }
    }

    template<class G>
    inline typename G::VertexDescriptor Aggregator<G>::mergeNeighbour(const Vertex& vertex, const AggregatesMap<Vertex>& aggregates) const
    {
      typedef typename MatrixGraph::ConstEdgeIterator Iterator;

      Iterator end = graph_->endEdges(vertex);
      for(Iterator edge = graph_->beginEdges(vertex); edge != end; ++edge) {
        if(aggregates[edge.target()] != AggregatesMap<Vertex>::UNAGGREGATED &&
           graph_->getVertexProperties(edge.target()).isolated() == graph_->getVertexProperties(edge.source()).isolated()) {
          if( graph_->getVertexProperties(vertex).isolated() ||
              ((edge.properties().depends() || edge.properties().influences())
               && admissible(vertex, aggregates[edge.target()], aggregates)))
            return edge.target();
        }
      }
      return AggregatesMap<Vertex>::UNAGGREGATED;
    }

    template<class G>
    Aggregator<G>::FrontNeighbourCounter::FrontNeighbourCounter(const MatrixGraph& graph)
      : Counter(), graph_(graph)
    {}

    template<class G>
    void Aggregator<G>::FrontNeighbourCounter::operator()(const typename MatrixGraph::ConstEdgeIterator& edge)
    {
      if(graph_.getVertexProperties(edge.target()).front())
        Counter::increment();
    }

    template<class G>
    int Aggregator<G>::noFrontNeighbours(const Vertex& vertex) const
    {
      FrontNeighbourCounter counter(*graph_);
      visitNeighbours(*graph_, vertex, counter);
      return counter.value();
    }
    template<class G>
    inline bool Aggregator<G>::connected(const Vertex& vertex,
                                         const AggregateDescriptor& aggregate,
                                         const AggregatesMap<Vertex>& aggregates) const
    {
      typedef typename G::ConstEdgeIterator iterator;
      const iterator end = graph_->endEdges(vertex);
      for(iterator edge = graph_->beginEdges(vertex); edge != end; ++edge)
        if(aggregates[edge.target()]==aggregate)
          return true;
      return false;
    }
    template<class G>
    inline bool Aggregator<G>::connected(const Vertex& vertex,
                                         const SLList<AggregateDescriptor>& aggregateList,
                                         const AggregatesMap<Vertex>& aggregates) const
    {
      typedef typename SLList<AggregateDescriptor>::const_iterator Iter;
      for(Iter i=aggregateList.begin(); i!=aggregateList.end(); ++i)
        if(connected(vertex, *i, aggregates))
          return true;
      return false;
    }

    template<class G>
    template<class C>
    void Aggregator<G>::growIsolatedAggregate(const Vertex& seed, const AggregatesMap<Vertex>& aggregates, const C& c)
    {
      SLList<Vertex> connectedAggregates;
      nonisoNeighbourAggregate(seed, aggregates,connectedAggregates);

      while(aggregate_->size()< c.minAggregateSize() && aggregate_->connectSize() < c.maxConnectivity()) {
        double maxCon=-1;
        std::size_t maxFrontNeighbours=0;

        Vertex candidate=AggregatesMap<Vertex>::UNAGGREGATED;

        typedef typename std::vector<Vertex>::const_iterator Iterator;

        for(Iterator vertex = front_.begin(); vertex != front_.end(); ++vertex) {
          if(distance(*vertex, aggregates)>c.maxDistance())
            continue; // distance of proposes aggregate too big

          if(connectedAggregates.size()>0) {
            // there is already a neighbour cluster
            // front node must be connected to same neighbour cluster

            if(!connected(*vertex, connectedAggregates, aggregates))
              continue;
          }

          double con = connectivity(*vertex, aggregates);

          if(con == maxCon) {
            std::size_t frontNeighbours = noFrontNeighbours(*vertex);

            if(frontNeighbours >= maxFrontNeighbours) {
              maxFrontNeighbours = frontNeighbours;
              candidate = *vertex;
            }
          }else if(con > maxCon) {
            maxCon = con;
            maxFrontNeighbours = noFrontNeighbours(*vertex);
            candidate = *vertex;
          }
        }

        if(candidate==AggregatesMap<Vertex>::UNAGGREGATED)
          break;

        aggregate_->add(candidate);
      }
    }

    template<class G>
    template<class C>
    void Aggregator<G>::growAggregate(const Vertex& seed, const AggregatesMap<Vertex>& aggregates, const C& c)
    {
      using std::min;

      std::size_t distance_ =0;
      while(aggregate_->size() < c.minAggregateSize()&& distance_<c.maxDistance()) {
        int maxTwoCons=0, maxOneCons=0, maxNeighbours=-1;
        double maxCon=-1;

        std::vector<Vertex> candidates;
        candidates.reserve(30);

        typedef typename std::vector<Vertex>::const_iterator Iterator;

        for(Iterator vertex = front_.begin(); vertex != front_.end(); ++vertex) {
          // Only nonisolated nodes are considered
          if(graph_->getVertexProperties(*vertex).isolated())
            continue;

          int twoWayCons = twoWayConnections(*vertex, aggregate_->id(), aggregates);

          /* The case of two way connections. */
          if( maxTwoCons == twoWayCons && twoWayCons > 0) {
            double con = connectivity(*vertex, aggregates);

            if(con == maxCon) {
              int neighbours = noFrontNeighbours(*vertex);

              if(neighbours > maxNeighbours) {
                maxNeighbours = neighbours;
                candidates.clear();
                candidates.push_back(*vertex);
              }else{
                candidates.push_back(*vertex);
              }
            }else if( con > maxCon) {
              maxCon = con;
              maxNeighbours = noFrontNeighbours(*vertex);
              candidates.clear();
              candidates.push_back(*vertex);
            }
          }else if(twoWayCons > maxTwoCons) {
            maxTwoCons = twoWayCons;
            maxCon = connectivity(*vertex, aggregates);
            maxNeighbours = noFrontNeighbours(*vertex);
            candidates.clear();
            candidates.push_back(*vertex);

            // two way connections precede
            maxOneCons = std::numeric_limits<int>::max();
          }

          if(twoWayCons > 0)
          {
            continue; // THis is a two-way node, skip tests for one way nodes
          }

          /* The one way case */
          int oneWayCons = oneWayConnections(*vertex, aggregate_->id(), aggregates);

          if(oneWayCons==0)
            continue; // No strong connections, skip the tests.

          if(!admissible(*vertex, aggregate_->id(), aggregates))
            continue;

          if( maxOneCons == oneWayCons && oneWayCons > 0) {
            double con = connectivity(*vertex, aggregates);

            if(con == maxCon) {
              int neighbours = noFrontNeighbours(*vertex);

              if(neighbours > maxNeighbours) {
                maxNeighbours = neighbours;
                candidates.clear();
                candidates.push_back(*vertex);
              }else{
                if(neighbours==maxNeighbours)
                {
                  candidates.push_back(*vertex);
                }
              }
            }else if( con > maxCon) {
              maxCon = con;
              maxNeighbours = noFrontNeighbours(*vertex);
              candidates.clear();
              candidates.push_back(*vertex);
            }
          }else if(oneWayCons > maxOneCons) {
            maxOneCons = oneWayCons;
            maxCon = connectivity(*vertex, aggregates);
            maxNeighbours = noFrontNeighbours(*vertex);
            candidates.clear();
            candidates.push_back(*vertex);
          }
        }


        if(!candidates.size())
          break; // No more candidates found
        distance_=distance(seed, aggregates);
        candidates.resize(min(candidates.size(), c.maxAggregateSize()-
                                   aggregate_->size()));
        aggregate_->add(candidates);
      }
    }

    template<typename V>
    template<typename M, typename G, typename C>
    std::tuple<int,int,int,int> AggregatesMap<V>::buildAggregates(const M& matrix, G& graph, const C& criterion,
                                                                  bool finestLevel)
    {
      Aggregator<G> aggregator;
      return aggregator.build(matrix, graph, *this, criterion, finestLevel);
    }

    template<class G>
    template<class M, class C>
    std::tuple<int,int,int,int> Aggregator<G>::build(const M& m, G& graph, AggregatesMap<Vertex>& aggregates, const C& c,
                                                     bool finestLevel)
    {
      using std::max;
      using std::min;
      // Stack for fast vertex access
      Stack stack_(graph, *this, aggregates);

      graph_ = &graph;

      aggregate_ = new Aggregate<G,VertexSet>(graph, aggregates, connected_, front_);

      Timer watch;
      watch.reset();

      buildDependency(graph, m, c, finestLevel);

      dverb<<"Build dependency took "<< watch.elapsed()<<" seconds."<<std::endl;
      int noAggregates, conAggregates, isoAggregates, oneAggregates;
      std::size_t maxA=0, minA=1000000, avg=0;
      int skippedAggregates;
      noAggregates = conAggregates = isoAggregates = oneAggregates =
                                                       skippedAggregates = 0;

      while(true) {
        Vertex seed = stack_.pop();

        if(seed == Stack::NullEntry)
          // No more unaggregated vertices. We are finished!
          break;

        // Debugging output
        if((noAggregates+1)%10000 == 0)
          Dune::dverb<<"c";
        unmarkFront();

        if(graph.getVertexProperties(seed).excludedBorder()) {
          aggregates[seed]=AggregatesMap<Vertex>::ISOLATED;
          ++skippedAggregates;
          continue;
        }

        if(graph.getVertexProperties(seed).isolated()) {
          if(c.skipIsolated()) {
            // isolated vertices are not aggregated but skipped on the coarser levels.
            aggregates[seed]=AggregatesMap<Vertex>::ISOLATED;
            ++skippedAggregates;
            // skip rest as no agglomeration is done.
            continue;
          }else{
            aggregate_->seed(seed);
            growIsolatedAggregate(seed, aggregates, c);
          }
        }else{
          aggregate_->seed(seed);
          growAggregate(seed, aggregates, c);
        }

        /* The rounding step. */
        while(!(graph.getVertexProperties(seed).isolated()) && aggregate_->size() < c.maxAggregateSize()) {

          std::vector<Vertex> candidates;
          candidates.reserve(30);

          typedef typename std::vector<Vertex>::const_iterator Iterator;

          for(Iterator vertex = front_.begin(); vertex != front_.end(); ++vertex) {

            if(graph.getVertexProperties(*vertex).isolated())
              continue; // No isolated nodes here

            if(twoWayConnections( *vertex, aggregate_->id(), aggregates) == 0 &&
               (oneWayConnections( *vertex, aggregate_->id(), aggregates) == 0 ||
                !admissible( *vertex, aggregate_->id(), aggregates) ))
              continue;

            std::pair<int,int> neighbourPair=neighbours(*vertex, aggregate_->id(),
                                                        aggregates);

            //if(aggregateNeighbours(*vertex, aggregate_->id(), aggregates) <= unusedNeighbours(*vertex, aggregates))
            // continue;

            if(neighbourPair.first >= neighbourPair.second)
              continue;

            if(distance(*vertex, aggregates) > c.maxDistance())
              continue; // Distance too far
            candidates.push_back(*vertex);
            break;
          }

          if(!candidates.size()) break; // no more candidates found.

          candidates.resize(min(candidates.size(), c.maxAggregateSize()-
                                     aggregate_->size()));
          aggregate_->add(candidates);

        }

        // try to merge aggregates consisting of only one nonisolated vertex with other aggregates
        if(aggregate_->size()==1 && c.maxAggregateSize()>1) {
          if(!graph.getVertexProperties(seed).isolated()) {
            Vertex mergedNeighbour = mergeNeighbour(seed, aggregates);

            if(mergedNeighbour != AggregatesMap<Vertex>::UNAGGREGATED) {
              // assign vertex to the neighbouring cluster
              aggregates[seed] = aggregates[mergedNeighbour];
              aggregate_->invalidate();
            }else{
              ++avg;
              minA=min(minA,static_cast<std::size_t>(1));
              maxA=max(maxA,static_cast<std::size_t>(1));
              ++oneAggregates;
              ++conAggregates;
            }
          }else{
            ++avg;
            minA=min(minA,static_cast<std::size_t>(1));
            maxA=max(maxA,static_cast<std::size_t>(1));
            ++oneAggregates;
            ++isoAggregates;
          }
          ++avg;
        }else{
          avg+=aggregate_->size();
          minA=min(minA,aggregate_->size());
          maxA=max(maxA,aggregate_->size());
          if(graph.getVertexProperties(seed).isolated())
            ++isoAggregates;
          else
            ++conAggregates;
        }

      }

      Dune::dinfo<<"connected aggregates: "<<conAggregates;
      Dune::dinfo<<" isolated aggregates: "<<isoAggregates;
      if(conAggregates+isoAggregates>0)
        Dune::dinfo<<" one node aggregates: "<<oneAggregates<<" min size="
                   <<minA<<" max size="<<maxA
                   <<" avg="<<avg/(conAggregates+isoAggregates)<<std::endl;

      delete aggregate_;
      return std::make_tuple(conAggregates+isoAggregates,isoAggregates,
                             oneAggregates,skippedAggregates);
    }


    template<class G>
    Aggregator<G>::Stack::Stack(const MatrixGraph& graph, const Aggregator<G>& aggregatesBuilder,
                                const AggregatesMap<Vertex>& aggregates)
      : graph_(graph), aggregatesBuilder_(aggregatesBuilder), aggregates_(aggregates), begin_(graph.begin()), end_(graph.end())
    {
      //vals_ = new  Vertex[N];
    }

    template<class G>
    Aggregator<G>::Stack::~Stack()
    {
      //Dune::dverb << "Max stack size was "<<maxSize_<<" filled="<<filled_<<std::endl;
      //delete[] vals_;
    }

    template<class G>
    const typename Aggregator<G>::Vertex Aggregator<G>::Stack::NullEntry
      = std::numeric_limits<typename G::VertexDescriptor>::max();

    template<class G>
    inline typename G::VertexDescriptor Aggregator<G>::Stack::pop()
    {
      for(; begin_!=end_ && aggregates_[*begin_] != AggregatesMap<Vertex>::UNAGGREGATED; ++begin_) ;

      if(begin_!=end_)
      {
        typename G::VertexDescriptor current=*begin_;
        ++begin_;
        return current;
      }else
        return NullEntry;
    }

#endif // DOXYGEN

    template<class V>
    void printAggregates2d(const AggregatesMap<V>& aggregates, int n, int m,  std::ostream& os)
    {
      using std::max;

      std::ios_base::fmtflags oldOpts=os.flags();

      os.setf(std::ios_base::right, std::ios_base::adjustfield);

      V maxVal=0;
      int width=1;

      for(int i=0; i< n*m; i++)
        maxVal=max(maxVal, aggregates[i]);

      for(int i=10; i < 1000000; i*=10)
        if(maxVal/i>0)
          width++;
        else
          break;

      for(int j=0, entry=0; j < m; j++) {
        for(int i=0; i<n; i++, entry++) {
          os.width(width);
          os<<aggregates[entry]<<" ";
        }

        os<<std::endl;
      }
      os<<std::endl;
      os.flags(oldOpts);
    }


  } // namespace Amg

} // namespace Dune


#endif
