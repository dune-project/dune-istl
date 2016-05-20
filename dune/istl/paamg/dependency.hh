// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_DEPENDENCY_HH
#define DUNE_AMG_DEPENDENCY_HH


#include <bitset>
#include <ostream>

#include "graph.hh"
#include "properties.hh"
#include <dune/common/propertymap.hh>
#include <dune/common/unused.hh>


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
     * @brief Provides classes for initializing the link attributes of a matrix graph.
     */

    /**
     * @brief Class representing the properties of an ede in the matrix graph.
     *
     * During the coarsening process the matrix graph needs to hold different
     * properties of its edges.
     * This class ontains methods for getting and setting these edge attributes.
     */
    class EdgeProperties
    {
      friend std::ostream& operator<<(std::ostream& os, const EdgeProperties& props);
    public:
      /** @brief Flags of the link.*/
      enum {INFLUENCE, DEPEND, SIZE};

    private:

      std::bitset<SIZE> flags_;
    public:
      /** @brief Constructor. */
      EdgeProperties();

      /** @brief Access the bits directly */
      std::bitset<SIZE>::reference operator[](std::size_t v);

      /** @brief Access the bits directly */
      bool operator[](std::size_t v) const;

      /**
       * @brief Checks wether the vertex the edge points to depends on
       * the vertex the edge starts.
       * @return True if it depends on the starting point.
       */
      bool depends() const;

      /**
       * @brief Marks the edge as one of which the end point depends on
       * the starting point.
       */
      void setDepends();

      /**
       * @brief Resets the depends flag.
       */
      void resetDepends();

      /**
       * @brief Checks wether the start vertex is influenced by the end vertex.
       * @return True if it is influenced.
       */
      bool influences() const;

      /**
       * @brief Marks the edge as one of which the start vertex by the end vertex.
       */
      void setInfluences();

      /**
       * @brief Resets the influence flag.
       */
      void resetInfluences();

      /**
       * @brief Checks wether the edge is one way.
       * I.e. either the influence or the depends flag but is set.
       */
      bool isOneWay() const;

      /**
       * @brief Checks wether the edge is two way.
       * I.e. both the influence flag and the depends flag are that.
       */
      bool isTwoWay() const;

      /**
       * @brief Checks wether the edge is strong.
       * I.e. the influence or depends flag is set.
       */
      bool isStrong()  const;

      /**
       * @brief Reset all flags.
       */
      void reset();

      /**
       * @brief Prints the attributes of the edge for debugging.
       */
      void printFlags() const;
    };

    /**
     * @brief Class representing a node in the matrix graph.
     *
     * Contains methods for getting and setting node attributes.
     */
    class VertexProperties {
      friend std::ostream& operator<<(std::ostream& os, const VertexProperties& props);
    public:
      enum { ISOLATED, VISITED, FRONT, BORDER, SIZE };
    private:

      /** @brief The attribute flags. */
      std::bitset<SIZE> flags_;

    public:
      /** @brief Constructor. */
      VertexProperties();

      /** @brief Access the bits directly */
      std::bitset<SIZE>::reference operator[](std::size_t v);

      /** @brief Access the bits directly */
      bool operator[](std::size_t v) const;

      /**
       * @brief Marks that node as being isolated.
       *
       * A node is isolated if it ha not got any strong connections to other nodes
       * in the matrix graph.
       */
      void setIsolated();

      /**
       * @brief Checks wether the node is isolated.
       */
      bool isolated() const;

      /**
       * @brief Resets the isolated flag.
       */
      void resetIsolated();

      /**
       * @brief Mark the node as already visited.
       */
      void setVisited();

      /**
       * @brief Checks wether the node is marked as visited.
       */
      bool visited() const;

      /**
       * @brief Resets the visited flag.
       */
      void resetVisited();

      /**
       * @brief Marks the node as belonging to the current clusters front.
       */
      void setFront();

      /**
       * @brief Checks wether the node is marked as a front node.
       */
      bool front() const;

      /**
       *  @brief Resets the front node flag.
       */
      void resetFront();

      /**
       * @brief Marks the vertex as excluded from the aggregation.
       */
      void setExcludedBorder();

      /**
       * @brief Tests whether the vertex is excluded from the aggregation.
       * @return True if the vertex is excluded from the aggregation process.
       */
      bool excludedBorder() const;

      /**
       * @brief Marks the vertex as included in the aggregation.
       */
      void resetExcludedBorder();

      /**
       * @brief Reset all flags.
       */
      void reset();

    };

    template<typename G, std::size_t i>
    class PropertyGraphVertexPropertyMap
      : public RAPropertyMapHelper<typename std::bitset<VertexProperties::SIZE>::reference,
            PropertyGraphVertexPropertyMap<G,i> >
    {
    public:

      typedef ReadWritePropertyMapTag Category;

      enum {
        /** @brief the index to access in the bitset. */
        index = i
      };

      /**
       * @brief The type of the graph with internal properties.
       */
      typedef G Graph;

      /**
       * @brief The type of the bitset.
       */
      typedef std::bitset<VertexProperties::SIZE> BitSet;

      /**
       * @brief The reference type.
       */
      typedef typename BitSet::reference Reference;

      /**
       * @brief The value type.
       */
      typedef bool ValueType;

      /**
       * @brief The vertex descriptor.
       */
      typedef typename G::VertexDescriptor Vertex;

      /**
       * @brief Constructor.
       * @param g The graph whose properties we access.
       */
      PropertyGraphVertexPropertyMap(G& g)
        : graph_(&g)
      {}

      /**
       * @brief Default constructor.
       */
      PropertyGraphVertexPropertyMap()
        : graph_(0)
      {}


      /**
       * @brief Get the properties associated to a vertex.
       * @param vertex The vertex whose Properties we want.
       */
      Reference operator[](const Vertex& vertex) const
      {
        return graph_->getVertexProperties(vertex)[index];
      }
    private:
      Graph* graph_;
    };

  } // end namespace Amg

  template<typename G, typename EP, typename VM, typename EM>
  struct PropertyMapTypeSelector<Amg::VertexVisitedTag,Amg::PropertiesGraph<G,Amg::VertexProperties,EP,VM,EM> >
  {
    typedef Amg::PropertyGraphVertexPropertyMap<Amg::PropertiesGraph<G,Amg::VertexProperties,EP,VM,EM>, Amg::VertexProperties::VISITED> Type;
  };

  template<typename G, typename EP, typename VM, typename EM>
  typename PropertyMapTypeSelector<Amg::VertexVisitedTag,Amg::PropertiesGraph<G,Amg::VertexProperties,EP,VM,EM> >::Type
  get(const Amg::VertexVisitedTag& tag, Amg::PropertiesGraph<G,Amg::VertexProperties,EP,VM,EM>& graph)
  {
    DUNE_UNUSED_PARAMETER(tag);
    return Amg::PropertyGraphVertexPropertyMap<Amg::PropertiesGraph<G,Amg::VertexProperties,EP,VM,EM>, Amg::VertexProperties::VISITED>(graph);
  }

  namespace Amg
  {
    inline std::ostream& operator<<(std::ostream& os, const EdgeProperties& props)
    {
      return os << props.flags_;
    }

    inline EdgeProperties::EdgeProperties()
      : flags_()
    {}

    inline std::bitset<EdgeProperties::SIZE>::reference
    EdgeProperties::operator[](std::size_t v)
    {
      return flags_[v];
    }

    inline bool EdgeProperties::operator[](std::size_t i) const
    {
      return flags_[i];
    }

    inline void EdgeProperties::reset()
    {
      flags_.reset();
    }

    inline void EdgeProperties::setInfluences()
    {
      // Set the INFLUENCE bit
      //flags_ |= (1<<INFLUENCE);
      flags_.set(INFLUENCE);
    }

    inline bool EdgeProperties::influences() const
    {
      // Test the INFLUENCE bit
      return flags_.test(INFLUENCE);
    }

    inline void EdgeProperties::setDepends()
    {
      // Set the first bit.
      //flags_ |= (1<<DEPEND);
      flags_.set(DEPEND);
    }

    inline void EdgeProperties::resetDepends()
    {
      // reset the first bit.
      //flags_ &= ~(1<<DEPEND);
      flags_.reset(DEPEND);
    }

    inline bool EdgeProperties::depends() const
    {
      // Return the first bit.
      return flags_.test(DEPEND);
    }

    inline void EdgeProperties::resetInfluences()
    {
      // reset the second bit.
      flags_ &= ~(1<<INFLUENCE);
    }

    inline bool EdgeProperties::isOneWay() const
    {
      // Test whether only the first bit is set
      //return isStrong() && !isTwoWay();
      return ((flags_) & std::bitset<SIZE>((1<<INFLUENCE)|(1<<DEPEND)))==(1<<DEPEND);
    }

    inline bool EdgeProperties::isTwoWay() const
    {
      // Test whether the first and second bit is set
      return ((flags_) & std::bitset<SIZE>((1<<INFLUENCE)|(1<<DEPEND)))==((1<<INFLUENCE)|(1<<DEPEND));
    }

    inline bool EdgeProperties::isStrong() const
    {
      // Test whether the first or second bit is set
      return ((flags_) & std::bitset<SIZE>((1<<INFLUENCE)|(1<<DEPEND))).to_ulong();
    }


    inline std::ostream& operator<<(std::ostream& os, const VertexProperties& props)
    {
      return os << props.flags_;
    }

    inline VertexProperties::VertexProperties()
      : flags_()
    {}


    inline std::bitset<VertexProperties::SIZE>::reference
    VertexProperties::operator[](std::size_t v)
    {
      return flags_[v];
    }

    inline bool VertexProperties::operator[](std::size_t v) const
    {
      return flags_[v];
    }

    inline void VertexProperties::setIsolated()
    {
      flags_.set(ISOLATED);
    }

    inline bool VertexProperties::isolated() const
    {
      return flags_.test(ISOLATED);
    }

    inline void VertexProperties::resetIsolated()
    {
      flags_.reset(ISOLATED);
    }

    inline void VertexProperties::setVisited()
    {
      flags_.set(VISITED);
    }

    inline bool VertexProperties::visited() const
    {
      return flags_.test(VISITED);
    }

    inline void VertexProperties::resetVisited()
    {
      flags_.reset(VISITED);
    }

    inline void VertexProperties::setFront()
    {
      flags_.set(FRONT);
    }

    inline bool VertexProperties::front() const
    {
      return flags_.test(FRONT);
    }

    inline void VertexProperties::resetFront()
    {
      flags_.reset(FRONT);
    }

    inline void VertexProperties::setExcludedBorder()
    {
      flags_.set(BORDER);
    }

    inline bool VertexProperties::excludedBorder() const
    {
      return flags_.test(BORDER);
    }

    inline void VertexProperties::resetExcludedBorder()
    {
      flags_.reset(BORDER);
    }

    inline void VertexProperties::reset()
    {
      flags_.reset();
    }

    /** @} */
  }
}
#endif
