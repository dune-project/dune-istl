#include <config.h>

#include <iostream>

#include <dune/common/memory/blocked_allocator.hh>

#include <dune/istl/blockvector/host.hh>
#include <dune/istl/io.hh>

int main(int argc, char** argv)
{
  using Dune::printvector;
  typedef Dune::Memory::blocked_std_allocator<double, std::size_t, 4> A;
  Dune::ISTL::BlockVector<double, A> v(8, 4);
  v = 0.0;
  printvector(std::cout, v, "", "");
}
