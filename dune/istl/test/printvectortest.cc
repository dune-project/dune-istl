#include <config.h>

#include <iostream>

#include <dune/common/fvector.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/io.hh>

int main(int argc, char** argv)
{
  using Dune::printvector;
  Dune::BlockVector<Dune::FieldVector<double, 4> > v(8);
  v = 0.0;
  printvector(std::cout, v, "", "");
}
