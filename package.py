# Meta data
name="dune-istl"
version="2.8.201013"
author="The Dune Core developers"
author_email="dune@lists.dune-project.org"
description="This is the iterative solver template library which provides generic sparse matrix/vector classes and a variety of solvers based on these classes. A special feature is the use of templates to exploit the recursive block structure of finite element matrices at compile time. Available solvers include Krylov methods, (block-) incomplete decompositions and aggregation-based algebraic multigrid."
url="https://gitlab.dune-project.org/core/dune-istl"

# Package dependencies
install_requires=['dune-common']

# Python packages to be installed
packages=['dune.istl']
package_dir={'dune.istl': 'python/dune/istl'}

# Module libaries that have to be compiled (without the _ prefix)
modules=['istl']
libraries=['dunecommon']

# Files to include in the source package
manifest='''\
graft cmake
graft doc
graft dune
graft python
include CMakeLists.txt
include config.h.cmake
include dune-istl.pc.in
include dune.module
include LICENSE.md
include pyproject.toml
include README.md
'''
