# Meta data
name="dune-istl"
version="2.8"
author="The Dune Core developers"
author_email="dune@lists.dune-project.org"
description="This is the iterative solver template library which provides generic sparse matrix/vector classes and a variety of solvers based on these classes."
url="https://gitlab.dune-project.org/core/dune-istl"

# DUNE dependencies
dune_dependencies=['dune-common']

# Package dependencies
install_requires=[]

# Module libaries that have to be compiled (without the _ prefix)
modules=['istl']

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
include README.md
'''
