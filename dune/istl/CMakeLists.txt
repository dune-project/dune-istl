# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

add_subdirectory("common")
add_subdirectory("eigenvalue")
add_subdirectory("paamg")
add_subdirectory(test)

#install headers
install(FILES
   allocator.hh
   basearray.hh
   bccsmatrix.hh
   bccsmatrixinitializer.hh
   bcrsmatrix.hh
   bdmatrix.hh
   blocklevel.hh
   btdmatrix.hh
   bvector.hh
   cholmod.hh
   foreach.hh
   gsetc.hh
   ildl.hh
   ilu.hh
   ilusubdomainsolver.hh
   io.hh
   istlexception.hh
   ldl.hh
   matrix.hh
   matrixindexset.hh
   matrixmarket.hh
   matrixmatrix.hh
   matrixredistribute.hh
   matrixutils.hh
   multitypeblockmatrix.hh
   multitypeblockvector.hh
   novlpschwarz.hh
   operators.hh
   overlappingschwarz.hh
   owneroverlapcopy.hh
   preconditioner.hh
   preconditioners.hh
   repartition.hh
   scalarproducts.hh
   scaledidmatrix.hh
   schwarz.hh
   solvercategory.hh
   solver.hh
   solverfactory.hh
   solverregistry.hh
   solvers.hh
   solvertype.hh
   spqr.hh
   superlu.hh
   superlufunctions.hh
   supermatrix.hh
   umfpack.hh
   vbvector.hh
   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/istl)
