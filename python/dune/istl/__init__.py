# SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

import dune.common

from ._istl import BCRSMatrix as BCRSMatrix11
from ._istl import BlockVector as BlockVector1
from ._istl import *

from dune.generator.generator import SimpleGenerator
from dune.common.hashit import hashIt
generator = SimpleGenerator("BCRSMatrix","Dune::Python")
generatorvec = SimpleGenerator("BlockVector","Dune::Python")
generatormatrixindexset = SimpleGenerator("MatrixIndexSet","Dune::Python")

def load(includes ,typeName ,constructors=None, methods=None):
    includes = includes + ["dune/python/istl/bcrsmatrix.hh"]
    typeHash = "istlbcrsmatrix_" + hashIt(typeName)
    return generator.load(includes ,typeName ,typeHash ,constructors ,methods)

def loadvec(includes ,typeName ,constructors=None, methods=None):
    includes = includes + ["dune/python/common/fvector.hh",
                           "dune/python/istl/bvector.hh"]
    typeHash = "istlbvector_" + hashIt(typeName)
    return generatorvec.load(includes ,typeName ,typeHash ,constructors ,methods)

def loadmatrixindexset(includes ,typeName ,constructors=None, methods=None):
    includes = includes + ["dune/python/istl/matrixindexset.hh"]
    typeHash = "matrixindexset_" + hashIt(typeName)
    return generatormatrixindexset.load(includes ,typeName ,typeHash ,constructors ,methods)

def MatrixIndexSet(rows, cols):
    includes = ["dune/istl/matrixindexset.hh"]
    typeName = "Dune::MatrixIndexSet"
    return loadmatrixindexset(includes, typeName).MatrixIndexSet(rows,cols)

def BCRSMatrix(blockType):
    includes = ["dune/istl/bcrsmatrix.hh"]
    try:
        typeName = "Dune::BCRSMatrix<" + blockType.cppTypeName + " >"
        return load(includes,typeName).BCRSMatrix
    except AttributeError:
        #check whether blocksize is 1,1
        if blockType[0] == blockType[1] == 1:
            return BCRSMatrix11

        typeName = "Dune::BCRSMatrix< Dune::FieldMatrix< double,"\
                + str(blockType[0]) + "," + str(blockType[1]) \
                + " > >"
        return load(includes, typeName).BCRSMatrix

def bcrsMatrix(size=0, *args, **kwargs):
    blockType = kwargs.get("blockType",[1,1])
    if size != 0 :
        return BCRSMatrix(blockType)(size,*args)
    return BCRSMatrix(blockType)()

def BlockVector(typeOrBlockSize,size=None):
    """
    Create a BlockVector object with specified type or block size for a FieldVector.

    Args:
        typeOrBlockSize (int or str or object): The type of the block vector or the block size.
            - If int, specifies the block size directly.
            - If typeOrBlockSize is not an int, it should have a 'cppTypeName' attribute specifying the type and 'cppIncludes' attribute specifying additional include files.
        size (int, optional): The size of the vector. Default is None.

    Returns:
        BlockVector: The created BlockVector object.
    """
    typeName=[]
    includes=["dune/python/istl/bvector.hh"]
    if hasattr (typeOrBlockSize, "cppTypeName") and hasattr (typeOrBlockSize, "cppIncludes"):

        typeName = f"Dune::BlockVector< {typeOrBlockSize.cppTypeName} >"

        includes += typeOrBlockSize.cppIncludes
        if size is None:
           return loadvec(includes,typeName).BlockVector
        bvec = loadvec(includes,typeName).BlockVector(size)
        for i in range(len(bvec)):
            bvec[i]=typeOrBlockSize
        return bvec
    else:
        if typeOrBlockSize == 1:
            if size is None:
                return BlockVector1
            else:
                return BlockVector1(size)
        dune.common.FieldVector(typeOrBlockSize) # make sure that the FieldVector is registered    
        typeName = "Dune::BlockVector<Dune::FieldVector< double, " + str(typeOrBlockSize) + " > >"
    if size is None:
        return loadvec(includes, typeName).BlockVector
    else:
        return loadvec(includes, typeName).BlockVector(size)

def blockVector(size, blockSize=1):
    """
    Creates a Dune BlockVector object of size `size` with FieldVector<double,`blockSize`> as blocks.

    Args:
        size (int): The size of the vector.
        blockSize (int, optional): The block size. Default is 1.

    Returns:
        BlockVector: The created BlockVector object.
    """
    return BlockVector(blockSize,size)
