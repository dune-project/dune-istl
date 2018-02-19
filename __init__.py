from .. import common

from .._istl import BCRSMatrix as BCRSMatrix11
from .._istl import BlockVector as BlockVector1
from .._istl import *

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

def BCRSMatrix(blockSize):
    includes = ["dune/istl/bcrsmatrix.hh"]
    try:
        typeName = "Dune::BCRSMatrix<Dune::" + blockSize._typeName + " >"
        return load(includes,typeName).BCRSMatrix(blockSize)
    except AttributeError:
        #check whether blocksize is 1,1
        if blockSize[0] == blockSize[1] == 1:
            return BCRSMatrix11

        typeName = "Dune::BCRSMatrix< Dune::FieldMatrix< double,"\
                + str(blockSize[0]) + "," + str(blockSize[1]) \
                + " > >"
        # todo: provide other constructors
        return load(includes, typeName).BCRSMatrix

def bcrsMatrix(size=0, *args, **kwargs):
    blockSize = kwargs.get("blockSize",[1,1])
    if size != 0 :
        return BCRSMatrix(blockSize)(size,*args)
    return BCRSMatrix(blockSize)()


def BlockVector(blockSize):
    if blockSize == 1:
        return BlockVector1
    typeName = "Dune::BlockVector< Dune::FieldVector< double ," + str(blockSize) + " > >"
    includes = ["dune/python/istl/bvector.hh"]
    # todo: provide other constructors
    return loadvec(includes, typeName).BlockVector

def blockVector(size, blockSize=1):
    if blockSize == 1:
        return BlockVector1(size)
    typeName = "Dune::BlockVector< Dune::FieldVector< double ," + str(blockSize) + " > >"
    includes = ["dune/istl/bvector.hh"]
    # todo: provide other constructors
    return loadvec(includes, typeName).BlockVector(size)
