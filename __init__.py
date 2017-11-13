from .. import common

from .._istl import BCRSMatrix as BCRSMatrix11
from .._istl import BlockVector as BlockVector1
from .._istl import *

from dune.generator.generator import SimpleGenerator
from dune.common.hashit import hashIt
generator = SimpleGenerator("BCRSMatrix","Dune::Python")
generatorvec = SimpleGenerator("BlockVector","Dune::Python")

def load(includes ,typeName ,constructors=None, methods=None):

    #this contains the registration functions for the class
    includes = includes + ["dune/python/istl/bcrsmatrix.hh"]
    typeHash = "istlbcrsmatrix_" + hashIt(typeName)
    return generator.load(includes ,typeName ,typeHash ,constructors ,methods)


def loadvec(includes ,typeName ,constructors=None, methods=None):

    #this contains the registration functions for the class
    includes = includes + ["dune/python/istl/bvector.hh"]
    includes = includes + ["dune/python/common/fvector.hh"]
    typeHash = "istlbvector_" + hashIt(typeName)
    return generatorvec.load(includes ,typeName ,typeHash ,constructors ,methods)


def BCRSMatrix(blockSize):
    if blockSize[0] == blockSize[1] == 1:
        print("returning singular object")
        return BCRSMatrix11
    typeName = "Dune::BCRSMatrix< Dune::FieldMatrix< double,"\
            + str(blockSize[0]) + "," + str(blockSize[1]) \
            + " > >"
    includes = ["dune/istl/bcrsmatrix.hh"]
    print("returning bigger object")
    # todo: provide other constructors
    return load(includes, typeName).BCRSMatrix


def BlockVector(blockSize):
    if blockSize == 1:
        return BlockVector1
    typeName = "Dune::BlockVector< Dune::FieldVector< double," + str(blockSize) + " > >"
    includes = ["dune/istl/bvector.hh"]
    includes = includes + ["dune/common/fmatrix.hh"]
    print("bindings for bigger than 1 blocksize")
    # todo: provide other constructors
    return loadvec(includes, typeName).BlockVector
