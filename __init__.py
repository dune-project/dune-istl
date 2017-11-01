from .. import common

from .._istl import BCRSMatrix as BCRSMatrix11
from .._istl import *

from dune.generator.generator import SimpleGenerator
from dune.common.hashit import hashIt
generator = SimpleGenerator("BCRSMatrix","Dune::Python")
def load(includes ,typeName ,constructors=None, methods=None):

    #this contains the registration functions for the class
    includes = includes + ["dune/python/istl/bcrsmatrix.hh"]
    typeHash = "istlbcrsmatrix_" + hashIt(typeName)
    return generator.load(includes ,typeName ,typeHash ,constructors ,methods)

def BCRSMatrix(blockSize):
    if blockSize[0] == blockSize[1] == 1:
        return BCRSMatrix11
    typeName = "Dune::BCRSMatrix< Dune::FieldMatrix< double,"\
            + str(blockSize[0]) + "," + str(blockSize[1]) \
            + " > >"
    includes = ["dune/istl/bcrsmatrix.hh"]
    # todo: provide other constructors
    return load(includes, typeName).BCRSMatrix
