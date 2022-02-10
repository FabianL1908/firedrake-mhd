from firedrake import *
import numpy
from pyop2.datatypes import IntType

def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]


def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])


def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])


def acurl(x):
    return as_vector([
                     x[2].dx(1),
                     -x[2].dx(0),
                     x[1].dx(0) - x[0].dx(1)
                     ])

def get_distribution_parameters():
    return {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

def eps(x):
    return sym(grad(x))

def message(msg, mesh):
    if mesh.comm.rank == 0:
        warning(msg)

class PressureFixBC(DirichletBC):
    def __init__(self, V, val, subdomain, method="topological"):
        super().__init__(V, val, subdomain, method)
        sec = V.dm.getDefaultSection()
        dm = V.mesh().topology_dm

        coordsSection = dm.getCoordinateSection()
        dim = dm.getCoordinateDim()
        coordsVec = dm.getCoordinatesLocal()

        (vStart, vEnd) = dm.getDepthStratum(0)
        indices = []
        for pt in range(vStart, vEnd):
            x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
            if x.dot(x) == 0.0:  # fix [0, 0] in original mesh coordinates (bottom left corner)
                if dm.getLabelValue("pyop2_ghost", pt) == -1:
                    indices = [pt]
                break

        nodes = []
        for i in indices:
            if sec.getDof(i) > 0:
                nodes.append(sec.getOffset(i))

        if V.mesh().comm.rank == 0:
            nodes = [0]
        else:
            nodes = []
        self.nodes = numpy.asarray(nodes, dtype=IntType)

        if len(self.nodes) > 0:
            print("Fixing nodes %s" % self.nodes)
        import sys
        sys.stdout.flush()
