from firedrake import *
from utils import get_distribution_parameters

class MHDProblem(object):

    def __init__(self, args):
        self.dim = args.dim
        self.k = args.k
        self.ns_discr = args.ns_discr
        self.mw_discr = args.mw_discr
        self.mhd_type = args.mhd_type
        self.hierarchy = args.hierarchy
        self.distribution_parameters = get_distribution_parameters()
        self.mesh = self.mesh_hierarchy()[-1]
        

    def base_mesh(self):
        raise NotImplementedError

    def mesh_hierarchy(self):
        def before(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+1)


        def after(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+2)

        baseMesh = self.base_mesh(distribution_parameters)
        if self.hierarchy == "bary":
            mh = alfi.BaryMeshHierarchy(base, nref, callbacks=(before, after))
        elif self.hierarchy == "uniform":
            mh = MeshHierarchy(base, nref, reorder=True, callbacks=(before, after),
                               distribution_parameters=self.distribution_parameters)
        else:
            raise NotImplementedError("Only know bary and uniform for the hierarchy.")

        return mh

    def get_variant(self):
        if self.dim == 2:
            variant = "integral"
        elif self.dim == 3:
            variant = f"integral({self.k+1})"

        return variant
    
    def get_up_space(self):
        variant = self.get_variant()
        if self.ns_discr == "hdivbdm":
            Vel = FiniteElement("N2div", self.mesh.ufl_cell(), self.k, variant=variant)
            V = FunctionSpace(self.mesh, Vel)
            Q = FunctionSpace(self.mesh, "DG", k-1)
        elif self.ns_discr == "hdivrt":
            Vel = FiniteElement("N1div", self.mesh.ufl_cell(), self.k, variant=variant)
            V = FunctionSpace(mesh, Vel)
            Q = FunctionSpace(mesh, "DG", self.k-1)
        elif self.ns_discr == "sv":
            V = VectorFunctionSpace(self.mesh, "CG", self.k)
            Q = FunctionSpace(mesh, "DG", self.k-1)
        elif self.ns_discr == "th":
            V = VectorFunctionSpace(self.mesh, "CG", self.k)
            Q = FunctionSpace(self.mesh, "CG", self.k-1)

        return V, Q

    def get_E_space():
        if self.dim == 2:
            R = FunctionSpace(self.mesh, "CG", self.k)  # E
        elif self.dim == 3:
            variant = self.get_variant()
            Rel = FiniteElement("N1curl", self.mesh.ufl_cell(), self.k, variant=variant)
            R = FunctionSpace(self.mesh, Rel)  # E
        return E

    def get_B_space():
        Wel = FiniteElement("N1div", self.mesh.ufl_cell(), self.k, variant=variant)
        W = FunctionSpace(self.mesh, Wel)
        return W

        
    def function_space(self):
        raise NotImplementedError

        if self.mhd_type == "boussinesq":
            pass

        return Z
        
    def bcs(self, Z):
        return NotImplementedError

    def rhs(self, Z):
        return None


class StandardMHDProblem(MHDProblem):

    def __init__(self):
        super.__init__()
        self.mhd_type == "standard"

    def function_space(self):
        V, Q = self.get_up_space()
        W = self.get_B_space()
        
        if self.mw_discr == "BE":
            R = self.get_E_space()
            return MixedFunctionSpace([V, Q, W, R])
        elif self.mw_discr == "Br":
            R = FunctionSpace(self.mesh, "CG", self.k-1)
            return MixedFunctionSpace([V, Q, W, R])

class HallMHDProblem(MHDProblem):

    def __init__(self):
        super.__init__()
        self.mhd_type == "hall"

    def function_space(self):
        V, Q = self.get_up_space()
        W = self.get_B_space()
        R = self.get_E_space()

        if self.dim == 2:
            variant = self.get_variant()
            V3 = B3 = E3 = J3 = FunctionSpace(self.mesh, "CG", self.k)
            Eel = FiniteElement("N1curl", self.mesh.ufl_cell(), self.k, variant=variant)
            R = FunctionSpace(mesh, Eel)
            Jel = FiniteElement("N1curl", self.mesh.ufl_cell(), self.k, variant=variant)
            JJ = FunctionSpace(mesh, Jel)
            return MixedFunctionSpace([V, V3, Q, W, B3, R, E3, JJ, J3])

        elif self.dim == 3:
             return MixedFunctionSpace([V, Q, W, R, R])

class BoussinesqProblem(MHDProblem):
    def __init__(self):
        super.__init__()
        self.mhd_type == "boussinesq"

    def get_T_space():
        return FunctionSpace(self.mesh, "CG", self.k)

    def function_space(self):
        V, Q = self.get_up_space()
        W = self.get_B_space()
        TT = self.get_T_space()
        R = self.get_E_space()

        return MixedFunctionSpace([V, Q, TT, W, R])
