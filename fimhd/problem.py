from firedrake import *
from utils import get_distribution_parameters, eps, scross, scurl, vcross, vcurl
import ufl.algorithms

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
        self.Z = self.function_space()
        self.gamma = args.gamma
        self.advect = args.advect
        self.linearisation = args.linearisation
        self.z = Function(self.Z)
        self.z_test = TestFunction(self.Z)

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

    def get_ns_dg_form(self, u, p, v, q):
        _, bcs_ids_apply, bcs_ids_dont_apply, u_ex = self.bcs(self.Z)
        h = CellVolume(self.mesh)/FacetArea(self.mesh)
        n = FacetNormal(self.mesh)
        if self.dim == 2:
            sigma_fac = Constant(10)
        elif self.dim == 3:
            sigma_fac = Constant(1)
        sigma = sigma_fac * self.Z.sub(0).ufl_element().degree()**2  # penalty term
        theta = Constant(1)  # theta=1 means upwind discretization of nonlinear advection term
        uflux_int = 0.5*(dot(u, n) + theta*abs(dot(u, n))) * u
        uflux_ext_1 = 0.5*(inner(u, n) + theta*abs(inner(u, n))) * u
        uflux_ext_2 = 0.5*(inner(u, n) - theta*abs(inner(u, n))) * u_ex

        F_DG = (
             - 1/self.Re * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
             - 1/self.Re * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
             + 1/self.Re * sigma/avg(h) * inner(2*avg(outer(u, n)), 2*avg(outer(v, n))) * dS
             - inner(outer(v, n), 2/self.Re*sym(grad(u))) * ds
             - inner(outer(u-u_ex, n), 2/self.Re*sym(grad(v))) * ds(bcs_ids_apply)
             + 1/self.Re*(sigma/h)*inner(v, u-u_ex) * ds(bcs_ids_apply)
             - self.advect * dot(u, div(outer(v, u))) * dx
             + self.advect * dot(v('+')-v('-'), uflux_int('+')-uflux_int('-')) * dS
             + self.advect * dot(v, uflux_ext_1) * ds
             + self.advect * dot(v, uflux_ext_2) * ds(bcs_ids_apply)
        )

        if bcs_ids_dont_apply is not None:
            F_DG += (
                - inner(outer(u, n), 2/self.Re*sym(grad(v))) * ds(bcs_ids_dont_apply)
                + 1/self.Re*(sigma/h)*inner(v, u) * ds(bcs_ids_dont_apply)
               )

        return F_DG

    def function_space(self):
        raise NotImplementedError

    def form(self):
        raise NotImplementedError
       
    def bcs(self, Z):
        raise NotImplementedError

    def rhs(self, Z):
        return None

    def jacobian(self):
        raise NotImplementedError

class StandardMHDProblem(MHDProblem):

    def __init__(self):
        super.__init__()
        self.mhd_type == "standard"
        Re = Constant(1.0)
        self.Re = Re
        Rem = Constant(1.0)
        self.Rem = Rem
        S = Constant(1.0)
        self.S = S

    def function_space(self):
        V, Q = self.get_up_space()
        W = self.get_B_space()
        
        if self.mw_discr == "BE":
            R = self.get_E_space()
            return MixedFunctionSpace([V, Q, W, R])
        elif self.mw_discr == "Br":
            R = FunctionSpace(self.mesh, "CG", self.k-1)
            return MixedFunctionSpace([V, Q, W, R])

    def form(self):
        (u, p, B, E) = split(self.z)
        (v, q, C, Ff) = split(self.z_test)

        if self.dim == 2:
            F = (
                  2/self.Re * inner(eps(u), eps(v))*dx
                # + advect * inner(dot(grad(u), u), v) * dx
                + self.gamma * inner(div(u), div(v)) * dx
                + self.S * inner(vcross(B, E), v) * dx
                + self.S * inner(vcross(B, scross(u, B)), v) * dx
                - inner(p, div(v)) * dx
                - inner(div(u), q) * dx
                + inner(E, Ff) * dx
                + inner(scross(u, B), Ff) * dx
                - 1/self.Rem * inner(B, vcurl(Ff)) * dx
                + inner(vcurl(E), C) * dx
                + 1/self.Rem * inner(div(B), div(C)) * dx
            )
        elif self.dim == 3:
            F = (
                2/self.Re * inner(eps(u), eps(v))*dx
                # + advect * inner(dot(grad(u), u), v) * dx
                + self.gamma * inner(div(u), div(v)) * dx
                + self.S * inner(cross(B, E), v) * dx
                + self.S * inner(cross(B, cross(u, B)), v) * dx
                - inner(p, div(v)) * dx
                - inner(div(u), q) * dx
                + inner(E, Ff) * dx
                + inner(cross(u, B), Ff) * dx
                - 1/self.Rem * inner(B, curl(Ff)) * dx
                + inner(curl(E), C) * dx
                + 1/self.Rem * inner(div(B), div(C)) * dx
            )

        if self.ns_discr in ["hdivbdm", "hdivrt"]:        
            F += self.get_ns_dg_form(u, p, v, q)
        elif self.ns_discr in ["sv", "th"]:
            F += self.advect * inner(dot(grad(u), u), v) * dx

        rhs = self.rhs()
        if rhs is not None:
           f1, f2, f3, f4 = rhs
           F -= inner(f1, v) * dx + inner(f3, Ff) * dx + inner(f2, C) * dx

        return F

    def jacobian(self, F):
        (u, p, B, E) = split(self.z)
        (v, q, C, Ff) = split(self.z_test)

        w = TrialFunction(self.Z)
        [w_u, w_p, w_B, w_E] = split(w)

        J_newton = ufl.algorithms.expand_derivatives(derivative(F, self.z, w))

        if self.dim == 2:            
            if self.linearisation == "newton":
                J = J_newton
            elif self.linearisation == "mdp":
                J_mdp = (
                      J_newton
                    - inner(scross(w_u, B), Ff) * dx  # G
                        )
                J = J_mdp

            elif self.linearisation == "picard":
                J_picard = (
                      J_newton
                    - self.S * inner(vcross(w_B, E), v) * dx  # J_tilde
                    - self.S * inner(vcross(w_B, scross(u, B)), v) * dx  # D_1_tilde
                    - self.S * inner(vcross(B, scross(u, w_B)), v) * dx  # D_2_tilde
                    - inner(scross(u, w_B), Ff) * dx  # G_tilde
                        )
                J = J_picard
        elif self.dim == 3:
            if self.linearisation == "newton":
                J = J_newton

            elif self.linearisation == "mdp":
                J_mdp = (
                      J_newton
                    - inner(cross(w_u, B), Ff) * dx  # G
                        )
                J = J_mdp

            elif self.linearisation == "picard":
                J_picard = (
                      J_newton
                    - S * inner(cross(w_B, E), v) * dx  # J_tilde
                    - S * inner(cross(w_B, cross(u, B)), v) * dx  # D_1_tilde
                    - S * inner(cross(B, cross(u, w_B)), v) * dx  # D_2_tilde
                    - inner(cross(u, w_B), Ff) * dx  # G_tilde
                        )
                J = J_picard
        return J
        
        
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
