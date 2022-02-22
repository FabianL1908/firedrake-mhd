from firedrake import *
from fimhd import *

class LidDrivenCavityStandardProblem(StandardMHDProblem):
    def __init__(self, args):
        super().__init__(args)

    def base_mesh(self, distribution_parameters):
        if self.args.dim == 2:
            base = UnitSquareMesh(self.args.baseN, self.args.baseN, diagonal="crossed",
                                  distribution_parameters=distribution_parameters)
        elif self.args.dim == 3:
            base = UnitCubeMesh(self.args.baseN, self.args.baseN, self.args.baseN,
                                distribution_parameters=distribution_parameters)
        return base

    def factor_update_coords_in_mh(self):
#        return 0.5
        return 0.0

    def bcs(self, Z):
        mesh = Z.mesh()
        if self.args.dim == 2:
            u_ex = Constant((1, 0), domain=mesh)
            B_ex = Constant((0, 1), domain=mesh)
            B_ex = project(B_ex, Z.sub(2))
            p_ex = Constant(0, domain=mesh)
            E_ex = Constant(0, domain=mesh)

            bcs_ids_apply = 4
            bcs_ids_dont_apply = (1, 2, 3)
            bcs = [DirichletBC(Z.sub(0), u_ex, bcs_ids_apply),  # 4 == upper boundary (y==1)
                   DirichletBC(Z.sub(0), 0, bcs_ids_dont_apply),
                   DirichletBC(Z.sub(2), B_ex, "on_boundary"),
                   DirichletBC(Z.sub(3), 0, "on_boundary"),
                   PressureFixBC(Z.sub(1), 0, 1)]
            sol_ex = (u_ex, p_ex, B_ex, E_ex)
        elif self.args.dim == 3:
            u_ex = Constant((1, 0, 0), domain=mesh)
            B_ex = Constant((0, 1, 0), domain=mesh)
            B_ex = project(B_ex, Z.sub(2))
            p_ex = Constant(0, domain=mesh)
            E_ex = Constant((0, 0, 0), domain=mesh)

            bcs_ids_apply = 4
            bcs_ids_dont_apply = (1, 2, 3, 5, 6)
            bcs = [DirichletBC(Z.sub(0), u_ex, bcs_ids_apply),  # 4 == upper boundary (y==1)
                   DirichletBC(Z.sub(0), 0, bcs_ids_dont_apply),
                   DirichletBC(Z.sub(2), B_ex, "on_boundary"),
                   DirichletBC(Z.sub(3), 0, "on_boundary"),
                   PressureFixBC(Z.sub(1), 0, 1)]
            sol_ex = (u_ex, p_ex, B_ex, E_ex)
        return bcs, bcs_ids_apply, bcs_ids_dont_apply, sol_ex

    def rhs(self, Z):
        return None

if __name__ == "__main__":

    parser = get_default_parser()
    args, _ = parser.parse_known_args()
    problem = LidDrivenCavityStandardProblem(args)
    solver = StandardMHDSolver(args, problem)

#    solver.run()
    solver.print_iteration_numbers()

    
