import firedrake as fd
import netgen
from netgen.geom2d import SplineGeometry
import numpy as np

import warnings


def generate_ngmesh_spline():
    if fd.COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        geo.AddRectangle((0, 0), (2.2, 0.41), bcs=(2, 3, 2, 1))
        geo.AddCircle ((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc=5)
        ngmesh = geo.GenerateMesh(maxh=0.2)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
    return ngmesh


def generate_mesh(ngmsh, order):
    mesh = fd.Mesh(ngmsh, comm=fd.COMM_WORLD)
    if order > 1:
        cf = mesh.curve_field(order, cg_field=True)
        mesh = fd.Mesh(cf)
    clamp_to_cylinder(mesh)
    return mesh


def extract_cylinder_submesh(mesh):
    submesh = fd.Submesh(mesh, 1, 5, name='cylinder')
    degree = mesh.ufl_coordinate_element().degree()

    if degree > 1:
        element = submesh.ufl_coordinate_element().reconstruct(degree=degree)
        V = fd.FunctionSpace(submesh, element)
        coords = fd.assemble(fd.interpolate(submesh.coordinates, V))
        submesh = fd.Mesh(coords, comm=fd.COMM_WORLD)

    clamp_to_cylinder(submesh)

    return submesh


def clamp_to_cylinder(mesh):
    dofs = mesh.coordinates.function_space().boundary_nodes(5)
    coords = mesh.coordinates.dat.data_ro[dofs, :] - [0.2, 0.2]
    for _ in range(2):
        scale = 0.05 / np.linalg.norm(coords, axis=1)
        coords *= scale[:, None]
    mesh.coordinates.dat.data_wo[dofs, :] = coords + [0.2, 0.2]
    mesh.clear_spatial_index()


def form_a(v, v_):
    nu = 0.001
    Dv = 0.5*(fd.grad(v)+fd.grad(v).T)
    return (fd.inner(fd.grad(v)*v, v_) + fd.inner(2*nu*Dv, fd.grad(v_)))*fd.dx


def form_b(q, v):
    return fd.div(v)*q*fd.dx


def create_velocity_pressure_pair(mesh, family, degree):
    if family == 'TH':
        element_v = fd.VectorElement('P', mesh.ufl_cell(), degree)
        element_p = fd.FiniteElement('P', mesh.ufl_cell(), degree-1)
    elif family == 'MINI':
        P = fd.FiniteElement('P', mesh.ufl_cell(), degree)
        B = fd.FiniteElement('B', mesh.ufl_cell(), degree+mesh.topological_dimension())
        element_v = fd.VectorElement(fd.NodalEnrichedElement(P, B))
        element_p = fd.FiniteElement('P', mesh.ufl_cell(), degree)
    elif family == 'SV':
        element_v = fd.VectorElement('P', mesh.ufl_cell(), degree, variant='alfeld')
        element_p = fd.FiniteElement('DP', mesh.ufl_cell(), degree-1, variant='alfeld')
    else:
        raise ValueError(f"Unknown family {family} and degree {degree}")
    W = fd.FunctionSpace(mesh, element_v * element_p)
    return W


def solve_navier_stokes(w):
    W = w.function_space()
    x = fd.SpatialCoordinate(W.mesh())
    v_in = fd.as_vector([4.0*0.3*x[1]*(0.41-x[1])/0.41/0.41, 0])
    bc_in = fd.DirichletBC(W.sub(0), v_in, 1)
    bc_walls = fd.DirichletBC(W.sub(0), (0, 0), 2)
    bc_cylinder = fd.DirichletBC(W.sub(0), (0, 0), 5)
    bc_out = fd.DirichletBC(W.sub(0).sub(1), 0, 3)
    bcs = [bc_in, bc_walls, bc_cylinder, bc_out]
    v_, p_ = fd.TestFunctions(W)
    v, p = fd.split(w)
    F = form_a(v, v_) - form_b(p, v_) + form_b(p_, v)
    problem = fd.NonlinearVariationalProblem(F, w, bcs=bcs)
    param = {
        'mat_type': 'aij',
        'snes_type': 'newtonls',
        'snes_monitor': None,
        'snes_converged_reason': None,
        'snes_max_it': 12,
        'snes_rtol': 1e-11,
        'snes_atol': 5e-10,
        'snes_linesearch_type': 'basic',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=param)
    solver.solve()
    return w


def compute_traction(Vt, w):
    t = fd.TrialFunction(Vt)
    t_ = fd.TestFunction(Vt)
    a = fd.inner(t, t_)*fd.dx
    A = fd.assemble(a)

    v, p = w.subfunctions
    Vv = v.function_space()
    v_ = fd.TestFunction(Vv)
    L = form_a(v, v_) - form_b(p, v_)
    bc = fd.DirichletBC(Vv, (0, 0), [1, 2, 3])  # not needed
    b = fd.assemble(L, bcs=bc)
    dofs = Vv.boundary_nodes(5)

    b = fd.Cofunction(Vt.dual(), val=b.dat.data_ro[dofs, :])  # FIXME: correct ordering?
    t = fd.Function(Vt)
    fd.solve(A, t, b)
    return t


def run_regular_refinement(num_refinements, order, family):

    def solve_step(ngmsh, w_old=None):
        mesh = generate_mesh(ngmsh, order)
        submesh = extract_cylinder_submesh(mesh)
        W = create_velocity_pressure_pair(mesh, family, order)
        w = fd.Function(W)
        if w_old is not None:
            w.interpolate(w_old)
        w = solve_navier_stokes(w)
        Vt = fd.VectorFunctionSpace(submesh, 'P', order)
        t = compute_traction(Vt, w)
        report_traction(t, W)
        return w

    ngmsh = generate_ngmesh_spline()
    w = solve_step(ngmsh, w_old=None)
    for _ in range(num_refinements):
        ngmsh.Refine()
        w = solve_step(ngmsh, w)


def report_traction(t, W):
    drag = -2.0/(0.2*0.2*0.1) * fd.assemble(t[0]*fd.dx)
    lift = -2.0/(0.2*0.2*0.1) * fd.assemble(t[1]*fd.dx)
    drag_ref = 5.57953523384   # Nabh
    lift_ref = 0.010618948146  # Nabh
    fd.info(f"dim(W)={W.dim()} {drag=:.16} {lift=:.16} "
            f"err_drag={drag-drag_ref} err_lift={lift-lift_ref}")


def main():
    num_refinements = 3
    order = 5
    family = 'TH'
    fd.set_log_level(fd.INFO)
    warnings.filterwarnings('ignore', message='The symbolic `interpolate` has been moved')
    run_regular_refinement(num_refinements, order, family)


if __name__ == '__main__':
    main()
