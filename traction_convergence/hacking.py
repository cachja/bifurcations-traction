import firedrake as fd
import netgen
from netgen.geom2d import SplineGeometry
import numpy as np

from warnings import filterwarnings
from time import time


def generate_ngmesh_spline(maxh):
    if fd.COMM_WORLD.rank == 0:
        geo = SplineGeometry()
        geo.AddRectangle((0, 0), (2.2, 0.41), bcs=(2, 3, 2, 1))
        geo.AddCircle ((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc=5)
        ngmesh = geo.GenerateMesh(maxh=maxh)
    else:
        ngmesh = netgen.libngpy._meshing.Mesh(2)
    return ngmesh


def generate_mesh(ngmsh, order):
    mesh = fd.Mesh(ngmsh, comm=fd.COMM_WORLD)
    if order > 1:
        cf = mesh.curve_field(order, cg_field=True)
        make_mesh_displacement_normal(cf, mesh.coordinates)
        mesh = fd.Mesh(cf)
    clamp_to_cylinder(mesh)
    return mesh


def extract_cylinder_submesh(mesh):
    submesh = fd.Submesh(mesh, 1, 5, name='cylinder')
    degree = mesh.ufl_coordinate_element().degree()
    if degree > 1:
        element = submesh.ufl_coordinate_element().reconstruct(degree=degree)
        V = fd.FunctionSpace(submesh, element)
        coords = fd.Function(V)
        make_mesh_displacement_normal(coords, submesh.coordinates)
        submesh = fd.Mesh(coords)
    clamp_to_cylinder(submesh)
    return submesh


def make_mesh_displacement_normal(coords_pk, coords_p1):
    """This enforces the displacement nodes uniformly distributed on P1 edges.
    This makes `check_submesh_geometry()` pass with zero angle errors. Hence
    the angles in the traction space (on the submesh) correspond exactly to
    angles in the velocity space (on the full mesh) as assumed in
    `compute_subspace_dof_ordering()`.

    Doing this together with `clamp_to_cylinder()` thus renders ngsPETSc's
    curved mesh handling moot. We could live without it.
    """
    coords_pk.interpolate(coords_p1)


def clamp_to_cylinder(mesh):
    dofs = mesh.coordinates.function_space().boundary_nodes(5)
    coords = mesh.coordinates.dat.data_ro[dofs, :] - [0.2, 0.2]
    for _ in range(2):
        scale = 0.05 / np.linalg.norm(coords, axis=1)
        coords *= scale[:, None]
    mesh.coordinates.dat.data_wo[dofs, :] = coords + [0.2, 0.2]
    mesh.clear_spatial_index()


def reconstruct_function_space_on_affine_mesh(V, family=None):
    element = V.mesh().ufl_coordinate_element().reconstruct(degree=1, family=family)
    coord_space = fd.FunctionSpace(V.mesh(), element)
    coords = fd.assemble(fd.interpolate(V.mesh().coordinates, coord_space))
    mesh = fd.Mesh(coords)
    return V.reconstruct(mesh=mesh)


def transfer_function(f_src, f_dest):

    # Non-matching projection not implemented on curved meshes:
    #
    #   f_dest.sub(0).project(f_src.sub(0))
    #   f_dest.sub(1).project(f_src.sub(1))
    #
    # Non-matching interpolation blows up with order (not sure whether
    # the element order and/or the mesh order):
    #
    #   f_dest.interpolate(f_src)

    # Hence move temporarily to affine mesh
    W_src = reconstruct_function_space_on_affine_mesh(f_src.function_space())
    W_dest = reconstruct_function_space_on_affine_mesh(f_dest.function_space())
    # Following function share data with input functions
    f_src = fd.Function(W_src, val=f_src.dat)
    f_dest = fd.Function(W_dest, val=f_dest.dat)

    # Actually transfer between meshes
    # This is still quite slow, but way faster than non-affine
    t0 = time()
    f_dest.interpolate(f_src)
    fd.debug(f"non-matching interpolate time {time()-t0:.2f}s")


def compute_subspace_dof_ordering(Vv, dofs, Vt):
    v = fd.Function(Vv)
    v.dat.data_wo[dofs, 0] = np.arange(len(dofs))
    t = fd.assemble(fd.interpolate(v, Vt))
    perm = t.dat.data_ro[:, 0].round().astype(fd.PETSc.IntType)
    assert np.all(np.sort(perm) == np.arange(len(dofs)))

    check_submesh_geometry(Vv, dofs, Vt, perm)

    return dofs[perm]


def check_submesh_geometry(Vv, dofs, Vt, perm):
    coords_v = Vv.mesh().coordinates.dat.data_ro[dofs[perm], :]
    coords_t = Vt.mesh().coordinates.dat.data_ro[:, :]
    assert np.allclose(np.linalg.norm(coords_v-[0.2, 0.2], axis=1), 0.05)
    assert np.allclose(np.linalg.norm(coords_t-[0.2, 0.2], axis=1), 0.05)
    phi_v = np.atan2(coords_v[:, 1]-0.2, coords_v[:, 0]-0.2)
    phi_t = np.atan2(coords_t[:, 1]-0.2, coords_t[:, 0]-0.2)
    assert np.allclose(phi_v, phi_t, atol=1e-14, rtol=0)


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
    bc_out = fd.DirichletBC(W.sub(0).sub(1), 0, 3)  # controversial BC
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
        'snes_rtol': 1e-12,
        'snes_atol': 1e-25,
        'snes_linesearch_type': 'basic',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=param)
    solver.solve()


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
    dofs = compute_subspace_dof_ordering(Vv, dofs, Vt)

    b = fd.Cofunction(Vt.dual(), val=b.dat.data_ro[dofs, :])
    t = fd.Function(Vt)
    fd.solve(A, t, b)
    return t


def run_regular_refinement(h_initial, num_refinements, order, family):

    convergence_data = []
    solution_data = []

    def solve_step(ngmsh, w_old=None):
        mesh = generate_mesh(ngmsh, order)
        submesh = extract_cylinder_submesh(mesh)
        W = create_velocity_pressure_pair(mesh, family, order)
        w = fd.Function(W)
        if w_old is not None:
            transfer_function(w_old, w)  # SLOW, can be commented out
        solve_navier_stokes(w)

        Vt = fd.VectorFunctionSpace(submesh, 'P', order)
        t = compute_traction(Vt, w)

        fd.info("Babuska-Miller trick:")
        drag, lift = compute_integral_traction_babuska(w)
        errs = report_traction(W.dim(), drag, lift)

        fd.info("Integrate pointwise traction:")
        drag, lift = compute_integral_traction_from_pointwise_traction(t)
        errs2 = report_traction(W.dim(), drag, lift)

        convergence_data.append((W.dim(), Vt.dim(), *errs, *errs2))
        solution_data.append((W.dim(), Vt.dim(), w, t))
        return w

    ngmsh = generate_ngmesh_spline(h_initial)
    w = solve_step(ngmsh, w_old=None)
    for _ in range(num_refinements):
        ngmsh.Refine()
        w = solve_step(ngmsh, w_old=w)
    return convergence_data, solution_data


def compute_integral_traction_babuska(w):
    w_ = fd.Function(w.function_space())
    v, p = w.subfunctions
    v_, p_ = w_.subfunctions
    L = form_a(v, v_) - form_b(p, v_)
    fd.DirichletBC(w_.function_space().sub(0), (1, 0), 5).apply(w_)
    drag = -2.0/(0.2*0.2*0.1) * fd.assemble(L)
    fd.DirichletBC(w_.function_space().sub(0), (0, 1), 5).apply(w_)
    lift = -2.0/(0.2*0.2*0.1) * fd.assemble(L)
    return drag, lift


def compute_integral_traction_from_pointwise_traction(t):
    drag = -2.0/(0.2*0.2*0.1) * fd.assemble(t[0]*fd.dx)
    lift = -2.0/(0.2*0.2*0.1) * fd.assemble(t[1]*fd.dx)
    return drag, lift


def report_traction(Wdim, drag, lift):
    drag_nabh = 5.57953523384   # Nabh
    drag_hron = 5.5795352338502 # Hron
    lift_nabh = 0.010618948146  # Nabh
    lift_hron = 0.0106189481265 # Hron
    fd.info(f"dim(W)={Wdim} {drag=:.16} {lift=:.16} "
            f"err_drag={drag_nabh-drag} {drag_hron-drag} "
            f"err_lift={lift_nabh-lift} {lift_hron-lift}")
    return drag_nabh-drag, drag_hron-drag, lift_nabh-lift, lift_hron-lift


def postprocess_convergence_data(data):
    data = np.array(data)
    h = data[:, (0,)]**-0.5
    errs = np.abs(data[:, 2:])
    rates = np.log(errs[1:, :]/errs[:-1, :]) / np.log(h[1:]/h[:-1])
    with np.printoptions(precision=2):
        fd.info(f"Convergence rates:\n{rates}")


def transfer_traction_to_interval(t):
    V = t.function_space()
    # TODO: Perhaps we could unwind t directly on the curved mesh?
    #       And have better accuracy?
    V = reconstruct_function_space_on_affine_mesh(V, family='DP')
    coords = V.mesh().coordinates.dat.data
    coords[:, 0] = np.atan2(coords[:, 1] - 0.2, coords[:, 0] - 0.2)
    coords[:, 1] = 0
    V.mesh().clear_spatial_index()
    return fd.Function(V, val=t.dat)


def transfer_traction_to_interval_2(t):
    V = t.function_space()
    coords = V.mesh().coordinates.copy(deepcopy=True)
    V = V.reconstruct(mesh=fd.Mesh(coords))
    coords = V.mesh().coordinates.dat.data
    coords[:, 0] = np.atan2(coords[:, 1] - 0.2, coords[:, 0] - 0.2)
    coords[:, 1] = 0
    V.mesh().clear_spatial_index()
    return fd.Function(V, val=t.dat)


def postprocess_solution_data(data, map_traction):

    t_fine = map_traction(data[-1][3])
    convergence = []
    for i, (dim_vp, dim_t, _, t) in enumerate(data[:-1]):
        t_coarse = map_traction(t)
        t_coarse = fd.assemble(fd.interpolate(t_coarse, t_fine.function_space()))
        mesh_fine = t_fine.function_space().mesh()
        err_t = fd.inner(t_fine-t_coarse, t_fine-t_coarse)*fd.dx(domain=mesh_fine)
        err_t = fd.assemble(err_t) ** 0.5
        fd.info(f"Traction L2 error level {i}: {err_t}")
        convergence.append((dim_vp, dim_t, err_t))

    postprocess_convergence_data(convergence)


def main():
    h_initial = 1.0
    num_refinements = 4
    order = 4
    family = 'TH'
    fd.set_log_level(fd.DEBUG)
    filterwarnings('ignore', message='The symbolic `interpolate` has been moved')
    convergence_data, solution_data = run_regular_refinement(h_initial,
        num_refinements, order, family)
    postprocess_convergence_data(convergence_data)
    postprocess_solution_data(solution_data, transfer_traction_to_interval)
    postprocess_solution_data(solution_data, transfer_traction_to_interval_2)


if __name__ == '__main__':
    main()
