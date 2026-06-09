import firedrake as fd
import numpy as np

def get_boundary_values_old(V, mesh, bndry_num, fun, fun_index):
    # Get values vertex by vertex on a given boundary
    def create_vertex_values(V, bndry_num):
        dofs_in_vertex = {}
        # Use DirichletBC to extract DoFs on the boundary
        boundary_dofs = fd.DirichletBC(V, 0, bndry_num).nodes
        
        # Get coordinates of DoFs
        mesh_coords = mesh.coordinates.dat.data  # Coordinates of the mesh

        for dof in boundary_dofs:
            coord = tuple(mesh_coords[dof])
            dofs_in_vertex[dof] = {
                "coordinates": coord,
                "dof": dof
            }
        return dofs_in_vertex

    dofs_in_vertex = create_vertex_values(V, bndry_num)

    # Append values we are interested in
    # Choose parametrization as follows: "angle", "x", "y", "xy"
    def get_values_to_list(fun, fun_index):
        fun_values = []
        fun_coord = []

        for vertex in dofs_in_vertex.values():
            coord = vertex["coordinates"]
            fun_value = fun.at(coord)[fun_index]  # Evaluate function at coordinates
            fun_values.append(fun_value)
            fun_coord.append(coord)

        return (fun_values, fun_coord)

    (fun_values, fun_coord) = get_values_to_list(fun, fun_index)

    return (fun_coord, fun_values)

def get_boundary_values(V, mesh, bndry_num, fun, fun_index):
    comm = mesh.comm

    # --- get boundary dofs ---
    bc = fd.DirichletBC(V, 0, bndry_num)
    boundary_dofs = np.array(bc.nodes, dtype=int)

    # --- DOF coordinates (safe for CG1) ---
    coords = mesh.coordinates.dat.data_ro_with_halos
    coords_local = coords[boundary_dofs]

    # --- function values (NO .at(), direct access!) ---
    values_local_all = fun.dat.data_ro_with_halos
    if fun.function_space().value_size > 1:
        values_local = values_local_all[boundary_dofs][:, fun_index]
    else:
        values_local = values_local_all[boundary_dofs]

    # --- gather to rank 0 ---
    coords_all = comm.gather(coords_local, root=0)
    values_all = comm.gather(values_local, root=0)


    if comm.rank == 0:
        # --- flatten ---
        coords_all = np.vstack(coords_all)
        values_all = np.concatenate(values_all)

        # --- remove duplicates correctly ---
        coords_all = np.ascontiguousarray(coords_all)
        _, unique_idx = np.unique(coords_all, axis=0, return_index=True)

        coords_all = coords_all[unique_idx]
        values_all = values_all[unique_idx]

        return coords_all, values_all
    else:
        return None, None

def get_boundary_values_HO(V, mesh, bndry_num, fun, fun_index):
    comm = mesh.comm

    # --- 1. Boundary DOFs ---
    bc = fd.DirichletBC(V, 0, bndry_num)
    boundary_dofs = np.array(bc.nodes, dtype=int)

    # --- 2. Coordinates in SAME FE space (works for CGk + curved) ---
    x, y = fd.SpatialCoordinate(mesh)

    xf = fd.Function(V).interpolate(x)
    yf = fd.Function(V).interpolate(y)

    xs_local = xf.dat.data_ro_with_halos[boundary_dofs]
    ys_local = yf.dat.data_ro_with_halos[boundary_dofs]

    coords_local = np.column_stack((xs_local, ys_local))

    # --- 3. Function values (direct DOF access) ---
    values_all_local = fun.dat.data_ro_with_halos

    if fun.function_space().value_size > 1:
        values_local = values_all_local[boundary_dofs][:, fun_index]
    else:
        values_local = values_all_local[boundary_dofs]

    # --- 4. Gather to rank 0 ---
    coords_all = comm.gather(coords_local, root=0)
    values_all = comm.gather(values_local, root=0)

    # --- 5. Only rank 0 continues ---
    if comm.rank != 0:
        return None, None

    coords_all = np.vstack(coords_all)
    values_all = np.concatenate(values_all)

    # --- 6. Remove duplicates (MPI ghost DOFs) ---
    coords_all = np.ascontiguousarray(coords_all)
    _, unique_idx = np.unique(coords_all, axis=0, return_index=True)

    # preserve ordering
    unique_idx = np.sort(unique_idx)

    coords_all = coords_all[unique_idx]
    values_all = values_all[unique_idx]

    return coords_all, values_all
