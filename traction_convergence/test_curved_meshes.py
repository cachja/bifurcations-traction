import firedrake as fd
import turek_traction_utils

circle_init = 8
max_ref = 2
order = 9

_, circle_points = turek_traction_utils.generate_ngmesh_manual(circle_init)
ngmsh = turek_traction_utils.generate_ngmesh_spline()
assert circle_init % 8 == 0, "circle_init must be divisible by 8"
circle_points = circle_points[::circle_init // 8]
subngmsh = turek_traction_utils.create_boundary_layer_submesh(ngmsh)
fd.info(f"{circle_points=}")  # In these points we output pointwise traction values

fdmesh = fd.Mesh(ngmsh, comm=fd.COMM_WORLD)
cf = fdmesh.curve_field(order)
mesh = fd.Mesh(cf)
submesh = fd.Mesh(subngmsh)
hierarchy, interpolation_hierarchy, subhierarchy = \
   turek_traction_utils.turek_computational_interpolation_hierarchy(ngmsh, max_ref, order)
hierarchy.insert(0, mesh)

# Create function spaces on hierarchies beforehand
interpolation_hierarchy_space_CG1 = []
interpolation_hierarchy_space_DG1 = []
for _mesh in interpolation_hierarchy:
    interpolation_hierarchy_space_CG1.append(
        fd.VectorFunctionSpace(_mesh, "CG", 1))
    interpolation_hierarchy_space_DG1.append(
        fd.VectorFunctionSpace(_mesh, "DG", 1))

fd.info("Hierarchy created")
