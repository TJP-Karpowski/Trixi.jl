@testsnippet TreeMesh2DMacroCellFV begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_macrocellfv")

    # See `test_tree_1d_macrocellfv.jl`: the kernels do not allocate, `@batch` allocates
    # a fixed amount with more than one thread
    ALLOCATION_LIMIT = Threads.nthreads() == 1 ? 1000 : 2000

    # L2 error of the first variable against the exact subcell averages
    function subcell_average_error(sol, semi)
        # `wrap_array_native` does not keep the wrapped vectors alive
        u_ode = sol.u[end]
        u_exact_ode = Trixi.compute_coefficients(sol.t[end], semi)
        GC.@preserve u_ode u_exact_ode begin
            u = Trixi.wrap_array_native(u_ode, semi)
            u_exact = Trixi.wrap_array_native(u_exact_ode, semi)
            difference = selectdim(u, 1, 1) .- selectdim(u_exact, 1, 1)
            return sqrt(sum(abs2, difference) / length(difference))
        end
    end

    # The same for `sol` and `semi` defined in `mod` by a preceding `trixi_include`
    function subcell_average_error(mod::Module)
        sol = Base.invokelatest(getglobal, mod, :sol)
        semi = Base.invokelatest(getglobal, mod, :semi)
        return Base.invokelatest(subcell_average_error, sol, semi)
    end
end

@testitem "MacroCellFV 2D: elixir_advection_basic.jl" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.00028403191271788566],
                        linf=[0.0004016796490806662])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 2D: ReconstructionConstant agrees with BlockFV" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    # With piecewise constant reconstruction, MacroCellFV performs the same operations
    # as BlockFV
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        tspan=(0.0, 0.1), initial_condition_nodes=1,
                        solver=MacroCellFV(n_cells = 8,
                                           surface_flux = flux_lax_friedrichs,
                                           reconstruction = ReconstructionConstant()))
    u_ode = compute_coefficients(0.0, semi)
    du_macrocell = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_macrocell, u_ode, semi, 0.0)

    semi_blockfv = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                                BlockFV(n_nodes = n_cells,
                                                        surface_flux = flux_lax_friedrichs);
                                                boundary_conditions = boundary_condition_periodic)
    du_blockfv = similar(u_ode)
    ode_blockfv = semidiscretize(semi_blockfv, (0.0, 0.1))
    ode_blockfv.f(du_blockfv, u_ode, semi_blockfv, 0.0)
    @test du_macrocell≈du_blockfv atol=1.0e-13
end

@testitem "MacroCellFV 2D: elixir_advection_diffusion.jl" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion.jl"),
                        l2=[0.0001819255031322543],
                        linf=[0.000362976391015013],
                        tspan=(0.0, 0.5))
    @test_allocations(Trixi.rhs_macrocellfv!, semi, sol, ALLOCATION_LIMIT)
    # The diffusive time step: the eigenvalues of both directions add up
    u = Trixi.wrap_array(ode.u0, semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    h = 2 / (sqrt(nelements(solver, cache)) * n_cells)
    dt_parabolic = Trixi.max_dt(u, 0.0, mesh, Trixi.True(), equations,
                                equations_parabolic, solver, cache)
    @test dt_parabolic ≈
          h^2 / (4 * diffusivity * solver.volume_integral.diffusion_dt_factor)
end

@testitem "MacroCellFV 2D: elixir_advection_diffusion.jl convergence" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    # The face values and derivatives of the subcell averages along the lines give the
    # face averages of the linear fluxes to the order of the stencils: WENO5-Z with
    # sixth-order derivatives (order 5), central values and derivatives of order 4,
    # MUSCL with second-order derivatives
    for (reconstruction, order) in ((CombinedScheme(ReconstructionWENO5Z(),
                                                    GradientCentral(6)), 5),
                                    (CombinedScheme(ReconstructionCentral(4),
                                                    GradientCentral(4)), 4),
                                    (CombinedScheme(ReconstructionMUSCL(),
                                                    GradientCentral(2)), 2))
        errors = map((2, 3)) do level
            mod = Module()
            trixi_include(mod, joinpath(EXAMPLES_DIR, "elixir_advection_diffusion.jl");
                          initial_refinement_level = level, tspan = (0.0, 0.25),
                          reconstruction, diffusivity = 0.1,
                          advection_velocity = (0.1, 0.05), analysis_interval = 10^6,
                          stepsize_callback = StepsizeCallback(cfl = 0.2,
                                                               cfl_parabolic = 0.1))
            subcell_average_error(mod)
        end
        @test log2(errors[1] / errors[2]) > order - 0.3
    end
end

@testitem "MacroCellFV 2D: symmetry, conservation, and thread-local pads" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    # The x- and y-lines are computed by the same kernels: swapping the coordinates of
    # the initial condition and the components of the advection velocity transposes
    # the right-hand side (a test for mixed-up directions)
    initial_condition_x(x, t, equations) = SVector(1 + 0.3 * sinpi(x[1]) * cospi(2 * x[2]) +
                                                   0.1 * sinpi(3 * x[2]))
    initial_condition_y(x, t, equations) = initial_condition_x(SVector(x[2], x[1]), t,
                                                               equations)
    function rhs_transposed(initial_condition, advection_velocity, transpose)
        mod = Module()
        trixi_include(mod, joinpath(EXAMPLES_DIR, "elixir_advection_diffusion.jl");
                      initial_condition, advection_velocity, initial_refinement_level = 2,
                      tspan = (0.0, 0.0))
        semi = Base.invokelatest(getglobal, mod, :semi)
        u_ode = Trixi.compute_coefficients(0.0, semi)
        du_ode = similar(u_ode)
        Trixi.rhs_macrocellfv!(du_ode, u_ode, semi, 0.0)
        # The subcells sorted by their coordinates
        mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
        coordinates = reshape(cache.elements.node_coordinates, 2, :)
        order = sortperm(collect(eachcol(transpose ? coordinates[[2, 1], :] :
                                         coordinates)))
        return du_ode[order], du_ode
    end
    du_x, du_ode = rhs_transposed(initial_condition_x, (0.7, -0.4), false)
    du_y, _ = rhs_transposed(initial_condition_y, (-0.4, 0.7), true)
    @test maximum(abs, du_x - du_y) < 1.0e-13
    # Conservation: the fluxes at the faces at -x and -y are those of the neighbors
    @test abs(sum(du_ode)) < 1.0e-12

    # The pads of the threads are reused for all elements, and only the pad cells read
    # by the reconstruction are filled: the result does not depend on their previous
    # content
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion.jl"),
                        tspan=(0.0, 0.0))
    u_ode = compute_coefficients(0.0, semi)
    du_first = similar(u_ode)
    Trixi.rhs_macrocellfv!(du_first, u_ode, semi, 0.0)
    for pads in (semi.cache.pad_x_threaded, semi.cache.pad_y_threaded)
        foreach(pad -> fill!(pad, NaN), pads)
    end
    foreach(buffers -> foreach(b -> fill!(b, NaN), buffers),
            semi.cache.line_buffers_threaded)
    du_ode = similar(u_ode)
    Trixi.rhs_macrocellfv!(du_ode, u_ode, semi, 0.0)
    @test du_ode == du_first
end

@testitem "MacroCellFV 2D: unsupported setups" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    solver = MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                         reconstruction = ReconstructionWENO5Z())
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level = 1,
                    periodicity = false)
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh, equations,
                                                            initial_condition_convergence_test,
                                                            solver;
                                                            boundary_conditions = BoundaryConditionDirichlet(initial_condition_convergence_test))
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level = 1,
                    periodicity = true)
    solver = MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                         reconstruction = ReconstructionWENO5Z(),
                         moment_inversion = MomentInversionCentral())
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh, equations,
                                                            initial_condition_convergence_test,
                                                            solver;
                                                            boundary_conditions = boundary_condition_periodic)
end
