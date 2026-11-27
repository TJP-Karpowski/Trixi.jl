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
    # With point values (fourth-order moment inversion and conversion of the face
    # fluxes to face averages): order 4
    for (reconstruction, moment_inversion, order) in ((CombinedScheme(ReconstructionWENO5Z(),
                                                                      GradientCentral(6)),
                                                       MomentInversionIdentity(), 5),
                                                      (CombinedScheme(ReconstructionCentral(4),
                                                                      GradientCentral(4)),
                                                       MomentInversionIdentity(), 4),
                                                      (CombinedScheme(ReconstructionMUSCL(),
                                                                      GradientCentral(2)),
                                                       MomentInversionIdentity(), 2),
                                                      (CombinedScheme(ReconstructionWENO5Z(),
                                                                      GradientCentral(4)),
                                                       MomentInversionCentral(order = 4),
                                                       4))
        errors = map((2, 3)) do level
            mod = Module()
            trixi_include(mod, joinpath(EXAMPLES_DIR, "elixir_advection_diffusion.jl");
                          initial_refinement_level = level, tspan = (0.0, 0.25),
                          reconstruction, moment_inversion, diffusivity = 0.1,
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
    function rhs_transposed(initial_condition, advection_velocity, transpose,
                            moment_inversion)
        mod = Module()
        trixi_include(mod, joinpath(EXAMPLES_DIR, "elixir_advection_diffusion.jl");
                      initial_condition, advection_velocity, moment_inversion,
                      initial_refinement_level = 2, tspan = (0.0, 0.0))
        semi = Base.invokelatest(getglobal, mod, :semi)
        u_ode = Trixi.compute_coefficients(0.0, semi)
        du_ode = similar(u_ode)
        Trixi.rhs_macrocellfv!(du_ode, u_ode, semi, 0.0)
        # The subcells sorted by their coordinates
        mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
        coordinates = reshape(cache.elements.node_coordinates, 2, :)
        order = sortperm(collect(eachcol(transpose ? coordinates[[2, 1], :] :
                                         coordinates)))
        return du_ode[order], du_ode, u_ode, semi
    end
    # Averages and point values (moment inversion and conversion of the face fluxes)
    for moment_inversion in (MomentInversionIdentity(),
                             MomentInversionCentral(order = 4))
        du_x, du_ode, u_ode, semi = rhs_transposed(initial_condition_x, (0.7, -0.4),
                                                   false, moment_inversion)
        du_y, _ = rhs_transposed(initial_condition_y, (-0.4, 0.7), true,
                                 moment_inversion)
        @test maximum(abs, du_x - du_y) < 1.0e-13
        # Conservation: the fluxes at the faces at -x and -y are those of the neighbors
        @test abs(sum(du_ode)) < 1.0e-12

        # The buffers of the threads are reused for all elements, and only the pad
        # cells read by the reconstruction are filled (no corner subcells): the result
        # does not depend on their previous content
        for buffers in (semi.cache.pad_x_threaded, semi.cache.pad_y_threaded,
                        semi.cache.pad_points_x_threaded,
                        semi.cache.pad_points_y_threaded,
                        semi.cache.tangential_x_threaded,
                        semi.cache.tangential_y_threaded, semi.cache.face_flux_threaded,
                        semi.cache.face_flux_average_threaded)
            foreach(buffer -> fill!(buffer, NaN), buffers)
        end
        foreach(buffers -> foreach(b -> fill!(b, NaN), buffers),
                semi.cache.line_buffers_threaded)
        du_nan = similar(u_ode)
        Trixi.rhs_macrocellfv!(du_nan, u_ode, semi, 0.0)
        @test du_nan == du_ode
    end
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

@testitem "MacroCellFV 2D: elixir_euler_vortex.jl" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
                        l2=[
                            2.1581585701611975e-5,
                            0.0009258993877840325,
                            0.0009276994820451509,
                            0.0026075664927432273
                        ],
                        linf=[
                            0.0005215480067244371,
                            0.01157819057217474,
                            0.011544825439953915,
                            0.05309742322436506
                        ],
                        tspan=(0.0, 0.2))
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 2D: elixir_euler_vortex.jl convergence" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    # Point values: fourth order for the nonlinear fluxes; the averages are only second
    # order (not yet in the asymptotic range on these meshes, but much less accurate)
    function vortex_errors(moment_inversion, center2recon, face2flux)
        return map((3, 4)) do level
            mod = Module()
            trixi_include(mod, joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl");
                          initial_refinement_level = level, tspan = (0.0, 0.5),
                          moment_inversion, center2recon, face2flux,
                          analysis_interval = 10^6)
            subcell_average_error(mod)
        end
    end
    errors_points = vortex_errors(MomentInversionCentral(order = 4), cons2prim,
                                  prim2cons)
    @test log2(errors_points[1] / errors_points[2]) > 3.7
    errors_averages = vortex_errors(MomentInversionIdentity(), cons2cons, cons2cons)
    @test errors_averages[2] > 3 * errors_points[2]
end

@testitem "MacroCellFV 2D: elixir_navierstokes_convergence_periodic.jl" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[
                            1.347014954619707e-5,
                            0.0016457931334859777,
                            0.0016415763807729124,
                            0.002045876981800659
                        ],
                        linf=[
                            5.3442985353147066e-5,
                            0.003267658689079189,
                            0.0032371920241385244,
                            0.005255263958912959
                        ],
                        tspan=(0.0, 0.1))
    @test_allocations(Trixi.rhs_macrocellfv!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 2D: elixir_navierstokes_convergence_periodic.jl convergence" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    # Point values with fourth-order derivatives along and across the faces (order 4);
    # conservative reconstruction of the averages with the chain rule for the gradients
    # of the primitive variables (order 2)
    muscl_c2 = CombinedScheme(ReconstructionMUSCL(), GradientCentral(2))
    for (kwargs, order) in (((;), 4),
                            ((; schemes = (muscl_c2, muscl_c2, muscl_c2, muscl_c2),
                              moment_inversion = MomentInversionIdentity(),
                              center2recon = cons2cons,
                              face2flux = face2flux_navier_stokes_cons2prim), 2))
        errors = map((2, 3)) do level
            mod = Module()
            trixi_include(mod,
                          joinpath(EXAMPLES_DIR,
                                   "elixir_navierstokes_convergence_periodic.jl");
                          initial_refinement_level = level, tspan = (0.0, 0.25),
                          analysis_interval = 10^6, kwargs...)
            # The x-momentum (the density is constant in space)
            sol = Base.invokelatest(getglobal, mod, :sol)
            semi = Base.invokelatest(getglobal, mod, :semi)
            u_exact = Base.invokelatest(Trixi.compute_coefficients, sol.t[end], semi)
            difference = reshape(sol.u[end] - u_exact, 4, :)
            sqrt(sum(abs2, difference[2, :]) / size(difference, 2))
        end
        @test log2(errors[1] / errors[2]) > order - 0.3
    end
end

@testitem "MacroCellFV 2D: Navier-Stokes symmetry and conservation" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    # Swapping the coordinates and the velocity components of the initial condition
    # transposes the right-hand side and swaps its momentum components (a test for
    # mixed-up directions of the derivatives along the faces)
    function initial_condition_x(x, t, equations)
        rho = 1.5 + 0.2 * sinpi(x[1]) * cospi(2 * x[2])
        v1 = 0.3 * cospi(x[2]) + 0.1 * sinpi(x[1] + x[2])
        v2 = -0.2 * sinpi(2 * x[1]) * sinpi(x[2])
        p = 2 + 0.3 * cospi(x[1]) * sinpi(x[2])
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end
    function initial_condition_y(x, t, equations)
        u = initial_condition_x(SVector(x[2], x[1]), t, equations)
        return SVector(u[1], u[3], u[2], u[4])
    end
    for (moment_inversion, center2recon, face2flux, schemes) in ((MomentInversionCentral(order = 4),
                                                                  cons2prim_temperature_pressure,
                                                                  face2flux_navier_stokes_prim2cons,
                                                                  nothing),
                                                                 (MomentInversionIdentity(),
                                                                  cons2cons,
                                                                  face2flux_navier_stokes_cons2prim,
                                                                  ntuple(_ -> CombinedScheme(ReconstructionWENO5Z(),
                                                                                             GradientCentral(4)),
                                                                         4)))
        results = map(((initial_condition_x, false), (initial_condition_y, true))) do (initial_condition,
                                                                                       transpose)
            mod = Module()
            kwargs = schemes === nothing ? (;) : (; schemes)
            trixi_include(mod,
                          joinpath(EXAMPLES_DIR,
                                   "elixir_navierstokes_convergence_periodic.jl");
                          initial_condition, moment_inversion, center2recon, face2flux,
                          source_terms = nothing,
                          tspan = (0.0, 0.0), kwargs...)
            semi = Base.invokelatest(getglobal, mod, :semi)
            u_ode = Trixi.compute_coefficients(0.0, semi)
            du_ode = similar(u_ode)
            Trixi.rhs_macrocellfv!(du_ode, u_ode, semi, 0.0)
            mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
            coordinates = reshape(cache.elements.node_coordinates, 2, :)
            order = sortperm(collect(eachcol(transpose ? coordinates[[2, 1], :] :
                                             coordinates)))
            du = reshape(du_ode, 4, :)[:, order]
            if transpose
                du = du[[1, 3, 2, 4], :]
            end
            du, du_ode
        end
        (du_x, du_ode), (du_y, _) = results
        @test maximum(abs, du_x - du_y) < 1.0e-12
        # Conservation of all variables
        @test maximum(abs, sum(reshape(du_ode, 4, :), dims = 2)) < 1.0e-11
    end
end

@testitem "MacroCellFV 2D: unsupported Navier-Stokes setups" setup=[
    Setup,
    TreeMesh2DMacroCellFV
] tags=[:tree_part1] begin
    # The error thrown while including the elixir (wrapped in a `LoadError`)
    function include_error(; kwargs...)
        try
            trixi_include(Module(),
                          joinpath(EXAMPLES_DIR,
                                   "elixir_navierstokes_convergence_periodic.jl");
                          tspan = (0.0, 0.0), kwargs...)
        catch err
            return err isa LoadError ? err.error : err
        end
        return nothing
    end
    # Derivatives along the faces of order 6 are not implemented yet
    c6 = CombinedScheme(ReconstructionWENO5Z(), GradientCentral(6))
    @test include_error(schemes = (c6, c6, c6, c6, c6)) isa ArgumentError
    # The entropy variables as gradient variables are not supported
    equations_parabolic = CompressibleNavierStokesDiffusion2D(CompressibleEulerEquations2D(1.4),
                                                              mu = 0.01, Prandtl = 0.72,
                                                              gradient_variables = GradientVariablesEntropy())
    @test include_error(; equations_parabolic) isa ArgumentError
end
