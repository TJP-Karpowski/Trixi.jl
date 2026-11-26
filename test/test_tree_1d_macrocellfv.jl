@testsnippet TreeMesh1DMacroCellFV begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_macrocellfv")

    # The kernels do not allocate. With more than one thread, Polyester.jl's `@batch`
    # allocates a fixed amount for each of the two `@threaded` loops of the RHS
    # (independent of the number of threads and elements; DGSEM allocates a similar
    # amount for its loops), so threaded runs get a higher limit.
    ALLOCATION_LIMIT = Threads.nthreads() == 1 ? 1000 : 2000

    # L2 error of the first variable against the exact subcell averages (the errors of
    # the `AnalysisCallback` compare the averages with point values, which is only
    # second-order accurate)
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

@testitem "MacroCellFV 1D: elixir_advection_gaussian.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_gaussian.jl"),
                        l2=[0.12586020153550745],
                        linf=[0.3726415651944912],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: ReconstructionConstant agrees with BlockFV" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # With identity moment inversion and piecewise constant reconstruction, MacroCellFV
    # performs the same operations as BlockFV.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_gaussian.jl"),
                        tspan=(0.0, 0.1))
    u_ode = compute_coefficients(0.0, semi)
    du_macrocell = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_macrocell, u_ode, semi, 0.0)

    semi_blockfv = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                                BlockFV(n_nodes = n_cells,
                                                        surface_flux = flux_lax_friedrichs);
                                                boundary_conditions = boundary_condition_periodic)
    du_blockfv = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_blockfv, u_ode, semi_blockfv, 0.0)
    @test du_macrocell == du_blockfv
end

@testitem "MacroCellFV 1D: elixir_advection_sine.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        l2=[0.0008971312291256202],
                        linf=[0.0012687102434813802],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_sine.jl with minmod" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        l2=[0.007633348149705237],
                        linf=[0.02173487014838904],
                        slope_limiter=minmod,
                        tspan=(0.0, 0.25))
end

@testitem "MacroCellFV 1D: elixir_advection_sine.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Second-order convergence of MUSCL with unlimited central slopes
    eocs, errors = convergence_test(joinpath(EXAMPLES_DIR,
                                             "elixir_advection_sine.jl"),
                                    3, initial_refinement_level = 2)
    @test all(eocs[:l2] .> 1.9)
    @test all(eocs[:linf] .> 1.9)
end

@testitem "MacroCellFV 1D: initial condition as subcell averages" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Maximum error of the initial data against the exact subcell averages of
    # sin(2πx) on 8 * 2^level subcells
    function initial_average_error(level, initial_condition_nodes)
        equations = LinearScalarAdvectionEquation1D(1.0)
        initial_condition(x, t, equations) = SVector(sinpi(2 * x[1]))
        solver = MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                             initial_condition_nodes = initial_condition_nodes)
        mesh = TreeMesh(0.0, 1.0, initial_refinement_level = level, periodicity = true)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                            boundary_conditions = boundary_condition_periodic)
        # `wrap_array_native` does not keep `u_ode` alive
        u_ode = compute_coefficients(0.0, semi)
        GC.@preserve u_ode begin
            u = Trixi.wrap_array_native(u_ode, semi)
            x = semi.cache.elements.node_coordinates # subcell centers
            h = 1 / (8 * 2^level)
            u_exact = @. (cospi(2 * (x - h / 2)) - cospi(2 * (x + h / 2))) / (2 * pi * h)
            return maximum(abs, u .- u_exact)
        end
    end

    # Point values at the subcell centers: second order
    errors = [initial_average_error(level, 1) for level in 2:3]
    @test 1.9 < log2(errors[1] / errors[2]) < 2.1
    # Gauss-Legendre quadrature with 3 nodes: sixth order
    errors = [initial_average_error(level, 3) for level in 2:3]
    @test log2(errors[1] / errors[2]) > 5.8
end

@testitem "MacroCellFV 1D: constant state and block decomposition" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # A constant state is preserved exactly
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        initial_condition=initial_condition_constant,
                        tspan=(0.0, 0.1))
    du_ode = similar(sol.u[end])
    Trixi.rhs_hyperbolic!(du_ode, sol.u[end], semi, 0.0)
    @test maximum(abs, du_ode) == 0
    @test maximum(abs, sol.u[end] .- sol.u[1]) < 1.0e-14

    # The same 64 global subcells split into different numbers of blocks give the same
    # solution up to round-off errors.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        slope_limiter=minmod, n_cells=32,
                        initial_refinement_level=1, tspan=(0.0, 0.1))
    u_2blocks = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        slope_limiter=minmod, n_cells=4,
                        initial_refinement_level=4, tspan=(0.0, 0.1))
    @test isapprox(u_2blocks, sol.u[end]; atol = 1.0e-14, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_euler_sod.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sod.jl"),
                        l2=[0.15705319306155355, 0.16981100736981203, 0.450050048499648],
                        linf=[0.537210827922814, 0.3971661065734396, 1.4838542544932765],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)

    # Mass and energy are conserved (the waves have not reached the transmissive
    # boundaries yet); the momentum changes by the boundary pressure forces.
    integrals_0 = Trixi.integrate(sol.u[1], semi; normalize = false)
    integrals = Trixi.integrate(sol.u[end], semi; normalize = false)
    @test isapprox(integrals[1], integrals_0[1]; atol = 1.0e-14)
    @test isapprox(integrals[2] - integrals_0[2], (1 - 0.1) * 0.1; atol = 1.0e-14)
    @test isapprox(integrals[3], integrals_0[3]; atol = 1.0e-14)
end

@testitem "MacroCellFV 1D: elixir_euler_sod.jl with HLLC and van Leer" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sod.jl"),
                        l2=[0.15963946970756976, 0.1718073269196492, 0.45428878543978385],
                        linf=[0.5578053992129199, 0.39701672781070857, 1.5258586868699129],
                        surface_flux=flux_hllc, slope_limiter=vanleer,
                        tspan=(0.0, 0.1))
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        l2=[0.0006554009131478032],
                        linf=[0.0012318160241164833])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl with WENO5-JS" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        l2=[0.0008736186419141591],
                        linf=[0.002023947818738492],
                        reconstruction=ReconstructionWENO5JS())
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Orders of the errors against the exact subcell averages, with a small time step
    # (fourth-order time integration): fifth order of WENO5-Z; WENO5-JS loses accuracy
    # at the critical points of the solution on these coarse meshes
    for (reconstruction, eoc_min) in ((ReconstructionWENO5Z(), 4.8),
                                      (ReconstructionWENO5JS(), 4.2))
        errors = map((3, 4)) do level
            trixi_include(@__MODULE__,
                          joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl");
                          reconstruction, initial_refinement_level = level,
                          stepsize_callback = StepsizeCallback(cfl = 0.2))
            return subcell_average_error(@__MODULE__)
        end
        @test log2(errors[1] / errors[2]) > eoc_min
    end
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl convergence with Dirichlet boundaries" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Dirichlet boundaries with the exact solution: the ghost subcells hold the averages
    # of the boundary values (with the quadrature of the initial condition), so that
    # WENO5-Z keeps its order (point values at the ghost subcell centers would reduce
    # it to second order)
    errors = map((3, 4)) do level
        trixi_include(@__MODULE__,
                      joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl");
                      initial_refinement_level = level, periodicity = false,
                      stepsize_callback = StepsizeCallback(cfl = 0.2))
        return subcell_average_error(@__MODULE__)
    end
    @test log2(errors[1] / errors[2]) > 4.5
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl block decomposition" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The same subcells split into blocks of 3 subcells (the halo width of WENO5, so
    # that the halos span whole neighbors) and into 1 block give the same
    # solution up to round-off errors (the subcell centers, and thus the initial
    # subcell averages, differ by a few ulps between the decompositions, which the
    # nonlinear WENO weights amplify slightly; an error in the halos would give
    # differences of the order of the truncation error): periodic advection with 48
    # subcells
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        n_cells=48, initial_refinement_level=0, tspan=(0.0, 0.1))
    u_1block = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        n_cells=3, initial_refinement_level=4, tspan=(0.0, 0.1))
    @test isapprox(u_1block, sol.u[end]; atol = 1.0e-13, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl block decomposition" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The same 96 subcells split into 2 blocks and into blocks of 3 subcells (the halo
    # width of WENO5) give the same solution up to round-off errors, also with ghost
    # subcells at the transmissive boundaries
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        n_cells=48, initial_refinement_level=1, tspan=(0.0, 0.1))
    u_2blocks = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        n_cells=3, initial_refinement_level=5, tspan=(0.0, 0.1))
    @test isapprox(u_2blocks, sol.u[end]; atol = 1.0e-13, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        l2=[1.1929652719920263, 4.23670862934106, 15.323415000350867],
                        linf=[3.25549904946853, 11.435937173461083, 38.31689116352348],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: WENO5 face values" setup=[Setup] tags=[:tree_part1] begin
    # Mirrored data give mirrored face values (e.g., at reflecting walls), up to round-off
    # errors (fused multiply-add operations may differ between the two faces)
    data = ((1.0, 2.0, 4.0, 3.5, -1.0), (0.3, -0.2, 0.1, 5.0, 7.0),
            (1.0, 1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 0.0, 0.0))
    for reconstruction in (ReconstructionWENO5JS(), ReconstructionWENO5Z()),
        values in data

        w_minus, w_plus = Trixi.weno5_cell(reconstruction, values...)
        w_minus_mirror, w_plus_mirror = Trixi.weno5_cell(reconstruction,
                                                         reverse(values)...)
        @test w_minus≈w_plus_mirror rtol=10 * eps(Float64)
        @test w_plus≈w_minus_mirror rtol=10 * eps(Float64)
    end
    # Fifth order of both face values for smooth data: averages of exp(x) over the
    # cells [(i - 1/2) h, (i + 1/2) h], whose face values are exp(-h / 2), exp(h / 2)
    function face_errors(reconstruction, h)
        averages = Tuple((exp((i + 0.5) * h) - exp((i - 0.5) * h)) / h for i in -2:2)
        w_minus, w_plus = Trixi.weno5_cell(reconstruction, averages...)
        return abs(w_minus - exp(-h / 2)), abs(w_plus - exp(h / 2))
    end
    for reconstruction in (ReconstructionWENO5JS(), ReconstructionWENO5Z())
        errors_coarse = face_errors(reconstruction, 0.1)
        errors_fine = face_errors(reconstruction, 0.05)
        @test all(log2.(errors_coarse ./ errors_fine) .> 4.8)
    end
    # Float32: a constant state next to a step (zero smoothness indicators of two
    # substencils) gives finite face values with the default `epsilon` of WENO-Z
    for reconstruction in (ReconstructionWENO5JS(), ReconstructionWENO5Z())
        w_minus, w_plus = Trixi.weno5_cell(reconstruction, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                                           1.0f0)
        @test w_minus isa Float32
        @test isfinite(w_minus) && isfinite(w_plus)
    end
end

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl with Float32" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # One right-hand side evaluation in Float32 for the initial condition, which has
    # constant states next to the shock
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        tspan=(0.0, 0.0))
    solver_f32 = MacroCellFV(n_cells = n_cells, surface_flux = surface_flux,
                             reconstruction = reconstruction,
                             initial_condition_nodes = 5, RealT = Float32)
    mesh_f32 = TreeMesh(-5.0f0, 5.0f0, initial_refinement_level = 5,
                        periodicity = false, RealT = Float32)
    equations_f32 = CompressibleEulerEquations1D(1.4f0)
    semi_f32 = SemidiscretizationHyperbolic(mesh_f32, equations_f32, initial_condition,
                                            solver_f32;
                                            boundary_conditions = boundary_condition_do_nothing)
    u_ode = compute_coefficients(0.0f0, semi_f32)
    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi_f32, 0.0f0)
    @test eltype(du_ode) == Float32
    @test all(isfinite, du_ode)
    # The same with the WENO moment inversion and the reconstruction of the primitive
    # variables
    solver_f32 = MacroCellFV(n_cells = n_cells, surface_flux = flux_hllc_primitive,
                             reconstruction = reconstruction,
                             moment_inversion = MomentInversionWENO(),
                             center2recon = cons2prim, initial_condition_nodes = 5,
                             RealT = Float32)
    semi_f32 = SemidiscretizationHyperbolic(mesh_f32, equations_f32, initial_condition,
                                            solver_f32;
                                            boundary_conditions = boundary_condition_do_nothing)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi_f32, 0.0f0)
    @test all(isfinite, du_ode)
end

@testitem "MacroCellFV 1D: moment inversions" setup=[Setup] tags=[:tree_part1] begin
    # Coefficients of the central moment inversion: order 4 is `ū - Δ²ū / 24`, order 6
    # is `(9, -116, 2134, -116, 9) / 1920`
    @test MomentInversionCentral(order = 4).coefficients == (-1 / 24,)
    @test MomentInversionCentral().order == 6
    @test all(MomentInversionCentral(order = 6).coefficients .≈ (-116 / 1920, 9 / 1920))
    @test_throws ArgumentError MomentInversionCentral(order = 5)
    @test_throws ArgumentError MomentInversionCentral(order = 2)
    @test_throws ArgumentError MomentInversionWENO(linear_weight = 0.5)

    # Exactness: the averages of `x^p` over the cells of unit width centered at
    # `x_0 + k` give the value at `x_0`, for `p < order` (central) and `p < 3` (WENO,
    # whose nonlinear weights are not symmetric for cubic data)
    average(p, x_0) = k -> ((x_0 + k + 0.5)^(p + 1) - (x_0 + k - 0.5)^(p + 1)) / (p + 1)
    for order in (4, 6, 8)
        inversion = MomentInversionCentral(; order)
        @test Trixi.nhalo(inversion) == order ÷ 2 - 1
        for p in 0:(order - 1)
            @test Trixi.center_value(inversion, average(p, 0.3))≈0.3^p atol=1.0e-14
        end
    end
    for p in 0:2
        @test Trixi.center_value(MomentInversionWENO(), average(p, 0.3)) ≈ 0.3^p
    end
    # Next to a jump, the WENO moment inversion keeps a uniform state
    @test Trixi.center_value(MomentInversionWENO(), k -> k <= 1 ? 1.0 : 2.0) == 1.0
    # Mirror symmetry: mirrored data give the same point value (up to round-off errors)
    for inversion in (MomentInversionCentral(order = 8), MomentInversionWENO())
        data = (0.3, -0.2, 0.1, 5.0, 7.0, 2.0, -1.0)
        stencil = k -> data[k + 4]
        stencil_mirror = k -> data[4 - k]
        @test Trixi.center_value(inversion,
                                 stencil)≈Trixi.center_value(inversion,
                                                             stencil_mirror) rtol=10 *
                                                                                  eps(Float64)
    end

    # Point-value WENO5: exact for quadratic point values (each candidate is), fifth
    # order for smooth point values, mirror-symmetric, and finite in `Float32` for a
    # constant state next to a step
    for weno in (ReconstructionWENO5JS(), ReconstructionWENO5Z())
        reconstruction = Trixi.ReconstructionPointValues(weno)
        @test Trixi.nhalo(reconstruction) == 3
        for p in 0:2
            w_minus, w_plus = Trixi.reconstruct_cell(reconstruction,
                                                     k -> (0.3 + k)^p)
            @test w_minus≈(-0.2)^p atol=1.0e-14
            @test w_plus≈0.8^p atol=1.0e-14
        end
        function face_errors(h)
            w_minus, w_plus = Trixi.reconstruct_cell(reconstruction, k -> exp(k * h))
            return abs(w_minus - exp(-h / 2)), abs(w_plus - exp(h / 2))
        end
        @test all(log2.(face_errors(0.1) ./ face_errors(0.05)) .> 4.8)
        values = (0.3, -0.2, 0.1, 5.0, 7.0)
        w_minus, w_plus = Trixi.weno5_point_cell(weno, values...)
        w_minus_mirror, w_plus_mirror = Trixi.weno5_point_cell(weno, reverse(values)...)
        @test w_minus≈w_plus_mirror rtol=10 * eps(Float64)
        @test w_plus≈w_minus_mirror rtol=10 * eps(Float64)
        w_minus, w_plus = Trixi.weno5_point_cell(weno, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0)
        @test w_minus isa Float32
        @test isfinite(w_minus) && isfinite(w_plus)
    end

    # Point values outside of the admissible set are replaced by the averages
    equations = CompressibleEulerEquations1D(1.4)
    @test Trixi.macrocell_point_value_admissible(SVector(1.0, 0.0, 2.5), equations)
    @test !Trixi.macrocell_point_value_admissible(SVector(-1.0, 0.0, 2.5), equations)
    @test !Trixi.macrocell_point_value_admissible(SVector(1.0, 0.0, -2.5), equations)
    @test Trixi.macrocell_point_value_admissible(SVector(-1.0),
                                                 LinearScalarAdvectionEquation1D(1.0))
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl convergence with moment inversions" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # WENO5-Z interpolation of the point values from the central moment inversion of
    # the default order 6: fifth order (the exactness of all moment inversions is tested
    # in "moment inversions")
    errors = map((3, 4)) do level
        trixi_include(@__MODULE__,
                      joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl");
                      moment_inversion = MomentInversionCentral(),
                      initial_refinement_level = level,
                      stepsize_callback = StepsizeCallback(cfl = 0.2))
        return subcell_average_error(@__MODULE__)
    end
    @test log2(errors[1] / errors[2]) > 4.5
end

@testitem "MacroCellFV 1D: block decomposition with moment inversions" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # As for the reconstructions from the averages: blocks of the halo width (5 subcells
    # for WENO5 with the WENO moment inversion or the central moment inversion of
    # order 6) and larger blocks give the same solution up to round-off errors (the
    # Shu-Osher problem until t = 0.02: later, the unlimited central moment inversion at
    # the shock amplifies the round-off differences, to about 5e-12 at t = 0.1; an error
    # in the halos would give differences of the order of the truncation error at all
    # element boundaries from the first time step on)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        n_cells=80, initial_refinement_level=0, tspan=(0.0, 0.1),
                        moment_inversion=MomentInversionWENO())
    u_1block = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        n_cells=5, initial_refinement_level=4, tspan=(0.0, 0.1),
                        moment_inversion=MomentInversionWENO())
    @test isapprox(u_1block, sol.u[end]; atol = 1.0e-13, rtol = 0)

    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        n_cells=80, initial_refinement_level=1, tspan=(0.0, 0.02),
                        moment_inversion=MomentInversionCentral(),
                        center2recon=cons2prim, surface_flux=flux_hllc_primitive)
    u_2blocks = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        n_cells=5, initial_refinement_level=5, tspan=(0.0, 0.02),
                        moment_inversion=MomentInversionCentral(),
                        center2recon=cons2prim, surface_flux=flux_hllc_primitive)
    @test isapprox(u_2blocks, sol.u[end]; atol = 1.0e-13, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl with point values" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # WENO moment inversion, reconstruction of the primitive variables, and the HLLC
    # flux from the primitive variables
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        l2=[1.1944607245801735, 4.241560030318569, 15.336162283766798],
                        linf=[3.258848701586172, 11.442063287669447, 38.37150130144202],
                        tspan=(0.0, 0.5), moment_inversion=MomentInversionWENO(),
                        center2recon=cons2prim, surface_flux=flux_hllc_primitive)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
    # `prim2cons` at the faces and the HLLC flux of the conservative variables give the
    # same right-hand side up to round-off errors
    solver_conservative_flux = MacroCellFV(n_cells = n_cells, surface_flux = flux_hllc,
                                           reconstruction = reconstruction,
                                           moment_inversion = MomentInversionWENO(),
                                           center2recon = cons2prim,
                                           face2flux = prim2cons,
                                           initial_condition_nodes = 5)
    semi_conservative_flux = SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver_conservative_flux;
                                                          boundary_conditions = boundary_condition_do_nothing)
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    du_conservative_flux = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.5)
    Trixi.rhs_hyperbolic!(du_conservative_flux, u_ode, semi_conservative_flux, 0.5)
    @test isapprox(du_ode, du_conservative_flux; rtol = 1.0e-12,
                   atol = 1.0e-12 * maximum(abs, du_ode))
end

@testitem "MacroCellFV 1D: types" setup=[Setup] tags=[:tree_part1] begin
    solver = MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                         reconstruction = ReconstructionMUSCL(vanleer))
    @test solver isa MacroCellFV
    @test solver isa BlockFV
    @test solver.volume_integral isa VolumeIntegralMacroCellFV
    @test !(BlockFV(n_nodes = 4, surface_flux = flux_lax_friedrichs) isa MacroCellFV)
    @test Trixi.nnodes(solver) == 4
    @test Trixi.nhalo(solver) == 2
    # `n_nodes` is accepted for consistency with `BlockFV`
    solver_nodes = MacroCellFV(n_nodes = 4, surface_flux = flux_lax_friedrichs,
                               reconstruction = ReconstructionMUSCL(vanleer))
    @test typeof(solver_nodes) == typeof(solver)
    @test Trixi.nnodes(solver_nodes) == 4
    @test_throws ArgumentError MacroCellFV(surface_flux = flux_lax_friedrichs)
    @test_throws ArgumentError MacroCellFV(n_cells = 4, n_nodes = 4,
                                           surface_flux = flux_lax_friedrichs)
    @test ReconstructionMUSCL().slope_limiter === minmod
    @test Trixi.nhalo(MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                  reconstruction = ReconstructionConstant())) == 1
    # The identity moment inversion adds no halo cells
    @test Trixi.nhalo(MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                  moment_inversion = MomentInversionIdentity())) == 2

    @test_throws ArgumentError MacroCellFV(n_cells = 4,
                                           surface_flux = flux_lax_friedrichs,
                                           initial_condition_nodes = 0)

    solver_weno = MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                              reconstruction = ReconstructionWENO5Z())
    @test Trixi.nhalo(solver_weno) == 3
    @test Trixi.nhalo(MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                  reconstruction = ReconstructionWENO5JS())) == 3
    @test ReconstructionWENO5JS().epsilon == 1.0e-6
    # WENO-Z: default `epsilon = 1.0e-40`, at least `sqrt(floatmin)` of the solution type
    @test ReconstructionWENO5Z().epsilon === nothing
    @test Trixi.weno5z_epsilon(nothing, Float64) === 1.0e-40
    @test Trixi.weno5z_epsilon(nothing, Float32) === sqrt(floatmin(Float32))
    @test ReconstructionWENO5Z(epsilon = 1.0f-20).epsilon isa Float32
    @test Trixi.weno5z_epsilon(1.0f-20, Float32) === 1.0f-20
    # The halos are copied from the direct neighbors, so the halo width is limited by
    # the number of subcells per macro-cell (MUSCL: 2 halo subcells)
    mesh = TreeMesh(0.0, 1.0, initial_refinement_level = 2, periodicity = true)
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh,
                                                            LinearScalarAdvectionEquation1D(1.0),
                                                            initial_condition_constant,
                                                            MacroCellFV(n_cells = 1,
                                                                        surface_flux = flux_lax_friedrichs,
                                                                        reconstruction = ReconstructionMUSCL());
                                                            boundary_conditions = boundary_condition_periodic)
    # WENO5: 3 halo subcells
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh,
                                                            LinearScalarAdvectionEquation1D(1.0),
                                                            initial_condition_constant,
                                                            MacroCellFV(n_cells = 2,
                                                                        surface_flux = flux_lax_friedrichs,
                                                                        reconstruction = ReconstructionWENO5Z());
                                                            boundary_conditions = boundary_condition_periodic)

    # Moment inversions to point values: their halo subcells add to those of WENO5;
    # only the WENO reconstructions are supported
    @test Trixi.nhalo(MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                                  reconstruction = ReconstructionWENO5Z(),
                                  moment_inversion = MomentInversionCentral(order = 4))) ==
          4
    @test Trixi.nhalo(MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                                  reconstruction = ReconstructionWENO5JS(),
                                  moment_inversion = MomentInversionCentral(order = 6))) ==
          5
    solver_point = MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                               reconstruction = ReconstructionWENO5Z(),
                               moment_inversion = MomentInversionWENO())
    @test Trixi.nhalo(solver_point) == 5
    @test solver_point.volume_integral.reconstruction isa
          Trixi.ReconstructionPointValues{<:ReconstructionWENO5Z}
    @test MomentInversionWENO().epsilon === nothing
    @test_throws ArgumentError MacroCellFV(n_cells = 8,
                                           surface_flux = flux_lax_friedrichs,
                                           reconstruction = ReconstructionMUSCL(),
                                           moment_inversion = MomentInversionCentral())
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh,
                                                            LinearScalarAdvectionEquation1D(1.0),
                                                            initial_condition_constant,
                                                            MacroCellFV(n_cells = 4,
                                                                        surface_flux = flux_lax_friedrichs,
                                                                        reconstruction = ReconstructionWENO5Z(),
                                                                        moment_inversion = MomentInversionWENO());
                                                            boundary_conditions = boundary_condition_periodic)

    for item in (solver, solver.volume_integral, solver_weno.volume_integral,
                 solver_point.volume_integral, MomentInversionCentral(),
                 MomentInversionWENO(), solver_point.volume_integral.reconstruction)
        @test_nowarn show(IOContext(IOBuffer(), :compact => true), MIME"text/plain"(),
                          item)
        @test_nowarn show(IOContext(IOBuffer(), :compact => false),
                          MIME"text/plain"(), item)
    end
end

@testitem "MacroCellFV 1D: thread-local pads" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The pads of the threads are reused for all elements; repeated evaluations of the
    # right-hand side give the same result (reconstruction of primitive variables,
    # physical boundaries)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sod.jl"),
                        tspan=(0.0, 0.1))
    u_ode = copy(sol.u[end])
    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    du_first = copy(du_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    @test du_ode == du_first
    # Only the halo cells read by the reconstruction are filled and transformed: with a
    # left neighbor, the first pad cell is neither filled nor read
    # (also the line buffers are written before they are read)
    foreach(pad -> fill!(pad, NaN), semi.cache.pad_threaded)
    foreach(buffers -> foreach(b -> fill!(b, NaN), buffers),
            semi.cache.line_buffers_threaded)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    @test du_ode == du_first
    # The pads hold `n_cells + 2 nhalo` subcells, each variable contiguously
    @test all(pad -> size(pad) ==
                     (n_cells + 2 * Trixi.nhalo(solver), nvariables(equations)),
              semi.cache.pad_threaded)
    @test length(semi.cache.pad_threaded) == Threads.maxthreadid()
    @test length(semi.cache.line_buffers_threaded) == Threads.maxthreadid()
    # Without point values, the pads of the point values are empty
    @test all(isempty, semi.cache.pad_points_threaded)

    # With point values: the point values are computed only in the pad cells read by
    # the reconstruction, from filled cells only
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        tspan=(0.0, 0.1), moment_inversion=MomentInversionWENO(),
                        center2recon=cons2prim, surface_flux=flux_hllc_primitive)
    u_ode = copy(sol.u[end])
    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    du_first = copy(du_ode)
    foreach(pad -> fill!(pad, NaN), semi.cache.pad_threaded)
    foreach(pad -> fill!(pad, NaN), semi.cache.pad_points_threaded)
    foreach(buffers -> foreach(b -> fill!(b, NaN), buffers),
            semi.cache.line_buffers_threaded)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    @test du_ode == du_first
    @test all(pad -> size(pad) ==
                     (n_cells + 2 * Trixi.nhalo(solver), nvariables(equations)),
              semi.cache.pad_points_threaded)
end

@testitem "MacroCellFV 1D: central stencils" setup=[Setup] tags=[:tree_part1] begin
    # Half weights of the fourth-order stencils (see `ReconstructionCentral`)
    @test collect(ReconstructionCentral(4).weights) ≈ [7 / 12, -1 / 12]
    @test collect(ReconstructionCentral(4; points = true).weights) ≈ [9 / 16, -1 / 16]
    @test collect(GradientCentral(4).weights) ≈ [-15 / 12, 1 / 12]
    @test collect(GradientCentral(4; points = true).weights) ≈ [-27 / 24, 1 / 24]
    @test collect(GradientCentral(2).weights) ≈ [-1.0]
    @test Trixi.nhalo(GradientCentral(6)) == 3
    @test Trixi.nhalo(CombinedScheme(ReconstructionWENO5Z(), GradientCentral(4))) == 3
    @test Trixi.nhalo(CombinedScheme(ReconstructionMUSCL(), GradientCentral(6))) == 3
    @test_throws ArgumentError ReconstructionCentral(3)
    @test_throws ArgumentError GradientCentral(0)

    # The value and the derivative at the face x = 0 between the subcells [-h, 0] and
    # [0, h] are exact for polynomials of degree order - 1, from the averages and from
    # the point values at the subcell centers
    h = 0.1
    for order in (2, 4, 6), points in (false, true)
        m = order ÷ 2
        coefficients = [0.3, -1.1, 0.7, 0.2, -0.4, 0.9][1:order]
        p(x) = sum(coefficients[k] * x^(k - 1) for k in 1:order)
        dp(x) = sum((k - 1) * coefficients[k] * x^(k - 2) for k in 2:order)
        # Average of `p` over the subcell [x_c - h/2, x_c + h/2] (Gauss-Legendre)
        nodes, weights = Trixi.gauss_nodes_weights(4)
        average(x_c) = sum(weights[q] / 2 * p(x_c + h / 2 * nodes[q]) for q in 1:4)
        data(j) = points ? p((j - 0.5) * h) : average((j - 0.5) * h)
        reconstruction = ReconstructionCentral(order; points)
        gradient = GradientCentral(order; points)
        # Left half from the subcell 0 ([-h, 0]), right half from the subcell 1
        _, w_left = Trixi.reconstruct_cell(reconstruction, k -> data(k))
        w_right, _ = Trixi.reconstruct_cell(reconstruction, k -> data(1 + k))
        _, d_left = Trixi.differentiate_cell(gradient, k -> data(k))
        d_right, _ = Trixi.differentiate_cell(gradient, k -> data(1 + k))
        @test Trixi.macrocell_face_values(reconstruction, w_left, w_right)[1] ≈ p(0.0)
        @test (d_left + d_right) / h ≈ dp(0.0)
        # The combined scheme gives the same values as the separate schemes
        @test all(Trixi.evaluate_cell(CombinedScheme(reconstruction, gradient),
                                      k -> data(k)) .≈
                  (Trixi.reconstruct_cell(reconstruction, k -> data(k))...,
                   Trixi.differentiate_cell(gradient, k -> data(k))...))
    end

    # Stencil factors of the diffusive time step (largest magnitude of the symbol,
    # relative to the second-order stencil)
    @test Trixi.diffusion_dt_factor(GradientCentral(2), MomentInversionIdentity()) ≈ 1
    @test Trixi.diffusion_dt_factor(GradientCentral(4), MomentInversionIdentity()) ≈
          4 / 3
    @test Trixi.diffusion_dt_factor(GradientCentral(4; points = true),
                                    MomentInversionIdentity()) ≈ 7 / 6
    @test Trixi.diffusion_dt_factor(ReconstructionWENO5Z(), MomentInversionIdentity()) ==
          0
end

@testitem "MacroCellFV 1D: elixir_diffusion_sine.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_diffusion_sine.jl"),
                        l2=[0.000256850412711328],
                        linf=[0.00036280379705877497])
    # The fused advective-diffusive right-hand side of the `ODEProblem`
    @test ode.f.f === Trixi.rhs_macrocellfv!
    @test_allocations(Trixi.rhs_macrocellfv!, semi, sol, ALLOCATION_LIMIT)
    @test subcell_average_error(sol, semi) < 1.0e-7
end

@testitem "MacroCellFV 1D: elixir_diffusion_sine.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Order of the central derivatives of the averages (orders 2, 4, 6) and of the
    # point values (fourth order with the moment inversion of order 6)
    for (order, moment_inversion) in ((2, MomentInversionIdentity()),
                                      (4, MomentInversionIdentity()),
                                      (6, MomentInversionIdentity()),
                                      (4, MomentInversionCentral()))
        errors = map((2, 3)) do level
            mod = Module()
            trixi_include(mod, joinpath(EXAMPLES_DIR, "elixir_diffusion_sine.jl");
                          initial_refinement_level = level, tspan = (0.0, 0.5),
                          reconstruction = CombinedScheme(ReconstructionCentral(order),
                                                          GradientCentral(order)),
                          moment_inversion,
                          stepsize_callback = StepsizeCallback(cfl = 0.2,
                                                               cfl_parabolic = 0.2))
            subcell_average_error(mod)
        end
        @test isapprox(log2(errors[1] / errors[2]), order; atol = 0.2)
    end
end

@testitem "MacroCellFV 1D: elixir_advection_diffusion_high_order.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion_high_order.jl"),
                        l2=[0.0002811830248396486],
                        linf=[0.00039746348841607215])
    @test_allocations(Trixi.rhs_macrocellfv!, semi, sol, ALLOCATION_LIMIT)
    # The time step: the smaller of the advective and the diffusive restrictions
    u = Trixi.wrap_array(ode.u0, semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    h = (coordinates_max - coordinates_min) / (nelements(solver, cache) * n_cells)
    dt_parabolic = Trixi.max_dt(u, 0.0, mesh, Trixi.True(), equations,
                                equations_parabolic, solver, cache)
    @test dt_parabolic ≈
          h^2 / (2 * diffusivity * solver.volume_integral.diffusion_dt_factor)
    @test Trixi.calculate_dt(ode.u0, 0.0, Returns(0.5), Returns(0.5), semi) ≈
          min(0.5 * h / advection_velocity, 0.5 * dt_parabolic)
end

@testitem "MacroCellFV 1D: elixir_advection_diffusion_high_order.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # WENO5-Z face values and fourth-order central derivatives of the point values
    # (order 5 for advection-dominated flows), MUSCL with second-order derivatives,
    # fourth-order central values and derivatives of the averages; periodic and
    # Dirichlet boundaries
    for (reconstruction, moment_inversion, order, periodicity) in ((CombinedScheme(ReconstructionWENO5Z(),
                                                                                   GradientCentral(4)),
                                                                    MomentInversionCentral(),
                                                                    5, true),
                                                                   (CombinedScheme(ReconstructionWENO5Z(),
                                                                                   GradientCentral(4)),
                                                                    MomentInversionCentral(),
                                                                    5, false),
                                                                   (CombinedScheme(ReconstructionMUSCL(minmod),
                                                                                   GradientCentral(2)),
                                                                    MomentInversionIdentity(),
                                                                    2, true),
                                                                   (CombinedScheme(ReconstructionCentral(4),
                                                                                   GradientCentral(4)),
                                                                    MomentInversionIdentity(),
                                                                    4, false))
        errors = map((4, 5)) do level
            mod = Module()
            trixi_include(mod,
                          joinpath(EXAMPLES_DIR,
                                   "elixir_advection_diffusion_high_order.jl");
                          initial_refinement_level = level, tspan = (0.0, 0.5),
                          reconstruction, moment_inversion, periodicity,
                          stepsize_callback = StepsizeCallback(cfl = 0.1,
                                                               cfl_parabolic = 0.1))
            subcell_average_error(mod)
        end
        @test log2(errors[1] / errors[2]) > order - 0.3
    end
end

@testitem "MacroCellFV 1D: elixir_advection_diffusion_high_order.jl schemes per variable" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The same combined scheme as `reconstruction` (for all variables) and in `schemes`
    # (one per variable) gives the same right-hand side
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion_high_order.jl"),
                        tspan=(0.0, 0.1))
    u_ode = copy(sol.u[end])
    for (value, gradient, inversion) in ((ReconstructionWENO5Z(), GradientCentral(4),
                                          MomentInversionCentral()),
                                         (ReconstructionMUSCL(), GradientCentral(2),
                                          MomentInversionIdentity()),
                                         (ReconstructionCentral(4), GradientCentral(4),
                                          MomentInversionIdentity()),
                                         (ReconstructionCentral(4), GradientCentral(4),
                                          MomentInversionCentral()))
        du = map(((; reconstruction = CombinedScheme(value, gradient)),
                  (; schemes = (CombinedScheme(value, gradient),)))) do kwargs
            solver_ = MacroCellFV(; n_cells, surface_flux = flux_lax_friedrichs,
                                  moment_inversion = inversion,
                                  initial_condition_nodes = 5, kwargs...)
            semi_ = SemidiscretizationHyperbolicParabolic(mesh,
                                                          (equations,
                                                           equations_parabolic),
                                                          initial_condition, solver_;
                                                          boundary_conditions = (boundary_conditions,
                                                                                 boundary_conditions))
            du_ode = similar(u_ode)
            Trixi.rhs_macrocellfv!(du_ode, u_ode, semi_, 0.0)
            du_ode
        end
        @test isapprox(du[2], du[1]; rtol = 1.0e-13,
                       atol = 1.0e-13 * maximum(abs, du[1]))
    end
end

@testitem "MacroCellFV 1D: diffusion block decomposition" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The same 48 subcells in 1 block and in blocks of 6 subcells (more than the halo
    # width 5 of WENO5-Z with the moment inversion of order 6) give the same solution up
    # to round-off errors; conservation with periodic boundaries
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion_high_order.jl"),
                        n_cells=48, initial_refinement_level=0, tspan=(0.0, 0.1))
    u_1block = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion_high_order.jl"),
                        n_cells=6, initial_refinement_level=3, tspan=(0.0, 0.1))
    @test isapprox(u_1block, sol.u[end]; atol = 1.0e-13, rtol = 0)
    @test isapprox(sum(sol.u[end]), sum(sol.u[1]); atol = 1.0e-12)

    # Also with Dirichlet boundaries (ghost subcells for the advective and the diffusive
    # fluxes)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion_high_order.jl"),
                        n_cells=48, initial_refinement_level=0, tspan=(0.0, 0.1),
                        periodicity=false)
    u_1block = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_diffusion_high_order.jl"),
                        n_cells=6, initial_refinement_level=3, tspan=(0.0, 0.1),
                        periodicity=false)
    @test isapprox(u_1block, sol.u[end]; atol = 1.0e-13, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_navierstokes_convergence_periodic.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_navierstokes_convergence_periodic.jl"),
                        l2=[0.00013504296040055055, 0.0005980964591752545,
                            0.001632650137924783],
                        linf=[0.00019076849845767185, 0.0011243753820338043,
                            0.003339316062710651],
                        tspan=(0.0, 0.1))
    @test_allocations(Trixi.rhs_macrocellfv!, semi, sol, ALLOCATION_LIMIT)
    # Four reconstruction variables (rho, v1, p, R T) for three conservative variables;
    # face derivatives only for the velocity and the temperature
    @test all(pad -> size(pad, 2) == 4, semi.cache.pad_threaded)
    @test all(buffers -> size(buffers.w_minus, 2) == 4 &&
                  size(buffers.flux, 2) == 3,
              semi.cache.line_buffers_threaded)
    @test Trixi.has_gradients(solver.volume_integral) isa Trixi.True
    @test Trixi.has_gradient(solver.volume_integral.schemes[1]) isa Trixi.False
    # The face derivatives of the variables without derivative scheme are not read
    # (also all other buffers are written before they are read)
    u_ode = copy(sol.u[end])
    du_ode = similar(u_ode)
    Trixi.rhs_macrocellfv!(du_ode, u_ode, semi, 0.1)
    du_first = copy(du_ode)
    foreach(pad -> fill!(pad, NaN), semi.cache.pad_threaded)
    foreach(pad -> fill!(pad, NaN), semi.cache.pad_points_threaded)
    foreach(buffers -> foreach(b -> fill!(b, NaN), buffers),
            semi.cache.line_buffers_threaded)
    Trixi.rhs_macrocellfv!(du_ode, u_ode, semi, 0.1)
    @test du_ode == du_first
end

@testitem "MacroCellFV 1D: elixir_navierstokes_convergence_periodic.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Fourth-order viscous fluxes (point values of the velocity and the temperature),
    # fifth-order WENO5-Z of the point values: at least fourth order for all variables
    function variable_errors(sol, semi)
        u_ode = sol.u[end]
        u_exact_ode = Trixi.compute_coefficients(sol.t[end], semi)
        GC.@preserve u_ode u_exact_ode begin
            u = Trixi.wrap_array_native(u_ode, semi)
            u_exact = Trixi.wrap_array_native(u_exact_ode, semi)
            return [sqrt(sum(abs2, selectdim(u, 1, v) .- selectdim(u_exact, 1, v)) /
                         length(selectdim(u, 1, v))) for v in 1:3]
        end
    end
    errors = map((2, 3)) do level
        mod = Module()
        trixi_include(mod,
                      joinpath(EXAMPLES_DIR, "elixir_navierstokes_convergence_periodic.jl");
                      initial_refinement_level = level, tspan = (0.0, 0.1),
                      stepsize_callback = StepsizeCallback(cfl = 0.2,
                                                           cfl_parabolic = 0.2))
        sol = Base.invokelatest(getglobal, mod, :sol)
        semi = Base.invokelatest(getglobal, mod, :semi)
        Base.invokelatest(variable_errors, sol, semi)
    end
    @test all(log2.(errors[1] ./ errors[2]) .> 3.7)
end

@testitem "MacroCellFV 1D: source terms" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The source terms are averaged over the subcells with the Gauss-Legendre
    # quadrature of `initial_condition_nodes` (2 nodes: exact for cubic polynomials)
    equations = LinearScalarAdvectionEquation1D(1.0)
    source_terms_cubic(u, x, t, equations) = SVector(x[1]^3 + 2 * u[1])
    mesh = TreeMesh(0.0, 1.0, initial_refinement_level = 2, periodicity = true)
    solver = MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                         reconstruction = ReconstructionMUSCL(),
                         initial_condition_nodes = 2)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_constant,
                                        solver; source_terms = source_terms_cubic,
                                        boundary_conditions = boundary_condition_periodic)
    u_ode = compute_coefficients(0.0, semi)
    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    # Constant state: no flux differences; the average of x^3 over [a, b] is
    # (b^4 - a^4) / (4 (b - a))
    h = 1 / 16
    exact = [((i * h)^4 - ((i - 1) * h)^4) / (4 * h) + 2 * u_ode[i] for i in 1:16]
    @test du_ode ≈ exact
    @test_allocations(Trixi.rhs_hyperbolic!, semi, (; t = [0.0], u = [u_ode]),
                      ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: diffusion setup" setup=[Setup] tags=[:tree_part1] begin
    equations = LinearScalarAdvectionEquation1D(1.0)
    equations_parabolic = LaplaceDiffusion1D(0.1, equations)
    mesh = TreeMesh(0.0, 1.0, initial_refinement_level = 2, periodicity = true)
    function semi_diffusion(solver;
                            boundary_conditions = (boundary_condition_periodic,
                                                   boundary_condition_periodic),
                            source_terms_parabolic = nothing)
        return SemidiscretizationHyperbolicParabolic(mesh,
                                                     (equations, equations_parabolic),
                                                     initial_condition_convergence_test,
                                                     solver; boundary_conditions,
                                                     source_terms_parabolic)
    end
    solver = MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                         reconstruction = CombinedScheme(ReconstructionMUSCL(),
                                                         GradientCentral(2)))
    @test semidiscretize(semi_diffusion(solver), (0.0, 1.0)).f.f ===
          Trixi.rhs_macrocellfv!
    # Diffusion needs face derivatives
    solver_values = MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                reconstruction = ReconstructionMUSCL())
    @test_throws ArgumentError semidiscretize(semi_diffusion(solver_values), (0.0, 1.0))
    @test_throws ArgumentError semidiscretize(semi_diffusion(solver;
                                                             source_terms_parabolic = (u,
                                                             x,
                                                             t,
                                                             equations) -> u),
                                              (0.0, 1.0))
    @test_throws ArgumentError semidiscretize(semi_diffusion(solver), (0.0, 1.0);
                                              jac_prototype_parabolic = zeros(2, 2))
    # The parabolic boundary conditions must be the same as the hyperbolic ones
    mesh = TreeMesh(0.0, 1.0, initial_refinement_level = 2, periodicity = false)
    bc = BoundaryConditionDirichlet(initial_condition_convergence_test)
    @test semidiscretize(semi_diffusion(solver; boundary_conditions = (bc, bc)),
                         (0.0, 1.0)).f.f === Trixi.rhs_macrocellfv!
    @test_throws ArgumentError semidiscretize(semi_diffusion(solver;
                                                             boundary_conditions = (bc,
                                                                                    boundary_condition_do_nothing)),
                                              (0.0, 1.0))
    # No separate parabolic right-hand side
    semi = semi_diffusion(solver; boundary_conditions = (bc, bc))
    u_ode = compute_coefficients(0.0, semi)
    @test_throws ArgumentError Trixi.rhs_parabolic!(similar(u_ode), u_ode, semi, 0.0)
    # `reconstruction` or `schemes`
    @test_throws ArgumentError MacroCellFV(n_cells = 4,
                                           surface_flux = flux_lax_friedrichs,
                                           reconstruction = ReconstructionMUSCL(),
                                           schemes = (ReconstructionMUSCL(),))
    @test_throws ArgumentError MacroCellFV(n_cells = 4,
                                           surface_flux = flux_lax_friedrichs,
                                           schemes = ())
    # The schemes with point values
    solver_point = MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                               schemes = (CombinedScheme(ReconstructionCentral(4),
                                                         GradientCentral(4)),
                                          ReconstructionWENO5Z()),
                               moment_inversion = MomentInversionCentral())
    @test solver_point.volume_integral.schemes[1].reconstruction.points
    @test solver_point.volume_integral.schemes[1].gradient.points
    @test solver_point.volume_integral.schemes[2] isa Trixi.ReconstructionPointValues
    @test Trixi.nhalo(solver_point) == 5
    @test Trixi.has_gradients(solver_point.volume_integral) isa Trixi.True
    @test Trixi.has_gradients(solver_values.volume_integral) isa Trixi.False
    # Pad width from the output of `center2recon`
    euler = CompressibleEulerEquations1D(1.4)
    @test Trixi.nreconvariables(cons2prim_temperature_pressure, euler) == 4
    @test Trixi.nreconvariables(cons2prim, euler) == 3
    @test cons2prim_temperature_pressure(prim2cons(SVector(2.0, 0.5, 3.0), euler),
                                         euler) ≈ SVector(2.0, 0.5, 3.0, 1.5)
    for item in (solver_point.volume_integral, solver.volume_integral,
                 solver_point.volume_integral.schemes[1], GradientCentral(4),
                 ReconstructionCentral(2))
        @test_nowarn show(IOContext(IOBuffer(), :compact => false), MIME"text/plain"(),
                          item)
    end
end
