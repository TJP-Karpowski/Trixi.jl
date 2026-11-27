using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations on [-10, 10]^2, periodic:
# the isentropic vortex advected with the velocity (1, 1), an exact solution of the
# nonlinear equations (up to the negligible perturbation at the periodic boundaries)

equations = CompressibleEulerEquations2D(1.4)

# The isentropic vortex of `elixir_euler_vortex.jl` in `tree_2d_dgsem`, advected with the
# background velocity (1, 1)
function initial_condition_isentropic_vortex_advected(x, t,
                                                      equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    # Distance to the vortex center, which moves with the velocity (1, 1) through the
    # periodic domain [-10, 10]^2
    x1 = mod(x[1] - t + 10, 20) - 10
    x2 = mod(x[2] - t + 10, 20) - 10
    amplitude = 5
    rho, v1, v2, p = 1, 1, 1, convert(RealT, 25)
    r2 = x1^2 + x2^2
    du = amplitude / (2 * convert(RealT, pi)) * exp(0.5f0 * (1 - r2))
    dtemp = -(equations.gamma - 1) / (2 * equations.gamma * p / rho) * du^2
    rho = rho * (1 + dtemp)^(1 / (equations.gamma - 1))
    v1 = v1 - du * x2
    v2 = v2 + du * x1
    p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_isentropic_vortex_advected

# Macro-cell finite volume solver with WENO5-Z reconstruction of the primitive variables
# at the subcell centers (fourth-order moment inversion of the subcell averages, then
# `cons2prim` of the point values). The numerical fluxes at the face midpoints are
# converted to face averages, so the scheme is fourth-order accurate for the nonlinear
# fluxes. With `moment_inversion = MomentInversionIdentity()` and
# `center2recon = face2flux = cons2cons`, the subcell averages are reconstructed instead,
# which is only second-order accurate for nonlinear fluxes in 2D.
n_cells = 8
moment_inversion = MomentInversionCentral(order = 4)
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_lax_friedrichs,
                     reconstruction = ReconstructionWENO5Z(),
                     moment_inversion = moment_inversion,
                     center2recon = cons2prim, face2flux = prim2cons,
                     initial_condition_nodes = 3)

coordinates_min = (-10.0, -10.0)
coordinates_max = (10.0, 10.0)
initial_refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.4)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
