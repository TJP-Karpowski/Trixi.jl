using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the heat equation u_t = ν u_xx on [-pi, pi], periodic, with the
# exact solution of a decaying sine wave (linear advection with zero velocity plus
# diffusion)

equations = LinearScalarAdvectionEquation1D(0.0)
diffusivity() = 0.1
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

function initial_condition_decaying_sine(x, t, equations::LinearScalarAdvectionEquation1D)
    return SVector(sin(x[1]) * exp(-diffusivity() * t))
end
initial_condition = initial_condition_decaying_sine

# Fourth-order central face derivatives of the subcell averages (the face values are not
# used by the diffusive flux of `LaplaceDiffusion1D`, and the advective flux vanishes)
n_cells = 8
reconstruction = CombinedScheme(ReconstructionCentral(4), GradientCentral(4))
moment_inversion = MomentInversionIdentity()
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_central,
                     reconstruction = reconstruction,
                     moment_inversion = moment_inversion,
                     initial_condition_nodes = 4)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)
initial_refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Only the diffusive time step restriction applies (zero advection velocity)
stepsize_callback = StepsizeCallback(cfl = 1.0, cfl_parabolic = 1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
