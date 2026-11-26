using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation on [-pi, pi], periodic,
# with the exact solution of a decaying and advected sine wave

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity = 0.01
equations_parabolic = LaplaceDiffusion1D(diffusivity, equations)

# (a closure, so that `diffusivity` can be changed with `trixi_include`)
initial_condition = let nu = diffusivity
    function initial_condition_diffusive_sine(x, t,
                                              equations::LinearScalarAdvectionEquation1D)
        x_trans = x[1] - equations.advection_velocity[1] * t
        return SVector(sin(x_trans) * exp(-nu * t))
    end
end

# Macro-cell finite volume solver: WENO5-Z face values and fourth-order central face
# derivatives of the point values at the subcell centers (moment inversion of order 6),
# computed in the same loop over the subcells. The advective and the diffusive fluxes are
# evaluated in one pass with the `SemidiscretizationHyperbolicParabolic` below. The
# initial condition is set to the subcell averages (Gauss-Legendre quadrature with 5
# nodes per subcell).
n_cells = 8
reconstruction = CombinedScheme(ReconstructionWENO5Z(), GradientCentral(4))
moment_inversion = MomentInversionCentral()
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_lax_friedrichs,
                     reconstruction = reconstruction,
                     moment_inversion = moment_inversion,
                     initial_condition_nodes = 5)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)
initial_refinement_level = 3
# Periodic boundaries, or Dirichlet boundaries with the exact solution
# (`periodicity = false`), imposed by the ghost subcells of the hyperbolic boundary
# condition for both the advective and the diffusive fluxes (so the parabolic boundary
# condition must be the same)
periodicity = true
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = periodicity)

boundary_conditions = periodicity ? boundary_condition_periodic :
                      BoundaryConditionDirichlet(initial_condition)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Advective and diffusive time step restrictions
stepsize_callback = StepsizeCallback(cfl = 0.5, cfl_parabolic = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
