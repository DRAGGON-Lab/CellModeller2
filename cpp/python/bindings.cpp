#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cm2/simulation.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, module) {
  module.doc() = "CellModeller2 native simulation core";

  module.def("backend_device_count", &cm2::backend_device_count, "backend"_a);
  module.def("backend_available", &cm2::backend_available, "backend"_a, "device_index"_a = 0);

  nb::enum_<cm2::BackendKind>(module, "BackendKind")
      .value("CPU", cm2::BackendKind::cpu)
      .value("METAL", cm2::BackendKind::metal)
      .value("CUDA", cm2::BackendKind::cuda);

  nb::enum_<cm2::BackendFeature>(module, "BackendFeature")
      .value("GROWTH", cm2::BackendFeature::growth)
      .value("SPECIES", cm2::BackendFeature::species)
      .value("CELL_CONTACTS", cm2::BackendFeature::cell_contacts)
      .value("CELL_MECHANICS", cm2::BackendFeature::cell_mechanics)
      .value("EXTERNAL_CONSTRAINTS", cm2::BackendFeature::external_constraints)
      .value("SIGNALS", cm2::BackendFeature::signals)
      .value("COUPLED_RATES", cm2::BackendFeature::coupled_rates);

  nb::enum_<cm2::GridBoundaryKind>(module, "GridBoundaryKind")
      .value("NO_FLUX", cm2::GridBoundaryKind::no_flux)
      .value("PERIODIC", cm2::GridBoundaryKind::periodic)
      .value("FIXED", cm2::GridBoundaryKind::fixed);

  nb::enum_<cm2::RateOp>(module, "RateOp")
      .value("CONSTANT", cm2::RateOp::constant)
      .value("SPECIES", cm2::RateOp::species)
      .value("POSITION_X", cm2::RateOp::position_x)
      .value("POSITION_Y", cm2::RateOp::position_y)
      .value("POSITION_Z", cm2::RateOp::position_z)
      .value("CELL_LENGTH", cm2::RateOp::cell_length)
      .value("CELL_RADIUS", cm2::RateOp::cell_radius)
      .value("GROWTH_RATE", cm2::RateOp::growth_rate)
      .value("CELL_TYPE", cm2::RateOp::cell_type)
      .value("CELL_VOLUME", cm2::RateOp::cell_volume)
      .value("CELL_SURFACE_AREA", cm2::RateOp::cell_surface_area)
      .value("ADD", cm2::RateOp::add)
      .value("SUBTRACT", cm2::RateOp::subtract)
      .value("MULTIPLY", cm2::RateOp::multiply)
      .value("DIVIDE", cm2::RateOp::divide)
      .value("POWER", cm2::RateOp::power)
      .value("MINIMUM", cm2::RateOp::minimum)
      .value("MAXIMUM", cm2::RateOp::maximum)
      .value("NEGATE", cm2::RateOp::negate)
      .value("EXPONENTIAL", cm2::RateOp::exponential)
      .value("LOGARITHM", cm2::RateOp::logarithm)
      .value("LESS", cm2::RateOp::less)
      .value("LESS_EQUAL", cm2::RateOp::less_equal)
      .value("GREATER", cm2::RateOp::greater)
      .value("GREATER_EQUAL", cm2::RateOp::greater_equal)
      .value("EQUAL", cm2::RateOp::equal)
      .value("SELECT", cm2::RateOp::select)
      .value("SIGNAL", cm2::RateOp::signal);

  nb::enum_<cm2::SphereRegion>(module, "SphereRegion")
      .value("OUTSIDE", cm2::SphereRegion::outside)
      .value("INSIDE", cm2::SphereRegion::inside);

  nb::enum_<cm2::ExternalConstraintKind>(module, "ExternalConstraintKind")
      .value("PLANE", cm2::ExternalConstraintKind::plane)
      .value("SPHERE", cm2::ExternalConstraintKind::sphere);

  nb::enum_<cm2::RodEndpoint>(module, "RodEndpoint")
      .value("NEGATIVE", cm2::RodEndpoint::negative)
      .value("POSITIVE", cm2::RodEndpoint::positive);

  nb::enum_<cm2::SolverStatus>(module, "SolverStatus")
      .value("CONVERGED", cm2::SolverStatus::converged)
      .value("ITERATION_LIMIT", cm2::SolverStatus::iteration_limit)
      .value("BREAKDOWN", cm2::SolverStatus::breakdown);

  nb::enum_<cm2::SolverBreakdown>(module, "SolverBreakdown")
      .value("NONE", cm2::SolverBreakdown::none)
      .value("NON_FINITE_RESIDUAL", cm2::SolverBreakdown::non_finite_residual)
      .value("NON_FINITE_CURVATURE", cm2::SolverBreakdown::non_finite_curvature)
      .value("NON_POSITIVE_CURVATURE", cm2::SolverBreakdown::non_positive_curvature);

  nb::class_<cm2::Vec3>(module, "Vec3")
      .def(nb::init<float, float, float>(), "x"_a = 0.0F, "y"_a = 0.0F, "z"_a = 0.0F)
      .def_rw("x", &cm2::Vec3::x)
      .def_rw("y", &cm2::Vec3::y)
      .def_rw("z", &cm2::Vec3::z);

  nb::class_<cm2::BackendInfo>(module, "BackendInfo")
      .def_ro("kind", &cm2::BackendInfo::kind)
      .def_ro("name", &cm2::BackendInfo::name)
      .def_ro("device", &cm2::BackendInfo::device)
      .def_ro("device_index", &cm2::BackendInfo::device_index)
      .def_ro("native", &cm2::BackendInfo::native);

  nb::class_<cm2::GridBoundary>(module, "GridBoundary")
      .def(nb::init<>())
      .def_rw("kind", &cm2::GridBoundary::kind)
      .def_rw("values", &cm2::GridBoundary::values)
      .def("validate", &cm2::GridBoundary::validate, "signal_count"_a);

  nb::class_<cm2::GridShape>(module, "GridShape")
      .def(nb::init<>())
      .def_rw("x", &cm2::GridShape::x)
      .def_rw("y", &cm2::GridShape::y)
      .def_rw("z", &cm2::GridShape::z);

  nb::class_<cm2::SignalGridSpec>(module, "SignalGridSpec")
      .def(nb::init<>())
      .def_rw("signal_count", &cm2::SignalGridSpec::signal_count)
      .def_rw("shape", &cm2::SignalGridSpec::shape)
      .def_rw("origin", &cm2::SignalGridSpec::origin)
      .def_rw("spacing", &cm2::SignalGridSpec::spacing)
      .def_rw("diffusion", &cm2::SignalGridSpec::diffusion)
      .def_rw("advection", &cm2::SignalGridSpec::advection)
      .def_rw("x_lower", &cm2::SignalGridSpec::x_lower)
      .def_rw("x_upper", &cm2::SignalGridSpec::x_upper)
      .def_rw("y_lower", &cm2::SignalGridSpec::y_lower)
      .def_rw("y_upper", &cm2::SignalGridSpec::y_upper)
      .def_rw("z_lower", &cm2::SignalGridSpec::z_lower)
      .def_rw("z_upper", &cm2::SignalGridSpec::z_upper)
      .def_prop_ro("site_count", &cm2::SignalGridSpec::site_count)
      .def_prop_ro("level_count", &cm2::SignalGridSpec::level_count)
      .def_prop_ro("voxel_volume", &cm2::SignalGridSpec::voxel_volume)
      .def("validate", &cm2::SignalGridSpec::validate);

  nb::class_<cm2::SignalGridCheckpoint>(module, "_SignalGridCheckpoint")
      .def(nb::init<>())
      .def_rw("spec", &cm2::SignalGridCheckpoint::spec)
      .def_rw("levels", &cm2::SignalGridCheckpoint::levels)
      .def("validate", &cm2::SignalGridCheckpoint::validate);

  nb::class_<cm2::CellInit>(module, "CellInit")
      .def(nb::init<>())
      .def_rw("position", &cm2::CellInit::position)
      .def_rw("direction", &cm2::CellInit::direction)
      .def_rw("length", &cm2::CellInit::length)
      .def_rw("radius", &cm2::CellInit::radius)
      .def_rw("growth_rate", &cm2::CellInit::growth_rate)
      .def_rw("cell_type", &cm2::CellInit::cell_type)
      .def_rw("species", &cm2::CellInit::species);

  nb::class_<cm2::CellSnapshot>(module, "CellSnapshot")
      .def(nb::init<>())
      .def_rw("id", &cm2::CellSnapshot::id)
      .def_rw("slot", &cm2::CellSnapshot::slot)
      .def_rw("position", &cm2::CellSnapshot::position)
      .def_rw("direction", &cm2::CellSnapshot::direction)
      .def_rw("length", &cm2::CellSnapshot::length)
      .def_rw("radius", &cm2::CellSnapshot::radius)
      .def_rw("growth_rate", &cm2::CellSnapshot::growth_rate)
      .def_rw("cell_type", &cm2::CellSnapshot::cell_type)
      .def_rw("species", &cm2::CellSnapshot::species);

  nb::class_<cm2::LineageEntry>(module, "_LineageEntry")
      .def(nb::init<>())
      .def_rw("child", &cm2::LineageEntry::child)
      .def_rw("parent", &cm2::LineageEntry::parent);

  nb::class_<cm2::WorldStateCheckpoint>(module, "_WorldStateCheckpoint")
      .def(nb::init<>())
      .def_rw("species_count", &cm2::WorldStateCheckpoint::species_count)
      .def_rw("next_id", &cm2::WorldStateCheckpoint::next_id)
      .def_rw("cells", &cm2::WorldStateCheckpoint::cells)
      .def_rw("lineage", &cm2::WorldStateCheckpoint::lineage)
      .def("validate", &cm2::WorldStateCheckpoint::validate);

  nb::class_<cm2::RateInstruction>(module, "RateInstruction")
      .def(nb::init<>())
      .def_rw("operation", &cm2::RateInstruction::operation)
      .def_rw("first", &cm2::RateInstruction::first)
      .def_rw("second", &cm2::RateInstruction::second)
      .def_rw("third", &cm2::RateInstruction::third)
      .def_rw("value", &cm2::RateInstruction::value);

  nb::class_<cm2::SpeciesRatePlan>(module, "SpeciesRatePlan")
      .def(nb::init<std::size_t, std::vector<cm2::RateInstruction>, std::vector<std::uint32_t>>(),
           "species_count"_a, "instructions"_a, "outputs"_a)
      .def_static("zero", &cm2::SpeciesRatePlan::zero, "species_count"_a)
      .def_prop_ro("species_count", &cm2::SpeciesRatePlan::species_count)
      .def_prop_ro("instructions",
                   [](const cm2::SpeciesRatePlan& plan) {
                     return std::vector<cm2::RateInstruction>(plan.instructions().begin(),
                                                              plan.instructions().end());
                   })
      .def_prop_ro("outputs",
                   [](const cm2::SpeciesRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.outputs().begin(),
                                                       plan.outputs().end());
                   })
      .def("validate", &cm2::SpeciesRatePlan::validate);

  nb::class_<cm2::CoupledRatePlan>(module, "CoupledRatePlan")
      .def(nb::init<std::size_t, std::size_t, std::vector<cm2::RateInstruction>,
                    std::vector<std::uint32_t>, std::vector<std::uint32_t>>(),
           "species_count"_a, "signal_count"_a, "instructions"_a, "species_outputs"_a,
           "signal_outputs"_a)
      .def_prop_ro("species_count", &cm2::CoupledRatePlan::species_count)
      .def_prop_ro("signal_count", &cm2::CoupledRatePlan::signal_count)
      .def_prop_ro("instructions",
                   [](const cm2::CoupledRatePlan& plan) {
                     return std::vector<cm2::RateInstruction>(plan.instructions().begin(),
                                                              plan.instructions().end());
                   })
      .def_prop_ro("species_outputs",
                   [](const cm2::CoupledRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.species_outputs().begin(),
                                                       plan.species_outputs().end());
                   })
      .def_prop_ro("signal_outputs",
                   [](const cm2::CoupledRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.signal_outputs().begin(),
                                                       plan.signal_outputs().end());
                   })
      .def("validate", &cm2::CoupledRatePlan::validate);

  nb::class_<cm2::ContactParameters>(module, "ContactParameters")
      .def(nb::init<>())
      .def_rw("activation_margin", &cm2::ContactParameters::activation_margin)
      .def_rw("parallel_sine_threshold", &cm2::ContactParameters::parallel_sine_threshold)
      .def_rw("degeneracy_epsilon", &cm2::ContactParameters::degeneracy_epsilon);

  nb::class_<cm2::CellContact>(module, "CellContact")
      .def_ro("first_id", &cm2::CellContact::first_id)
      .def_ro("second_id", &cm2::CellContact::second_id)
      .def_ro("first_slot", &cm2::CellContact::first_slot)
      .def_ro("second_slot", &cm2::CellContact::second_slot)
      .def_ro("ordinal", &cm2::CellContact::ordinal)
      .def_ro("point_on_first", &cm2::CellContact::point_on_first)
      .def_ro("normal", &cm2::CellContact::normal)
      .def_ro("signed_separation", &cm2::CellContact::signed_separation)
      .def_ro("weight", &cm2::CellContact::weight);

  nb::class_<cm2::ContactGraph>(module, "ContactGraph")
      .def_prop_ro("cell_count", &cm2::ContactGraph::cell_count)
      .def_prop_ro("empty", &cm2::ContactGraph::empty)
      .def_prop_ro("contacts",
                   [](const cm2::ContactGraph& graph) {
                     return std::vector<cm2::CellContact>(graph.contacts().begin(),
                                                          graph.contacts().end());
                   })
      .def("__len__", &cm2::ContactGraph::size)
      .def(
          "incident_contact_indices",
          [](const cm2::ContactGraph& graph, cm2::Slot slot) {
            const auto indices = graph.incident_contact_indices(slot);
            return std::vector<std::size_t>(indices.begin(), indices.end());
          },
          "slot"_a)
      .def(
          "neighbor_ids",
          [](const cm2::ContactGraph& graph, cm2::Slot slot) {
            const auto ids = graph.neighbor_ids(slot);
            return std::vector<cm2::CellId>(ids.begin(), ids.end());
          },
          "slot"_a);

  nb::class_<cm2::PlaneConstraintInit>(module, "PlaneConstraintInit")
      .def(nb::init<>())
      .def_rw("point", &cm2::PlaneConstraintInit::point)
      .def_rw("inward_normal", &cm2::PlaneConstraintInit::inward_normal)
      .def_rw("coefficient", &cm2::PlaneConstraintInit::coefficient);

  nb::class_<cm2::SphereConstraintInit>(module, "SphereConstraintInit")
      .def(nb::init<>())
      .def_rw("center", &cm2::SphereConstraintInit::center)
      .def_rw("radius", &cm2::SphereConstraintInit::radius)
      .def_rw("coefficient", &cm2::SphereConstraintInit::coefficient)
      .def_rw("allowed_region", &cm2::SphereConstraintInit::allowed_region);

  nb::class_<cm2::PlaneConstraint>(module, "_PlaneConstraint")
      .def(nb::init<>())
      .def_rw("id", &cm2::PlaneConstraint::id)
      .def_rw("point", &cm2::PlaneConstraint::point)
      .def_rw("inward_normal", &cm2::PlaneConstraint::inward_normal)
      .def_rw("coefficient", &cm2::PlaneConstraint::coefficient);

  nb::class_<cm2::SphereConstraint>(module, "_SphereConstraint")
      .def(nb::init<>())
      .def_rw("id", &cm2::SphereConstraint::id)
      .def_rw("center", &cm2::SphereConstraint::center)
      .def_rw("radius", &cm2::SphereConstraint::radius)
      .def_rw("coefficient", &cm2::SphereConstraint::coefficient)
      .def_rw("allowed_region", &cm2::SphereConstraint::allowed_region);

  nb::class_<cm2::ConstraintSetCheckpoint>(module, "_ConstraintSetCheckpoint")
      .def(nb::init<>())
      .def_rw("next_id", &cm2::ConstraintSetCheckpoint::next_id)
      .def_rw("planes", &cm2::ConstraintSetCheckpoint::planes)
      .def_rw("spheres", &cm2::ConstraintSetCheckpoint::spheres)
      .def("validate", &cm2::ConstraintSetCheckpoint::validate);

  nb::class_<cm2::SimulationCheckpoint>(module, "_SimulationCheckpoint")
      .def(nb::init<>())
      .def_rw("schema_version", &cm2::SimulationCheckpoint::schema_version)
      .def_rw("time", &cm2::SimulationCheckpoint::time)
      .def_rw("world", &cm2::SimulationCheckpoint::world)
      .def_rw("constraints", &cm2::SimulationCheckpoint::constraints)
      .def_rw("species_rate_plan", &cm2::SimulationCheckpoint::species_rate_plan)
      .def_rw("signal_grid", &cm2::SimulationCheckpoint::signal_grid)
      .def_rw("coupled_rate_plan", &cm2::SimulationCheckpoint::coupled_rate_plan)
      .def("validate", &cm2::SimulationCheckpoint::validate);

  nb::class_<cm2::ConstraintContactParameters>(module, "ConstraintContactParameters")
      .def(nb::init<>())
      .def_rw("activation_margin", &cm2::ConstraintContactParameters::activation_margin)
      .def_rw("degeneracy_epsilon", &cm2::ConstraintContactParameters::degeneracy_epsilon);

  nb::class_<cm2::ExternalContact>(module, "ExternalContact")
      .def_ro("cell_id", &cm2::ExternalContact::cell_id)
      .def_ro("cell_slot", &cm2::ExternalContact::cell_slot)
      .def_ro("constraint_id", &cm2::ExternalContact::constraint_id)
      .def_ro("constraint_kind", &cm2::ExternalContact::constraint_kind)
      .def_ro("endpoint", &cm2::ExternalContact::endpoint)
      .def_ro("point_on_cell", &cm2::ExternalContact::point_on_cell)
      .def_ro("normal", &cm2::ExternalContact::normal)
      .def_ro("signed_separation", &cm2::ExternalContact::signed_separation)
      .def_ro("weight", &cm2::ExternalContact::weight);

  nb::class_<cm2::ExternalContactGraph>(module, "ExternalContactGraph")
      .def_prop_ro("cell_count", &cm2::ExternalContactGraph::cell_count)
      .def_prop_ro("empty", &cm2::ExternalContactGraph::empty)
      .def_prop_ro("contacts",
                   [](const cm2::ExternalContactGraph& graph) {
                     return std::vector<cm2::ExternalContact>(graph.contacts().begin(),
                                                              graph.contacts().end());
                   })
      .def("__len__", &cm2::ExternalContactGraph::size)
      .def(
          "incident_contact_indices",
          [](const cm2::ExternalContactGraph& graph, cm2::Slot slot) {
            const auto indices = graph.incident_contact_indices(slot);
            return std::vector<std::size_t>(indices.begin(), indices.end());
          },
          "slot"_a);

  nb::class_<cm2::CellCorrection>(module, "CellCorrection")
      .def_ro("translation", &cm2::CellCorrection::translation)
      .def_ro("rotation", &cm2::CellCorrection::rotation)
      .def_ro("length", &cm2::CellCorrection::length);

  nb::class_<cm2::MechanicsParameters>(module, "MechanicsParameters")
      .def(nb::init<>())
      .def_rw("mu_a", &cm2::MechanicsParameters::mu_a)
      .def_rw("gamma", &cm2::MechanicsParameters::gamma)
      .def_rw("residual_rms_tolerance", &cm2::MechanicsParameters::residual_rms_tolerance)
      .def_rw("max_iterations", &cm2::MechanicsParameters::max_iterations);

  nb::class_<cm2::MechanicsIntegrationParameters>(module, "MechanicsIntegrationParameters")
      .def(nb::init<>())
      .def_rw("max_rotation_radians", &cm2::MechanicsIntegrationParameters::max_rotation_radians)
      .def_rw("require_convergence", &cm2::MechanicsIntegrationParameters::require_convergence);

  nb::class_<cm2::SolverReport>(module, "SolverReport")
      .def_ro("status", &cm2::SolverReport::status)
      .def_ro("breakdown", &cm2::SolverReport::breakdown)
      .def_ro("iterations", &cm2::SolverReport::iterations)
      .def_ro("initial_residual_rms", &cm2::SolverReport::initial_residual_rms)
      .def_ro("final_residual_rms", &cm2::SolverReport::final_residual_rms);

  nb::class_<cm2::MechanicsSolveResult>(module, "MechanicsSolveResult")
      .def_ro("corrections", &cm2::MechanicsSolveResult::corrections)
      .def_ro("report", &cm2::MechanicsSolveResult::report);

  nb::class_<cm2::Simulation>(module, "Simulation")
      .def(nb::init<cm2::BackendKind, std::size_t, std::size_t, std::uint32_t>(),
           "backend"_a = cm2::BackendKind::cpu, "reserved_capacity"_a = 0, "species_count"_a = 0,
           "device_index"_a = 0)
      .def(nb::init<cm2::BackendKind, const cm2::SimulationCheckpoint&, std::uint32_t>(),
           "backend"_a, "checkpoint"_a, "device_index"_a = 0)
      .def_prop_ro("backend_info", &cm2::Simulation::backend_info)
      .def("supports", &cm2::Simulation::supports, "feature"_a)
      .def_prop_ro("time", &cm2::Simulation::time)
      .def_prop_ro("cell_count", &cm2::Simulation::cell_count)
      .def_prop_ro("species_count", &cm2::Simulation::species_count)
      .def_prop_ro("signal_count", &cm2::Simulation::signal_count)
      .def_prop_ro("has_signal_grid", &cm2::Simulation::has_signal_grid)
      .def_prop_ro("has_coupled_rate_plan", &cm2::Simulation::has_coupled_rate_plan)
      .def("add_cell", &cm2::Simulation::add_cell, "cell"_a)
      .def("add_plane_constraint", &cm2::Simulation::add_plane_constraint, "plane"_a)
      .def("add_sphere_constraint", &cm2::Simulation::add_sphere_constraint, "sphere"_a)
      .def("set_cell_attributes", &cm2::Simulation::set_cell_attributes, "id"_a,
           "growth_rate"_a, "cell_type"_a)
      .def(
          "set_species",
          [](cm2::Simulation& simulation, cm2::CellId id, const std::vector<float>& levels) {
            simulation.set_species(id, levels);
          },
          "id"_a, "levels"_a)
      .def("set_species_rate_plan", &cm2::Simulation::set_species_rate_plan, "plan"_a)
      .def("set_coupled_rate_plan", &cm2::Simulation::set_coupled_rate_plan, "plan"_a)
      .def("clear_coupled_rate_plan", &cm2::Simulation::clear_coupled_rate_plan)
      .def("configure_signal_grid", &cm2::Simulation::configure_signal_grid, "spec"_a,
           "levels"_a = std::vector<float>{})
      .def(
          "set_signal_levels",
          [](cm2::Simulation& simulation, const std::vector<float>& levels) {
            simulation.set_signal_levels(levels);
          },
          "levels"_a)
      .def("divide_equal", &cm2::Simulation::divide_equal, "parent_id"_a)
      .def("step", &cm2::Simulation::step, "dt"_a)
      .def("find_cell_contacts", &cm2::Simulation::find_cell_contacts,
           "parameters"_a = cm2::ContactParameters{})
      .def("find_external_contacts", &cm2::Simulation::find_external_contacts,
           "parameters"_a = cm2::ConstraintContactParameters{})
      .def("solve_cell_mechanics", &cm2::Simulation::solve_cell_mechanics,
           "mechanics_parameters"_a = cm2::MechanicsParameters{},
           "contact_parameters"_a = cm2::ContactParameters{},
           "constraint_parameters"_a = cm2::ConstraintContactParameters{})
      .def("relax_cell_mechanics", &cm2::Simulation::relax_cell_mechanics,
           "mechanics_parameters"_a = cm2::MechanicsParameters{},
           "contact_parameters"_a = cm2::ContactParameters{},
           "integration_parameters"_a = cm2::MechanicsIntegrationParameters{},
           "constraint_parameters"_a = cm2::ConstraintContactParameters{})
      .def("cell", &cm2::Simulation::cell, "id"_a)
      .def("cells", &cm2::Simulation::cells)
      .def("lineage_parent", &cm2::Simulation::lineage_parent, "id"_a)
      .def_prop_ro("signal_levels", &cm2::Simulation::signal_levels)
      .def("sample_signals", &cm2::Simulation::sample_signals, "position"_a)
      .def("_checkpoint", &cm2::Simulation::checkpoint)
      .def("validate", &cm2::Simulation::validate);
}
