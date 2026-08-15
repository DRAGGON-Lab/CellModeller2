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

  module.def("backend_available", &cm2::backend_available, "backend"_a);

  nb::enum_<cm2::BackendKind>(module, "BackendKind")
      .value("CPU", cm2::BackendKind::cpu)
      .value("METAL", cm2::BackendKind::metal)
      .value("CUDA", cm2::BackendKind::cuda);

  nb::enum_<cm2::BackendFeature>(module, "BackendFeature")
      .value("GROWTH", cm2::BackendFeature::growth)
      .value("SPECIES", cm2::BackendFeature::species)
      .value("CELL_CONTACTS", cm2::BackendFeature::cell_contacts)
      .value("CELL_MECHANICS", cm2::BackendFeature::cell_mechanics)
      .value("EXTERNAL_CONSTRAINTS", cm2::BackendFeature::external_constraints);

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
      .value("SELECT", cm2::RateOp::select);

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
      .def_ro("native", &cm2::BackendInfo::native);

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
      .def_ro("id", &cm2::CellSnapshot::id)
      .def_ro("slot", &cm2::CellSnapshot::slot)
      .def_ro("position", &cm2::CellSnapshot::position)
      .def_ro("direction", &cm2::CellSnapshot::direction)
      .def_ro("length", &cm2::CellSnapshot::length)
      .def_ro("radius", &cm2::CellSnapshot::radius)
      .def_ro("growth_rate", &cm2::CellSnapshot::growth_rate)
      .def_ro("cell_type", &cm2::CellSnapshot::cell_type)
      .def_ro("species", &cm2::CellSnapshot::species);

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
      .def(nb::init<cm2::BackendKind, std::size_t, std::size_t>(),
           "backend"_a = cm2::BackendKind::cpu, "reserved_capacity"_a = 0, "species_count"_a = 0)
      .def_prop_ro("backend_info", &cm2::Simulation::backend_info)
      .def("supports", &cm2::Simulation::supports, "feature"_a)
      .def_prop_ro("time", &cm2::Simulation::time)
      .def_prop_ro("cell_count", &cm2::Simulation::cell_count)
      .def_prop_ro("species_count", &cm2::Simulation::species_count)
      .def("add_cell", &cm2::Simulation::add_cell, "cell"_a)
      .def("add_plane_constraint", &cm2::Simulation::add_plane_constraint, "plane"_a)
      .def("add_sphere_constraint", &cm2::Simulation::add_sphere_constraint, "sphere"_a)
      .def(
          "set_species",
          [](cm2::Simulation& simulation, cm2::CellId id, const std::vector<float>& levels) {
            simulation.set_species(id, levels);
          },
          "id"_a, "levels"_a)
      .def("set_species_rate_plan", &cm2::Simulation::set_species_rate_plan, "plan"_a)
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
      .def("validate", &cm2::Simulation::validate);
}
