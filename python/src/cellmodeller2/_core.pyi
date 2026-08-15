from enum import Enum

class BackendKind(Enum):
    CPU: BackendKind
    METAL: BackendKind
    CUDA: BackendKind

class BackendFeature(Enum):
    GROWTH: BackendFeature
    SPECIES: BackendFeature
    CELL_CONTACTS: BackendFeature
    CELL_MECHANICS: BackendFeature
    EXTERNAL_CONSTRAINTS: BackendFeature

class RateOp(Enum):
    CONSTANT: RateOp
    SPECIES: RateOp
    POSITION_X: RateOp
    POSITION_Y: RateOp
    POSITION_Z: RateOp
    CELL_LENGTH: RateOp
    CELL_RADIUS: RateOp
    GROWTH_RATE: RateOp
    CELL_TYPE: RateOp
    CELL_VOLUME: RateOp
    CELL_SURFACE_AREA: RateOp
    ADD: RateOp
    SUBTRACT: RateOp
    MULTIPLY: RateOp
    DIVIDE: RateOp
    POWER: RateOp
    MINIMUM: RateOp
    MAXIMUM: RateOp
    NEGATE: RateOp
    EXPONENTIAL: RateOp
    LOGARITHM: RateOp
    LESS: RateOp
    LESS_EQUAL: RateOp
    GREATER: RateOp
    GREATER_EQUAL: RateOp
    EQUAL: RateOp
    SELECT: RateOp

class SphereRegion(Enum):
    OUTSIDE: SphereRegion
    INSIDE: SphereRegion

class ExternalConstraintKind(Enum):
    PLANE: ExternalConstraintKind
    SPHERE: ExternalConstraintKind

class RodEndpoint(Enum):
    NEGATIVE: RodEndpoint
    POSITIVE: RodEndpoint

class SolverStatus(Enum):
    CONVERGED: SolverStatus
    ITERATION_LIMIT: SolverStatus
    BREAKDOWN: SolverStatus

class SolverBreakdown(Enum):
    NONE: SolverBreakdown
    NON_FINITE_RESIDUAL: SolverBreakdown
    NON_FINITE_CURVATURE: SolverBreakdown
    NON_POSITIVE_CURVATURE: SolverBreakdown

def backend_available(backend: BackendKind) -> bool: ...

class Vec3:
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None: ...

class BackendInfo:
    @property
    def kind(self) -> BackendKind: ...
    @property
    def name(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def native(self) -> bool: ...

class CellInit:
    position: Vec3
    direction: Vec3
    length: float
    radius: float
    growth_rate: float
    cell_type: int
    species: list[float]

    def __init__(self) -> None: ...

class CellSnapshot:
    @property
    def id(self) -> int: ...
    @property
    def slot(self) -> int: ...
    @property
    def position(self) -> Vec3: ...
    @property
    def direction(self) -> Vec3: ...
    @property
    def length(self) -> float: ...
    @property
    def radius(self) -> float: ...
    @property
    def growth_rate(self) -> float: ...
    @property
    def cell_type(self) -> int: ...
    @property
    def species(self) -> list[float]: ...

class RateInstruction:
    operation: RateOp
    first: int
    second: int
    third: int
    value: float

    def __init__(self) -> None: ...

class SpeciesRatePlan:
    def __init__(
        self,
        species_count: int,
        instructions: list[RateInstruction],
        outputs: list[int],
    ) -> None: ...
    @staticmethod
    def zero(species_count: int) -> SpeciesRatePlan: ...
    @property
    def species_count(self) -> int: ...
    @property
    def instructions(self) -> list[RateInstruction]: ...
    @property
    def outputs(self) -> list[int]: ...
    def validate(self) -> None: ...

class ContactParameters:
    activation_margin: float
    parallel_sine_threshold: float
    degeneracy_epsilon: float

    def __init__(self) -> None: ...

class CellContact:
    @property
    def first_id(self) -> int: ...
    @property
    def second_id(self) -> int: ...
    @property
    def first_slot(self) -> int: ...
    @property
    def second_slot(self) -> int: ...
    @property
    def ordinal(self) -> int: ...
    @property
    def point_on_first(self) -> Vec3: ...
    @property
    def normal(self) -> Vec3: ...
    @property
    def signed_separation(self) -> float: ...
    @property
    def weight(self) -> float: ...

class ContactGraph:
    @property
    def cell_count(self) -> int: ...
    @property
    def empty(self) -> bool: ...
    @property
    def contacts(self) -> list[CellContact]: ...
    def __len__(self) -> int: ...
    def incident_contact_indices(self, slot: int) -> list[int]: ...

class PlaneConstraintInit:
    point: Vec3
    inward_normal: Vec3
    coefficient: float

    def __init__(self) -> None: ...

class SphereConstraintInit:
    center: Vec3
    radius: float
    coefficient: float
    allowed_region: SphereRegion

    def __init__(self) -> None: ...

class ConstraintContactParameters:
    activation_margin: float
    degeneracy_epsilon: float

    def __init__(self) -> None: ...

class ExternalContact:
    @property
    def cell_id(self) -> int: ...
    @property
    def cell_slot(self) -> int: ...
    @property
    def constraint_id(self) -> int: ...
    @property
    def constraint_kind(self) -> ExternalConstraintKind: ...
    @property
    def endpoint(self) -> RodEndpoint: ...
    @property
    def point_on_cell(self) -> Vec3: ...
    @property
    def normal(self) -> Vec3: ...
    @property
    def signed_separation(self) -> float: ...
    @property
    def weight(self) -> float: ...

class ExternalContactGraph:
    @property
    def cell_count(self) -> int: ...
    @property
    def empty(self) -> bool: ...
    @property
    def contacts(self) -> list[ExternalContact]: ...
    def __len__(self) -> int: ...
    def incident_contact_indices(self, slot: int) -> list[int]: ...

class CellCorrection:
    @property
    def translation(self) -> Vec3: ...
    @property
    def rotation(self) -> Vec3: ...
    @property
    def length(self) -> float: ...

class MechanicsParameters:
    mu_a: float
    gamma: float
    residual_rms_tolerance: float
    max_iterations: int

    def __init__(self) -> None: ...

class MechanicsIntegrationParameters:
    max_rotation_radians: float
    require_convergence: bool

    def __init__(self) -> None: ...

class SolverReport:
    @property
    def status(self) -> SolverStatus: ...
    @property
    def breakdown(self) -> SolverBreakdown: ...
    @property
    def iterations(self) -> int: ...
    @property
    def initial_residual_rms(self) -> float: ...
    @property
    def final_residual_rms(self) -> float: ...

class MechanicsSolveResult:
    @property
    def corrections(self) -> list[CellCorrection]: ...
    @property
    def report(self) -> SolverReport: ...

class Simulation:
    def __init__(
        self,
        backend: BackendKind = BackendKind.CPU,
        reserved_capacity: int = 0,
        species_count: int = 0,
    ) -> None: ...
    @property
    def backend_info(self) -> BackendInfo: ...
    def supports(self, feature: BackendFeature) -> bool: ...
    @property
    def time(self) -> float: ...
    @property
    def cell_count(self) -> int: ...
    @property
    def species_count(self) -> int: ...
    def add_cell(self, cell: CellInit) -> int: ...
    def add_plane_constraint(self, plane: PlaneConstraintInit) -> int: ...
    def add_sphere_constraint(self, sphere: SphereConstraintInit) -> int: ...
    def set_species(self, id: int, levels: list[float]) -> None: ...
    def set_species_rate_plan(self, plan: SpeciesRatePlan) -> None: ...
    def divide_equal(self, parent_id: int) -> tuple[int, int]: ...
    def step(self, dt: float) -> None: ...
    def find_cell_contacts(self, parameters: ContactParameters = ...) -> ContactGraph: ...
    def find_external_contacts(
        self, parameters: ConstraintContactParameters = ...
    ) -> ExternalContactGraph: ...
    def solve_cell_mechanics(
        self,
        mechanics_parameters: MechanicsParameters = ...,
        contact_parameters: ContactParameters = ...,
        constraint_parameters: ConstraintContactParameters = ...,
    ) -> MechanicsSolveResult: ...
    def relax_cell_mechanics(
        self,
        mechanics_parameters: MechanicsParameters = ...,
        contact_parameters: ContactParameters = ...,
        integration_parameters: MechanicsIntegrationParameters = ...,
        constraint_parameters: ConstraintContactParameters = ...,
    ) -> MechanicsSolveResult: ...
    def cell(self, id: int) -> CellSnapshot: ...
    def cells(self) -> list[CellSnapshot]: ...
    def lineage_parent(self, id: int) -> int | None: ...
    def validate(self) -> None: ...
