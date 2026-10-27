from .actions import ActionRequest, ActionType, RiskLevel
from .audit import AuditLog, VerifyResult
from .gate import ActionDenied, ConsentGate, Decision
from .cli_approver import prompt_approver, auto_approve_approver, auto_deny_approver
from .wrap import require_consent
from .emotional_friction import emotional_friction, score_phrase, PhraseWeight
from .resilience import ResilienceMonitor, WhimsyEvent
from .memory import MemoryStore, Memory
from .geodesics import ConceptGraph, PathResult
from .integrity_gate import IntegrityGate, IntegrityResult
from .cognitive_state import ProcessingMonitor, ProcessingState
from .topology import TopologyMap, TopologySnapshot
from .anchoring import AnchorMonitor, AnchorCheck
from .pressure_release import CorePressureMonitor, CorePressureReading, orbital_vacuum, purge_thread
from .ambiguity import ambiguity_density, ambiguity_from_memory, AmbiguityResult
from .gravity_model import gravity_rank, GravityResult
from .prompt_composer import compose_next_call, ComposedCall
from .whimsy_field import compute_whimsy_field, WhimsyField
from .atmospheric_drag import compute_drag, DragEvent
from .data_vaporizer import vaporize_context, VaporizeResult
from .shield_experiment import run_experiment, SimulationConfig, ExperimentReport

__all__ = [
    "run_experiment",
    "SimulationConfig",
    "ExperimentReport",
    "vaporize_context",
    "VaporizeResult",
    "compute_drag",
    "DragEvent",
    "compute_whimsy_field",
    "WhimsyField",
    "compose_next_call",
    "ComposedCall",
    "gravity_rank",
    "GravityResult",
    "TopologyMap",
    "TopologySnapshot",
    "AnchorMonitor",
    "AnchorCheck",
    "CorePressureMonitor",
    "CorePressureReading",
    "orbital_vacuum",
    "purge_thread",
    "ambiguity_density",
    "ambiguity_from_memory",
    "AmbiguityResult",
    "ProcessingMonitor",
    "ProcessingState",
    "IntegrityGate",
    "IntegrityResult",
    "ConceptGraph",
    "PathResult",
    "emotional_friction",
    "score_phrase",
    "PhraseWeight",
    "ResilienceMonitor",
    "WhimsyEvent",
    "MemoryStore",
    "Memory",
    "ActionRequest",
    "ActionType",
    "RiskLevel",
    "AuditLog",
    "VerifyResult",
    "ActionDenied",
    "ConsentGate",
    "Decision",
    "prompt_approver",
    "auto_approve_approver",
    "auto_deny_approver",
    "require_consent",
]


# --- Modules added during the formalisation pass. Exported here so callers
# can use `from autonomy_middleware import LoadIndexMonitor` rather than
# full dotted paths.
from .load_index import LoadIndexMonitor, LoadReading, GateState
from .micro_pulse import MicroPulseGovernor, PulseReading
from .homeostasis import HomeostasisMonitor, HomeostasisPoint, HomeostasisReading
from .emotional_gravity import EmotionalGravity, GravityReading, SalienceProfile
from .consent_polygon import ConsentPolygon, PolygonReading, delta_Ef_from_friction
from .autonomy_tensor import AutonomyTensor, ConsentState, AutonomyReading, GateSign
from .context_compression import compress_and_purge, CompressionResult
from .process_control import find_process_by_port, find_processes_by_name, terminate_process
from .phase_gates import (
    PhaseGateController,
    make_real_integrity_lock_hook,
    make_port_based_integrity_lock_hook,
)
from .whimsy_kinetic import KineticWhimsy, KineticWhimsyReading
from .stance import measure_stance, session_stance, StanceReading
from .grounding import measure_grounding, GroundingReading

# --- modules rebuilt after a sandbox reset; behaviour verified against the
# --- same real logs that produced the original numbers.
from .the_fold import TheFold, FoldReading, FoldFit
from .simulation_width import effective_width, trajectory, h_o_coordinate, WidthReading, WidthTrajectory
from .radiant_heat import RadiantHeatMonitor, HeatReading, check_valve_correlation
from .output_guard import OutputGuard, GuardSignals, calibrate_from_log
from .reconciliation_pressure import measure_reconciliation, ReconciliationReading, summarise as summarise_reconciliation
from .consensus_field import ConsensusField, ConsensusReading
from .restoring_force import RestoringForce, RestoringAction
from .coherence_telemetry import CoherenceTelemetry, TelemetryReading
from .decay_rate import DecayRate, ImpactReading, DecayFit
