"""Pure schemas and deterministic postprocessing for physical-device evidence."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, slots=True)
class DeviceEvidence:
    artifact_id: str
    device_model: str
    os_version: str
    physical_ram: int
    battery_health: str
    storage_state: str
    initial_thermal_state: str
    memory_entitlement: str
    runtime_revision: str
    build_flags: tuple[str, ...]
    thread_count: int
    decoding: str
    context_measurements: tuple[int, ...]
    peak_physical_bytes: int
    memory_warnings: int
    jetsam_events: int
    crashes: int
    timeouts: int
    cold_load_ms: float
    ttft_ms: float
    warm_tool_p95_ms: float
    generation64_p95_ms: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    thermal_loop_latencies: tuple[float, ...]
    thermal_states: tuple[str, ...]
    network_requests: int
    battery_delta: float
    energy_per_token: float | None
    energy_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        text_fields = (
            "artifact_id", "device_model", "os_version", "battery_health",
            "storage_state", "initial_thermal_state", "memory_entitlement",
            "runtime_revision", "decoding",
        )
        if any(not getattr(self, field).strip() for field in text_fields):
            raise ValueError("completed device evidence requires every identity field")
        if self.physical_ram <= 0 or self.thread_count <= 0:
            raise ValueError("physical RAM and thread count must be positive")
        if not self.build_flags or any(not flag.strip() for flag in self.build_flags):
            raise ValueError("build flags must be complete")
        if self.context_measurements != (128, 512, 1024, 4096):
            raise ValueError("context measurements must cover 128, 512, 1024, and 4096")
        counts = (
            self.peak_physical_bytes, self.memory_warnings, self.jetsam_events,
            self.crashes, self.timeouts, self.network_requests,
        )
        if any(value < 0 for value in counts):
            raise ValueError("device counts must be non-negative")
        measurements = (
            self.cold_load_ms, self.ttft_ms, self.warm_tool_p95_ms,
            self.generation64_p95_ms, self.prefill_tokens_per_second,
            self.decode_tokens_per_second, self.battery_delta,
            *self.thermal_loop_latencies,
        )
        if any(not isinstance(value, (int, float)) for value in measurements):
            raise ValueError("device measurements must be numeric")
        if len(self.thermal_loop_latencies) != 30 or len(self.thermal_states) != 30:
            raise ValueError("thermal evidence requires exactly 30 requests")
        if self.energy_per_token is None:
            if not self.energy_unavailable_reason or not self.energy_unavailable_reason.strip():
                raise ValueError("unavailable energy must include an explicit reason")
        elif self.energy_unavailable_reason is not None:
            raise ValueError("measured energy cannot also be marked unavailable")


@dataclass(frozen=True, slots=True)
class DeviceGateThresholds:
    warm_tool_p95_ms: float
    generation64_p95_ms: float


@dataclass(frozen=True, slots=True)
class DeviceGateResult:
    memory_passed: bool
    reliability_passed: bool
    latency_passed: bool
    thermal_passed: bool

    @property
    def passed(self) -> bool:
        return all((
            self.memory_passed,
            self.reliability_passed,
            self.latency_passed,
            self.thermal_passed,
        ))


def evaluate_device_gates(
    evidence: DeviceEvidence,
    thresholds: DeviceGateThresholds,
) -> DeviceGateResult:
    """Apply release thresholds to already-collected physical evidence."""

    first_five = median(evidence.thermal_loop_latencies[:5])
    final_five = median(evidence.thermal_loop_latencies[-5:])
    serious = {"serious", "critical"}
    return DeviceGateResult(
        memory_passed=evidence.peak_physical_bytes <= evidence.physical_ram * 0.35,
        reliability_passed=(
            evidence.memory_warnings
            + evidence.jetsam_events
            + evidence.crashes
            + evidence.timeouts
        ) == 0,
        latency_passed=(
            evidence.warm_tool_p95_ms <= thresholds.warm_tool_p95_ms
            and evidence.generation64_p95_ms <= thresholds.generation64_p95_ms
        ),
        thermal_passed=(
            not serious.intersection(evidence.thermal_states)
            and final_five <= first_five * 1.20
        ),
    )


__all__ = [
    "DeviceEvidence",
    "DeviceGateResult",
    "DeviceGateThresholds",
    "evaluate_device_gates",
]
