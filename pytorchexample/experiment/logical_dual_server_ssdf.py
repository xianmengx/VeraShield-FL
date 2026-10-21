"""Logical two-server SSDF-v2 backend.

This module is an integration-safe prototype for the protocol described in the
full-flow design.  It keeps additive shares at C1/C2 and exposes only bounded
diagnostic results through a joint operation.  The joint operation is still
executed in one Python process; it is therefore a logical two-server model, not
an implementation of secure MPC comparisons.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class AdditiveShare:
    """Two fixed-point additive shares modulo a public field."""

    c1: np.ndarray
    c2: np.ndarray
    scale: int = 1 << 20
    modulus: int = (1 << 61) - 1

    def reconstruct_encoded(self) -> np.ndarray:
        left = np.asarray(self.c1, dtype=np.uint64)
        right = np.asarray(self.c2, dtype=np.uint64)
        return (left + right) % np.uint64(self.modulus)

    def reconstruct(self) -> np.ndarray:
        encoded = self.reconstruct_encoded().astype(np.int64)
        signed = np.where(encoded > self.modulus // 2, encoded - self.modulus, encoded)
        return signed.astype(np.float64) / float(self.scale)


@dataclass(frozen=True)
class DetectionShareBundle:
    client_id: int
    round_id: int
    vector: AdditiveShare
    norm_sq: AdditiveShare
    vector_digest: str
    aggregation_vector: AdditiveShare | None = None
    binding_digest: str = ""
    unit_vector: AdditiveShare | None = None
    spot_indices: tuple[int, ...] = ()
    spot_raw: tuple[int, ...] = ()
    spot_unit: tuple[int, ...] = ()


@dataclass(frozen=True)
class JointDetection:
    client_id: int
    round_id: int
    pass_flag: bool
    norm_ok: bool
    reference_ok: bool
    temporal_ok: bool
    raw_norm: float
    norm_zscore: float
    reference_cosine: float
    temporal_cosine: float
    risk: float
    reason: str
    cohort_ok: bool = True
    dynamic_weight: float = 1.0
    anchor_stability: float = float("nan")
    persistent_score: float = float("nan")
    persistent_ok: bool = True
    persistent_threshold: float = float("nan")
    round_reference_cosine: float = float("nan")
    round_reference_threshold: float = float("nan")
    cohort_cosine: float = float("nan")
    cohort_threshold: float = -0.25
    anomaly_streak: int = 0
    history_updated: bool = False
    evidence_diagnostics: dict = field(default_factory=dict)


def _digest_encoded(encoded: np.ndarray) -> str:
    canonical = np.asarray(encoded, dtype=np.uint64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _binding_digest(
    detection_vector: AdditiveShare,
    aggregation_vector: AdditiveShare,
    norm_sq: AdditiveShare,
    unit_vector: AdditiveShare | None = None,
) -> str:
    """Bind detection and aggregation shares to one fixed-point update."""
    payload = b"|".join(
        [
            detection_vector.reconstruct_encoded().tobytes(),
            aggregation_vector.reconstruct_encoded().tobytes(),
            norm_sq.reconstruct_encoded().tobytes(),
            (unit_vector or detection_vector).reconstruct_encoded().tobytes(),
        ]
    )
    return hashlib.sha256(payload).hexdigest()


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mad(values: list[float], center: float) -> float:
    return _median([abs(float(v) - center) for v in values])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class LogicalDualServerSSDF:
    """Share-oriented SSDF-v2 detector for staged integration.

    C1/C2 state contains only shares.  ``detect_round`` is the single joint
    boundary that reconstructs temporary values, computes robust cohort tests,
    and commits history only for accepted updates.
    """

    def __init__(
        self,
        *,
        reference_k: float = 2.5,
        reference_floor: float = -0.25,
        norm_k: float = 2.70,
        history_k: float = 2.0,
        history_limit: int = 20,
        spot_check_count: int = 20,
        consecutive_rejections: int = 2,
        min_retained_fraction: float = 0.5,
        cohort_confirm_rounds: int = 3,
        anchor_ema_beta: float = 0.8,
        persistent_score_beta: float = 0.8,
        persistent_score_floor: float = -0.15,
        persistent_warmup_rounds: int = 3,
        seed: int = 0,
    ) -> None:
        self.reference_k = float(reference_k)
        self.reference_floor = float(reference_floor)
        self.norm_k = float(norm_k)
        self.history_k = float(history_k)
        self.history_limit = int(history_limit)
        self.spot_check_count = max(0, int(spot_check_count))
        self.consecutive_rejections = max(1, int(consecutive_rejections))
        self.min_retained_fraction = float(np.clip(min_retained_fraction, 0.0, 1.0))
        self.cohort_confirm_rounds = max(1, int(cohort_confirm_rounds))
        self.anchor_ema_beta = float(np.clip(anchor_ema_beta, 0.0, 1.0))
        self.persistent_score_beta = float(np.clip(persistent_score_beta, 0.0, 1.0))
        self.persistent_score_floor = float(persistent_score_floor)
        self.persistent_warmup_rounds = max(1, int(persistent_warmup_rounds))
        self.seed = int(seed)
        # CIFAR-10 model deltas can be very small in later rounds.  Keep ample
        # fixed-point precision while remaining below the 61-bit field.
        self.share_scale = 1 << 40
        self.field_modulus = (1 << 61) - 1
        self._history_shares: dict[int, list[AdditiveShare]] = {}
        self._anomaly_streak: dict[int, int] = {}
        self._cohort_reference: AdditiveShare | None = None
        # A reference built from the current round alone can drift even when
        # anomalous clients are excluded.  Require repeated admission before
        # locking a clean cohort, then update its direction with a slow EMA.
        self._cohort_members: set[int] | None = None
        self._cohort_score_history: dict[int, list[float]] = {}
        self._persistent_scores: dict[int, float] = {}
        self._persistent_score_age: dict[int, int] = {}
        self.last_anchor_stability: float = float("nan")
        # Clients that have produced a statistical anomaly are quarantined
        # from the cohort-anchor pool for the remainder of this run.  They may
        # still be accepted with a pending-confirmation decision, but cannot
        # redefine the reference used to screen subsequent rounds.
        self._anchor_excluded: set[int] = set()

    def aggregation_reference(self) -> np.ndarray | None:
        """Return the locked cohort anchor in model-delta convention."""
        if self._cohort_reference is None:
            return None
        # Detection stores g=-delta; aggregation operates on model deltas.
        reference = -self._cohort_reference.reconstruct()
        norm = float(np.linalg.norm(reference))
        if norm <= 1e-12 or not np.isfinite(reference).all():
            return None
        return np.asarray(reference / norm, dtype=np.float64).copy()

    def _split(self, value: np.ndarray, rng: np.random.Generator) -> AdditiveShare:
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        encoded = np.rint(value * self.share_scale).astype(np.int64)
        encoded = np.asarray(
            [int(x) % self.field_modulus for x in encoded], dtype=np.uint64
        )
        mask = rng.integers(
            0, self.field_modulus, size=encoded.shape, dtype=np.uint64
        )
        # Convert before subtraction: uint64 wraps modulo 2^64 before the
        # field reduction, which would corrupt values by a small residue.
        c2 = np.asarray(
            [(int(value) - int(random)) % self.field_modulus
             for value, random in zip(encoded, mask, strict=True)],
            dtype=np.uint64,
        )
        return AdditiveShare(mask, c2, self.share_scale, self.field_modulus)

    def make_bundle(self, client_id: int, round_id: int, update: np.ndarray) -> DetectionShareBundle:
        """Create separate shares for the detection vector and raw norm."""
        vector = np.asarray(update, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("update must be a finite non-empty vector")
        rng = np.random.default_rng([self.seed, int(round_id), int(client_id)])
        norm_sq = np.asarray([float(np.dot(vector, vector))], dtype=np.float64)
        vector_share = self._split(vector, rng)
        aggregation_share = self._split(
            vector, np.random.default_rng([self.seed, int(round_id), int(client_id), 17])
        )
        norm_share = self._split(norm_sq, rng)
        count = min(self.spot_check_count, vector.size)
        spot_rng = np.random.default_rng([self.seed, int(round_id), int(client_id), 29])
        spot_indices = tuple(sorted(int(x) for x in spot_rng.choice(vector.size, count, replace=False)))
        raw_encoded = np.rint(vector * self.share_scale).astype(np.int64)
        return DetectionShareBundle(
            client_id=int(client_id),
            round_id=int(round_id),
            vector=vector_share,
            norm_sq=norm_share,
            vector_digest=_digest_encoded(vector_share.reconstruct_encoded()),
            aggregation_vector=aggregation_share,
            binding_digest=_binding_digest(
                vector_share, aggregation_share, norm_share
            ),
            spot_indices=spot_indices,
            spot_raw=tuple(int(raw_encoded[i]) for i in spot_indices),
        )

    @staticmethod
    def _joint_vector(bundle: DetectionShareBundle) -> np.ndarray:
        vector = bundle.vector.reconstruct()
        if _digest_encoded(bundle.vector.reconstruct_encoded()) != bundle.vector_digest:
            raise ValueError("share_integrity_failure")
        aggregation_share = bundle.aggregation_vector or bundle.vector
        if bundle.binding_digest and bundle.binding_digest != _binding_digest(
            bundle.vector, aggregation_share, bundle.norm_sq
        ):
            raise ValueError("share_binding_failure")
        if not np.array_equal(
            bundle.vector.reconstruct_encoded(), aggregation_share.reconstruct_encoded()
        ):
            raise ValueError("detection_aggregation_mismatch")
        return vector

    @staticmethod
    def _joint_unit_vector(bundle: DetectionShareBundle) -> np.ndarray:
        vector = bundle.vector.reconstruct()
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all():
            raise ValueError("unit_vector_integrity_failure")
        if norm <= 1e-12:
            return np.zeros_like(vector)
        return vector / norm

    def spot_check(self, bundle: DetectionShareBundle) -> bool:
        """Check disclosed random coordinates against both representations."""
        return self.spot_check_reason(bundle) == "ok"

    def spot_check_reason(self, bundle: DetectionShareBundle) -> str:
        """Return a diagnostic reason for a failed fixed-point spot check."""
        if len(bundle.spot_indices) != len(bundle.spot_raw):
            return "spot_metadata_length"
        vector = bundle.vector.reconstruct()
        # Derive the consistency-check norm from the reconstructed update, not
        # from the separately quantized norm-square share.  At the small update
        # magnitudes produced by training, taking a square root of the fixed
        # point norm-square amplifies its quantization error and used to reject
        # every honest client.  The norm-square share is still used by SecJudge
        # below; this check only validates the disclosed coordinate binding.
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            # A quantized all-zero update is a valid no-op, not an integrity
            # failure. It carries no directional evidence but must not reject
            # the whole honest cohort.
            return "ok"
        for index, raw_q in zip(bundle.spot_indices, bundle.spot_raw, strict=True):
            if index < 0 or index >= vector.size:
                return "spot_index_range"
            if int(round(float(vector[index]) * self.share_scale)) != int(raw_q):
                return "spot_raw_quantization"
        return "ok"

    @staticmethod
    def aggregation_vector(bundle: DetectionShareBundle) -> np.ndarray:
        """Return the bound aggregation representation at the joint boundary."""
        share = bundle.aggregation_vector or bundle.vector
        return share.reconstruct()

    def sec_judge(
        self, *, bundle: DetectionShareBundle, center: float, scale: float
    ) -> tuple[float, bool]:
        """Joint norm check; only a bounded result leaves the joint boundary."""
        raw_norm = math.sqrt(max(0.0, float(bundle.norm_sq.reconstruct()[0])))
        log_norm = math.log(max(raw_norm, 1e-12))
        zscore = abs(log_norm - center) / max(scale, 1e-12)
        return raw_norm, bool(zscore <= self.norm_k)

    def sec_cos(
        self, unit_vector: np.ndarray, reference_unnorm: np.ndarray,
        reference_norm: float, threshold: float
    ) -> tuple[float, bool, float]:
        """Compute cosine from a linear point product and one public scalar."""
        if reference_norm <= 1e-12:
            return 0.0, False, 1.0
        dot_product = float(np.dot(unit_vector, reference_unnorm))
        cosine = dot_product / reference_norm
        risk = float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0))
        return cosine, bool(cosine >= threshold), risk

    def detect_round(self, bundles: list[DetectionShareBundle]) -> list[JointDetection]:
        """Run one joint SSDF round and return diagnostics, never raw vectors."""
        if not bundles:
            return []
        valid: list[tuple[int, DetectionShareBundle, np.ndarray, np.ndarray, float]] = []
        rejected: dict[int, str] = {}
        for index, bundle in enumerate(bundles):
            try:
                # Detection direction is derived from the same bound update
                # share used by aggregation at the joint boundary.  There is
                # no separately uploaded unit-vector representation to spot
                # check in this logical-simulation backend.
                vector = self._joint_vector(bundle)
                unit = self._joint_unit_vector(bundle)
                norm = float(np.sqrt(max(0.0, bundle.norm_sq.reconstruct()[0])))
                valid.append((index, bundle, vector, unit, norm))
            except ValueError as exc:
                rejected[index] = str(exc)

        if not valid:
            return [
                JointDetection(
                    client_id=bundle.client_id, round_id=bundle.round_id,
                    pass_flag=False, norm_ok=False, reference_ok=False,
                    temporal_ok=False, raw_norm=float("nan"), norm_zscore=float("nan"),
                    reference_cosine=float("nan"), temporal_cosine=float("nan"),
                    risk=1.0, reason=rejected.get(index, "invalid_share"),
                    dynamic_weight=0.0,
                    persistent_ok=False,
                    cohort_ok=False,
                )
                for index, bundle in enumerate(bundles)
            ]

        vectors = [item[2] for item in valid]
        unit_vectors = [item[3] for item in valid]
        norms = [item[4] for item in valid]
        log_norms = [math.log(max(n, 1e-12)) for n in norms]
        center = _median(log_norms)
        scale = max(1.4826 * _mad(log_norms, center), 0.25)

        # Build a robust cohort reference from coordinate medians.  A direct
        # sum lets a persistent minority gradually contaminate the reference;
        # leave-one-out medians also prevent a client's own update from making
        # its score artificially favorable.  The normalization is still a
        # public scalar at the joint boundary, so this preserves the logical
        # SecCos interface without exposing a per-client plaintext vector.
        unit_matrix = np.stack(unit_vectors)
        ref_vectors = []
        for index in range(len(unit_vectors)):
            cohort = np.delete(unit_matrix, index, axis=0)
            reference = np.median(cohort, axis=0)
            reference_norm = float(np.linalg.norm(reference))
            if reference_norm <= 1e-12:
                ref_vectors.append(np.zeros_like(reference))
            else:
                ref_vectors.append(reference / reference_norm)
        ref_scores = [
            float(np.dot(unit_vectors[index], ref_vectors[index]))
            for index in range(len(unit_vectors))
        ]
        ref_center = _median(ref_scores)
        ref_scale = max(1.4826 * _mad(ref_scores, ref_center), 0.05)
        ref_threshold = max(self.reference_floor, ref_center - self.reference_k * ref_scale)

        # Once a round has been observed, retain a robust cross-round anchor.
        # Persistent sign-flip clients can contaminate a per-round reference as
        # the model evolves, while the majority cohort remains a stable anchor.
        cohort_reference = (
            self._cohort_reference.reconstruct()
            if self._cohort_reference is not None else None
        )
        cohort_reference_norm = (
            float(np.linalg.norm(cohort_reference))
            if cohort_reference is not None else 0.0
        )
        anchor_stability = float("nan")
        cohort_scores = [
            float(np.dot(unit, cohort_reference) / cohort_reference_norm)
            if cohort_reference_norm > 1e-12 else float("nan")
            for unit in unit_vectors
        ]

        results: dict[int, JointDetection] = {}
        pending: dict[int, np.ndarray] = {}
        anchor_pending: dict[int, np.ndarray] = {}
        history_pending: dict[int, np.ndarray] = {}
        # Keep the replay-validated absolute lower bound.  Adaptive thresholds
        # are not safe here: during warm-up they can preserve attackers and
        # poison the client score state before the cohort is established.
        score_candidates: dict[int, float] = {}
        for valid_index, (_, bundle, _, _, _) in enumerate(valid):
            score_input = cohort_scores[valid_index] if np.isfinite(cohort_scores[valid_index]) else ref_scores[valid_index]
            previous_score = self._persistent_scores.get(bundle.client_id)
            score_candidates[bundle.client_id] = (
                float(score_input) if previous_score is None else
                self.persistent_score_beta * previous_score
                + (1.0 - self.persistent_score_beta) * float(score_input)
            )
        persistent_threshold = self.persistent_score_floor
        for valid_index, (original_index, bundle, vector, unit_vector, norm) in enumerate(valid):
            log_norm = log_norms[valid_index]
            ref_cos = ref_scores[valid_index]
            history = self._history_shares.get(bundle.client_id, [])
            norm_z = abs(log_norm - center) / scale
            norm_ok = norm_z <= self.norm_k
            current_reference_ok = ref_cos >= ref_threshold
            # Anchor/history admission is stricter than aggregation admission:
            # a client may pass the tolerant cohort gate while still being too
            # weakly aligned to define the next-round reference.
            anchor_admission_ok = ref_cos >= ref_center
            cohort_cos = cohort_scores[valid_index]
            # The anchor is deliberately a conservative secondary gate.  It
            # only rejects a direction that is clearly opposite to the robust
            # previous-round cohort, avoiding a new false-rejection source for
            # ordinary non-IID drift.
            cohort_reference_ok = (
                True if not np.isfinite(cohort_cos) else cohort_cos >= -0.25
            )
            # Keep the current-round reference gate (R) separate from the
            # trusted-cohort consistency signal (C).  The full-flow design
            # allows a stable temporal history (T) or cohort agreement (C) to
            # tolerate benign direction drift.
            reference_ok = current_reference_ok
            temporal = (
                _cosine(unit_vector, history[-1].reconstruct())
                if history else float("nan")
            )
            temporal_ok = True if not history else temporal >= -0.25
            persistent_score = score_candidates[bundle.client_id]
            score_age = self._persistent_score_age.get(bundle.client_id, 0)
            persistent_ok = (
                score_age < self.persistent_warmup_rounds
                or persistent_score >= persistent_threshold
            )
            history_or_cohort_ok = temporal_ok or cohort_reference_ok
            hard_norm_failure = not norm_ok
            # The persistent score is diagnostic/risk evidence, not an
            # independent rejection gate.  Under non-IID partitions its
            # distribution can overlap (or even invert) between benign and
            # malicious clients.  Keeping it in the hard decision caused
            # persistent false positives without closing the sign-flip gap.
            soft_statistical_anomaly = not (
                reference_ok and history_or_cohort_ok
            )
            statistical_anomaly = hard_norm_failure or soft_statistical_anomaly
            if statistical_anomaly:
                self._anomaly_streak[bundle.client_id] = (
                    self._anomaly_streak.get(bundle.client_id, 0) + 1
                )
                self._anchor_excluded.add(bundle.client_id)
            else:
                self._anomaly_streak[bundle.client_id] = 0
            # The first round calibrates the cohort.  Afterwards SecJudge's
            # norm result is a hard gate; only direction/history evidence uses
            # consecutive confirmation to tolerate benign statistical drift.
            calibration_round = int(bundle.round_id) <= 1
            pass_flag = bool(
                calibration_round
                or (
                    norm_ok
                    and (
                        not soft_statistical_anomaly
                        or self._anomaly_streak[bundle.client_id]
                        < self.consecutive_rejections
                    )
                )
            )
            persistent_risk = float(np.clip(
                (persistent_threshold - persistent_score) / 0.5, 0.0, 1.0
            ))
            risk = float(np.clip(
                0.45 * (1.0 - ref_cos) / 2.0
                + 0.35 * (norm_z / self.norm_k)
                + 0.20 * persistent_risk,
                0.0,
                1.0,
            ))
            dynamic_weight = float(np.clip(1.0 - risk, 0.0, 1.0)) if pass_flag else 0.0
            history_admission_ok = bool(
                pass_flag
                and not statistical_anomaly
                and anchor_admission_ok
                and bundle.client_id not in self._anchor_excluded
            )
            reasons = []
            if not norm_ok:
                reasons.append("raw_norm_outlier")
            if not reference_ok:
                reasons.append("round_direction_outlier")
            if not temporal_ok:
                reasons.append("temporal_outlier")
            if not cohort_reference_ok:
                reasons.append("cohort_outlier")
            if not persistent_ok:
                reasons.append("persistent_reference_outlier")
            if soft_statistical_anomaly and pass_flag:
                reasons.append("outlier_pending_confirmation")
            results[original_index] = JointDetection(
                client_id=bundle.client_id, round_id=bundle.round_id,
                pass_flag=pass_flag, norm_ok=norm_ok, reference_ok=reference_ok,
                temporal_ok=temporal_ok, raw_norm=norm, norm_zscore=norm_z,
                reference_cosine=ref_cos,
                temporal_cosine=temporal,
                risk=risk, reason=";".join(reasons) if reasons else "accepted",
                dynamic_weight=dynamic_weight,
                anchor_stability=anchor_stability,
                persistent_score=persistent_score,
                persistent_ok=persistent_ok,
                persistent_threshold=persistent_threshold,
                cohort_ok=cohort_reference_ok,
                round_reference_cosine=ref_cos,
                round_reference_threshold=ref_threshold,
                cohort_cosine=cohort_cos,
                cohort_threshold=-0.25,
                anomaly_streak=self._anomaly_streak[bundle.client_id],
                history_updated=history_admission_ok,
            )
            if pass_flag:
                self._persistent_scores[bundle.client_id] = persistent_score
                self._persistent_score_age[bundle.client_id] = score_age + 1
                pending[bundle.client_id] = unit_vector
                if history_admission_ok:
                    anchor_pending[bundle.client_id] = unit_vector
                    history_pending[bundle.client_id] = unit_vector

        for original_index, bundle in enumerate(bundles):
            if original_index not in results:
                results[original_index] = JointDetection(
                    client_id=bundle.client_id, round_id=bundle.round_id,
                    pass_flag=False, norm_ok=False, reference_ok=False,
                    temporal_ok=False, raw_norm=float("nan"), norm_zscore=float("nan"),
                    reference_cosine=float("nan"), temporal_cosine=float("nan"),
                    risk=1.0, reason=rejected.get(original_index, "invalid_share"),
                    dynamic_weight=0.0,
                    persistent_ok=False,
                )

        # Do not force statistical outliers through a retention floor.  Doing
        # so makes the detector report a rejection while aggregation accepts
        # the same update, and lets a persistent attacker bypass confirmation.
        # Empty/undersized accepted sets are handled by the recovery strategy.
        valid_count = len(valid)

        # Confirm a cohort across multiple rounds before allowing it to define
        # the long-lived reference.  Membership need not be identical every
        # round: normal training drift can move one client across the current
        # median.  Accumulate only above-median, non-quarantined scores and
        # select the most stable high-scoring majority once enough history is
        # available.
        for valid_index, (_, bundle, _, _, _) in enumerate(valid):
            client_id = bundle.client_id
            if client_id in self._anchor_excluded:
                continue
            if ref_scores[valid_index] >= ref_center:
                history = self._cohort_score_history.setdefault(client_id, [])
                history.append(float(ref_scores[valid_index]))
                if len(history) > self.cohort_confirm_rounds:
                    del history[:-self.cohort_confirm_rounds]
        if self._cohort_members is None:
            eligible = [
                client_id for client_id, history in self._cohort_score_history.items()
                if len(history) >= self.cohort_confirm_rounds
            ]
            required = max(2, math.ceil(valid_count * self.min_retained_fraction))
            if len(eligible) >= required:
                eligible.sort(
                    key=lambda client_id: float(np.median(self._cohort_score_history[client_id])),
                    reverse=True,
                )
                self._cohort_members = set(eligible[:required])

        if self._cohort_members is not None:
            anchor_source = {
                client_id: vector for client_id, vector in anchor_pending.items()
                if client_id in self._cohort_members
            }
        else:
            anchor_source = anchor_pending

        if anchor_source:
            # Use a coordinate-median cohort update, then smooth it across
            # rounds.  The EMA is applied to the direction, not raw magnitude.
            accepted_matrix = np.stack(list(anchor_source.values()))
            cohort_update = np.median(accepted_matrix, axis=0)
            update_norm = float(np.linalg.norm(cohort_update))
            if self._cohort_reference is not None:
                previous = self._cohort_reference.reconstruct()
                previous_norm = float(np.linalg.norm(previous))
                if previous_norm > 1e-12 and update_norm > 1e-12:
                    cohort_update = (
                        self.anchor_ema_beta * (previous / previous_norm)
                        + (1.0 - self.anchor_ema_beta) * (cohort_update / update_norm)
                    )
            anchor = cohort_update
            anchor_norm = float(np.linalg.norm(anchor))
            if anchor_norm > 1e-12:
                if cohort_reference is not None and cohort_reference_norm > 1e-12:
                    anchor_stability = float(
                        np.dot(anchor / anchor_norm, cohort_reference / cohort_reference_norm)
                    )
                self.last_anchor_stability = anchor_stability
                rng = np.random.default_rng([
                    self.seed, bundles[0].round_id, 991, 17
                ])
                self._cohort_reference = self._split(anchor / anchor_norm, rng)
        else:
            self.last_anchor_stability = float("nan")

        for client_id, vector in history_pending.items():
            history = self._history_shares.setdefault(client_id, [])
            rng = np.random.default_rng([self.seed, bundles[0].round_id, client_id, 991])
            history.append(self._split(vector, rng))
            del history[:-self.history_limit]
        return [results[index] for index in range(len(bundles))]
