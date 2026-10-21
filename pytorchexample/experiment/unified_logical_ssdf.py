"""Candidate unified detector; shares/cryptographic bindings reuse v2 primitives."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
from dataclasses import replace
import math
import numpy as np

from .logical_dual_server_ssdf import (
    LogicalDualServerSSDF, JointDetection, _median, _mad, _cosine,
)
from .unified_evidence import UnifiedEvidenceState, direction_evidence, clip


class UnifiedLogicalSSDF(LogicalDualServerSSDF):
    """One evidence state, with independently admitted trusted vector state.

    Deliberately does not call the legacy constructor: no cosine-EMA rejection,
    streak counter, permanent exclusions, or locked membership is instantiated.
    """
    def __init__(self, *, seed=0, norm_k=2.70, reference_k=2.5,
                 reference_floor=-0.25, history_k=2.0, history_limit=20,
                 spot_check_count=20, cohort_confirm_rounds=3,
                 anchor_ema_beta=0.8, min_retained_fraction=0.5,
                 evidence_beta=0.8, strong_threshold=1.0,
                 persistent_threshold=0.8, anchor_threshold=0.5,
                 evidence_policy="unified_v1_candidate", reference_max_age=3, **legacy):
        if legacy:
            raise ValueError(f"obsolete/unknown unified detector parameters: {sorted(legacy)}")
        if evidence_policy not in {"unified_v1_candidate", "unified_v2_candidate"}:
            raise ValueError("unknown unified evidence policy")
        if reference_max_age < 1:
            raise ValueError("reference_max_age must be positive")
        self.evidence_policy = evidence_policy
        self.lifecycle_enabled = evidence_policy == "unified_v2_candidate"
        self.reference_max_age = int(reference_max_age)
        if norm_k <= 0 or reference_k <= 0 or history_k <= 0:
            raise ValueError("evidence scales must be positive")
        if not -1 <= reference_floor < 1 or not 0 <= anchor_threshold < persistent_threshold:
            raise ValueError("invalid reference/admission bounds")
        if not 0 <= anchor_ema_beta < 1 or not 0 < min_retained_fraction <= 1:
            raise ValueError("invalid anchor parameters")
        if history_limit < 2 or cohort_confirm_rounds < 1:
            raise ValueError("history and qualification windows must be positive")
        self.seed = int(seed)
        self.norm_k, self.reference_k = float(norm_k), float(reference_k)
        self.reference_floor, self.history_k = float(reference_floor), float(history_k)
        self.history_limit = int(history_limit)
        self.spot_check_count = max(0, int(spot_check_count))
        self.share_scale, self.field_modulus = 1 << 40, (1 << 61) - 1
        self.anchor_ema_beta = float(anchor_ema_beta)
        self.anchor_threshold = float(anchor_threshold)
        self.min_retained_fraction = float(min_retained_fraction)
        self.cohort_confirm_rounds = int(cohort_confirm_rounds)
        self.evidence = UnifiedEvidenceState(evidence_beta, strong_threshold, persistent_threshold,
                                            "norm_only" if self.lifecycle_enabled else "all_channels")
        self._history_round = {}
        self._anchor_round = None
        self._history_shares = {}
        self._temporal_observations = {}
        self._qualification = {}
        self._cohort_members = set()
        self._cohort_reference = None
        self.last_anchor_stability = float("nan")
        self._last_round = 0

    def detect_round(self, bundles):
        if not bundles:
            return []
        rounds = {b.round_id for b in bundles}
        ids = [b.client_id for b in bundles]
        if len(rounds) != 1 or len(ids) != len(set(ids)) or min(rounds) <= self._last_round:
            raise ValueError("require unique clients and one strictly increasing round")
        self._last_round = min(rounds)
        valid, invalid = [], {}
        for index, bundle in enumerate(bundles):
            try:
                vector = self._joint_vector(bundle)
                if not np.isfinite(vector).all():
                    raise ValueError("nonfinite_update")
                norm = float(np.linalg.norm(vector))
                claimed_sq = float(bundle.norm_sq.reconstruct()[0])
                if abs(claimed_sq - norm * norm) > max(2 / self.share_scale, 1e-9 * norm * norm):
                    raise ValueError("norm_binding_mismatch")
                unit = vector / norm if norm > 1e-12 else np.zeros_like(vector)
                valid.append((index, bundle, unit, norm))
            except ValueError as exc:
                invalid[index] = str(exc)

        results = {}
        for index, reason in invalid.items():
            b = bundles[index]
            results[index] = JointDetection(
                b.client_id, b.round_id, False, False, False, False,
                math.nan, math.nan, math.nan, math.nan, 1.0, reason,
                dynamic_weight=0.0,
                evidence_diagnostics={"evidence_policy": self.evidence_policy, "evidence_updated": 0},
            )
            self._qualification[b.client_id] = 0
        if not valid:
            self._cohort_members = set()
            if self.lifecycle_enabled:
                self._cohort_reference = None
                self._anchor_round = None
                self.last_anchor_stability = math.nan
            return [results[i] for i in range(len(bundles))]

        nonzero = [v for v in valid if v[3] > 1e-12]
        required = max(2, math.ceil(len(nonzero) * self.min_retained_fraction))
        cohort_size = sum(v[1].client_id in self._cohort_members for v in nonzero)
        anchor_age = self._last_round - self._anchor_round if self._anchor_round is not None else math.nan
        logs = [math.log(v[3]) for v in nonzero]
        center = _median(logs) if logs else 0.0
        scale = max(1.4826 * _mad(logs, center), 0.25) if logs else 0.25
        ref_scores = {}
        for index, bundle, unit, norm in valid:
            others = [v[2] for v in nonzero if v[0] != index]
            ref = np.median(np.stack(others), axis=0) if others else None
            ref_scores[index] = _cosine(unit, ref) if ref is not None else math.nan
        finite_refs = [x for x in ref_scores.values() if np.isfinite(x)]
        ref_center = _median(finite_refs) if finite_refs else 0.0
        ref_scale = max(1.4826 * _mad(finite_refs, ref_center), 0.05) if finite_refs else 0.05
        ref_threshold = max(self.reference_floor, ref_center - self.reference_k * ref_scale)
        old_anchor = self._cohort_reference.reconstruct() if self._cohort_reference is not None else None
        if self.lifecycle_enabled and (cohort_size < required or not np.isfinite(anchor_age)
                                       or anchor_age > self.reference_max_age):
            old_anchor = None
        cohort_scores = {v[0]: _cosine(v[2], old_anchor) for v in nonzero} if old_anchor is not None else {}
        cohort_baseline = [cohort_scores[v[0]] for v in nonzero
                           if v[1].client_id in self._cohort_members and v[0] in cohort_scores]
        # A tiny surviving cohort cannot define an adaptive baseline.
        cohort_center = _median(cohort_baseline) if len(cohort_baseline) >= 2 else 0.0
        cohort_scale = max(1.4826 * _mad(cohort_baseline, cohort_center), 0.05) if len(cohort_baseline) >= 2 else 0.05
        cohort_threshold = max(self.reference_floor, cohort_center - self.reference_k * cohort_scale) if len(cohort_baseline) >= 2 else self.reference_floor
        trusted = {}
        temporal_pending = {}
        for index, bundle, unit, norm in valid:
            cid = bundle.client_id
            zero = norm <= 1e-12
            norm_z = abs(math.log(norm) - center) / scale if not zero else 0.0
            e_norm = clip(norm_z / self.norm_k)
            ref = ref_scores[index]
            e_ref = direction_evidence(ref, ref_center, ref_threshold) if np.isfinite(ref) and not zero else 0.0
            history = self._history_shares.get(cid, [])
            history_age = self._last_round - self._history_round[cid] if cid in self._history_round else math.nan
            reset_reason = "none"
            if self.lifecycle_enabled and history and (not np.isfinite(history_age) or history_age > self.reference_max_age):
                self._history_shares.pop(cid, None)
                self._temporal_observations.pop(cid, None)
                self._history_round.pop(cid, None)
                history = []
                reset_reason = "expired"
            temporal = _cosine(unit, history[-1].reconstruct()) if history and not zero else math.nan
            observations = self._temporal_observations.get(cid, [])
            temporal_center = _median(observations) if len(observations) >= 3 else math.nan
            temporal_scale = max(1.4826 * _mad(observations, temporal_center), 0.05) if len(observations) >= 3 else math.nan
            temporal_threshold = max(-1.0, temporal_center - self.history_k * temporal_scale) if np.isfinite(temporal_center) else math.nan
            e_temporal = direction_evidence(temporal, temporal_center, temporal_threshold) if np.isfinite(temporal_threshold) and np.isfinite(temporal) else None
            cohort = cohort_scores.get(index, math.nan)
            e_cohort = direction_evidence(cohort, cohort_center, cohort_threshold) if np.isfinite(cohort) and not zero else None
            decision = self.evidence.observe(cid, e_norm, e_ref, e_temporal, e_cohort)
            qualifies = (not zero and np.isfinite(ref) and decision.accepted
                         and decision.instantaneous <= self.anchor_threshold
                         and decision.accumulated <= self.anchor_threshold)
            self._qualification[cid] = self._qualification.get(cid, 0) + 1 if qualifies else 0
            anchor_admit = qualifies and self._qualification[cid] >= self.cohort_confirm_rounds
            if anchor_admit:
                trusted[cid] = unit
            # Seed trusted history only from confident observations; do not
            # require a temporal baseline to admit its first observations.
            history_admit = qualifies
            if self.lifecycle_enabled:
                # Refresh local time state independently of strict anchor membership.
                # Current norm and leave-one-out reference must both be low risk.
                history_admit = (not zero and np.isfinite(ref) and decision.accepted
                                 and e_norm <= self.anchor_threshold
                                 and e_ref <= self.anchor_threshold)
            if history_admit:
                temporal_pending[cid] = (unit, temporal)
            diagnostic = dict(
                evidence_policy=self.evidence_policy, evidence_norm=e_norm,
                evidence_reference=e_ref, evidence_temporal=e_temporal,
                evidence_cohort=e_cohort, evidence_direction=decision.direction,
                evidence_instantaneous=decision.instantaneous,
                evidence_previous=decision.previous, evidence_accumulated=decision.accumulated,
                evidence_strong_threshold=self.evidence.strong_threshold,
                evidence_persistent_threshold=self.evidence.persistent_threshold,
                evidence_updated=1, temporal_available=int(e_temporal is not None),
                cohort_available=int(e_cohort is not None), temporal_threshold=temporal_threshold,
                anchor_admitted=int(anchor_admit), confidence=decision.confidence,
                reference_center=ref_center, reference_scale=ref_scale,
                cohort_center=cohort_center, cohort_scale=cohort_scale,
                temporal_center=temporal_center, temporal_scale=temporal_scale,
                norm_median=center, norm_mad=_mad(logs, center) if logs else 0.0,
                history_length=len(history), mu=temporal_center, sigma=temporal_scale,
                warmup=int(e_temporal is None),
                evidence_beta=self.evidence.beta, evidence_immediate_policy=self.evidence.immediate_policy,
                temporal_history_age=history_age, temporal_baseline_updated=int(history_admit),
                temporal_reset_reason=reset_reason, cohort_size=cohort_size, cohort_required=required,
                anchor_age=anchor_age, direction_hard_gate=int(not self.lifecycle_enabled),
            )
            results[index] = JointDetection(
                cid, bundle.round_id, decision.accepted, e_norm < 1,
                e_ref < 1, e_temporal is None or e_temporal < 1,
                norm, norm_z, ref, temporal, 1 - decision.confidence, decision.reason,
                cohort_ok=e_cohort is None or e_cohort < 1,
                dynamic_weight=decision.confidence, round_reference_cosine=ref,
                round_reference_threshold=ref_threshold, cohort_cosine=cohort,
                cohort_threshold=cohort_threshold, history_updated=history_admit,
                evidence_diagnostics=diagnostic,
            )

        # Membership is renewed each round; long-absent identities requalify.
        observed = set(ids)
        for cid in set(self._qualification) - observed:
            self._qualification[cid] = 0
        self._cohort_members = set(trusted)
        anchor_updated = False
        if len(trusted) >= required:
            anchor = np.median(np.stack(list(trusted.values())), axis=0)
            if np.linalg.norm(anchor) > 1e-12:
                anchor /= np.linalg.norm(anchor)
                if old_anchor is not None:
                    anchor = self.anchor_ema_beta * old_anchor + (1 - self.anchor_ema_beta) * anchor
                if np.linalg.norm(anchor) > 1e-12:
                    anchor /= np.linalg.norm(anchor)
                    self.last_anchor_stability = _cosine(anchor, old_anchor) if old_anchor is not None else math.nan
                    self._cohort_reference = self._split(anchor, np.random.default_rng([self.seed, self._last_round, 991]))
                    self._anchor_round = self._last_round
                    anchor_updated = True
        else:
            self.last_anchor_stability = math.nan
        if self.lifecycle_enabled and not anchor_updated:
            self._cohort_reference = None
            self._anchor_round = None
            self.last_anchor_stability = math.nan
        for result in results.values():
            result.evidence_diagnostics["anchor_update_reason"] = (
                "updated" if anchor_updated else "insufficient_members" if len(trusted) < required else "degenerate_direction")
        for cid, (unit, temporal) in temporal_pending.items():
            history = self._history_shares.setdefault(cid, [])
            history.append(self._split(unit, np.random.default_rng([self.seed, self._last_round, cid, 991])))
            self._history_round[cid] = self._last_round
            del history[:-self.history_limit]
            if np.isfinite(temporal):
                observations = self._temporal_observations.setdefault(cid, [])
                observations.append(temporal)
                del observations[:-self.history_limit]
        return [replace(results[i], anchor_stability=self.last_anchor_stability) for i in range(len(bundles))]
