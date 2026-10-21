"""SSDF-v2 plaintext reference detector.

The original detector only compared a client with its own previous update.
SSDF-v2 keeps that temporal signal, but adds a robust round reference and a
cohort-relative raw-norm gate. The extra signals close the persistent
sign-flip and scaling blind spots while retaining the sign/log representation
required by the target secret-sharing protocol.

This module is still a plaintext experimental backend. It does not implement
the target SecJudge/SecCos secure-computation protocol.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

import math
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Preprocessed:
    g_hat: dict[str, torch.Tensor] | None
    sign: dict[str, torch.Tensor] | None
    log: dict[str, torch.Tensor] | None
    flat_hat: torch.Tensor | None
    norm: float
    degenerate: bool


@dataclass
class DetectionResult:
    client_id: int
    round: int
    is_malicious_gt: int
    attack_active: int
    is_malicious_pred: int
    secjudge_ok: bool
    norm_sum: float
    cosine: float
    threshold: float
    risk: float
    pass_flag: bool
    warmup: bool
    history_length: int = 0
    mu: float = float("nan")
    sigma: float = float("nan")
    preprocess_time: float = 0.0
    sec_judge_time: float = 0.0
    sec_cos_time: float = 0.0
    threshold_time: float = 0.0
    decision_time: float = 0.0
    detection_time: float = 0.0
    total_ssdf_time: float = 0.0
    logical_as1_time: float = 0.0
    logical_as2_time: float = 0.0
    raw_norm: float = float("nan")
    norm_median: float = float("nan")
    norm_mad: float = float("nan")
    norm_zscore: float = float("nan")
    reference_cosine: float = float("nan")
    reference_threshold: float = float("nan")
    sign_agreement: float = float("nan")
    sign_threshold: float = float("nan")
    temporal_ok: bool = True
    reference_ok: bool = True
    history_updated: bool = False
    decision_reason: str = ""
    detector_mode: str = "v2_full"


def _ema(values: list[float], beta: float) -> float:
    if not values:
        return float("nan")
    mu = values[0]
    for value in values[1:]:
        mu = beta * mu + (1.0 - beta) * value
    return float(mu)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _mad(values: list[float], center: float) -> float:
    return _median([abs(float(value) - center) for value in values])


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class SSDFDetector:
    """Round-aware SSDF-v2 detector with trusted per-client history."""

    def __init__(
        self,
        ema_beta: float = 0.9,
        k: float = 2.0,
        norm_tol: float = 1e-3,
        warmup_rounds: int = 2,
        eps: float = 1e-8,
        norm_floor: float = 1e-12,
        reference_k: float = 2.5,
        reference_floor: float = -0.25,
        reference_mad_floor: float = 0.05,
        norm_k: float = 2.70,
        norm_log_mad_floor: float = 0.25,
        history_limit: int = 20,
        detector_mode: str = "v2_full",
    ) -> None:
        self.beta = float(ema_beta)
        self.k = float(k)
        self.norm_tol = float(norm_tol)
        self.warmup_rounds = int(warmup_rounds)
        self.eps = float(eps)
        self.norm_floor = float(norm_floor)
        self.reference_k = float(reference_k)
        self.reference_floor = float(reference_floor)
        self.reference_mad_floor = float(reference_mad_floor)
        self.norm_k = float(norm_k)
        self.norm_log_mad_floor = float(norm_log_mad_floor)
        self.history_limit = int(history_limit)
        valid_modes = {
            "v1_temporal",
            "v2_reference_only",
            "v2_norm_only",
            "v2_full",
        }
        if detector_mode not in valid_modes:
            raise ValueError(
                f"unknown SSDF detector mode {detector_mode!r}; "
                f"expected one of {sorted(valid_modes)}"
            )
        self.detector_mode = detector_mode
        # Only accepted updates are committed to this history.
        self.history: dict[int, dict[str, Any]] = {}

    def _hist(self, client_id: int) -> dict[str, Any]:
        return self.history.setdefault(client_id, {"cos": [], "prev_g": None})

    def preprocess(self, delta: dict[str, torch.Tensor]) -> Preprocessed:
        g = {key: -value for key, value in delta.items()}
        flat = torch.cat([g[key].detach().reshape(-1).to(torch.float64) for key in g])
        norm = float(torch.linalg.vector_norm(flat))
        if not math.isfinite(norm) or norm < self.norm_floor:
            return Preprocessed(None, None, None, None, norm, True)
        flat_hat = flat / norm
        g_hat: dict[str, torch.Tensor] = {}
        sign: dict[str, torch.Tensor] = {}
        log: dict[str, torch.Tensor] = {}
        offset = 0
        for key, value in g.items():
            size = value.numel()
            normalized = flat_hat[offset : offset + size].reshape(value.shape)
            g_hat[key] = normalized
            sign[key] = torch.sign(normalized)
            log[key] = torch.log(torch.abs(normalized) + self.eps)
            offset += size
        return Preprocessed(g_hat, sign, log, flat_hat, norm, False)

    @staticmethod
    def _cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b))

    def detect_round(self, updates: list[dict[str, Any]]) -> list[DetectionResult]:
        """Detect a complete cohort before committing trusted history."""
        if not updates:
            return []

        prepared: list[tuple[dict[str, Any], Preprocessed, float]] = []
        for item in updates:
            started = time.perf_counter()
            pp = self.preprocess(item["delta"])
            prepared.append((item, pp, time.perf_counter() - started))

        valid = [pp for _, pp, _ in prepared if not pp.degenerate and pp.flat_hat is not None]
        if valid:
            reference = torch.zeros_like(valid[0].flat_hat)
            for pp in valid:
                reference.add_(pp.flat_hat)
            reference_norm = float(torch.linalg.vector_norm(reference))
            if reference_norm < self.norm_floor:
                reference = torch.stack([pp.flat_hat for pp in valid]).median(dim=0).values
                reference_norm = float(torch.linalg.vector_norm(reference))
            reference = reference / max(reference_norm, self.norm_floor)
        else:
            reference = None

        raw_logs = [math.log(max(pp.norm, self.norm_floor)) for pp in valid]
        norm_median_log = _median(raw_logs)
        norm_mad = _mad(raw_logs, norm_median_log) if raw_logs else float("nan")
        norm_scale = max(1.4826 * norm_mad, self.norm_log_mad_floor) if raw_logs else 1.0

        reference_scores = [
            self._cosine_flat(pp.flat_hat, reference)
            for pp in valid
            if reference is not None
        ]
        reference_median = _median(reference_scores)
        reference_mad = _mad(reference_scores, reference_median) if reference_scores else float("nan")
        reference_scale = (
            max(1.4826 * reference_mad, self.reference_mad_floor)
            if reference_scores
            else self.reference_mad_floor
        )
        reference_threshold = max(
            self.reference_floor,
            reference_median - self.reference_k * reference_scale,
        )

        sign_scores: list[float] = []
        if reference is not None:
            reference_sign = torch.sign(reference)
            nonzero_reference = reference_sign != 0
            for pp in valid:
                compared = torch.sign(pp.flat_hat)[nonzero_reference]
                expected = reference_sign[nonzero_reference]
                sign_scores.append(float((compared == expected).to(torch.float32).mean()))
        sign_median = _median(sign_scores)
        sign_mad = _mad(sign_scores, sign_median) if sign_scores else float("nan")
        sign_threshold = (
            max(0.2, sign_median - self.reference_k * max(1.4826 * sign_mad, 0.02))
            if sign_scores
            else 0.2
        )

        provisional: list[DetectionResult] = []
        for item, pp, preprocess_time in prepared:
            decision_started = time.perf_counter()
            client_id = int(item["client_id"])
            server_round = int(item["server_round"])
            history = self._hist(client_id)
            history_values = list(history["cos"])
            history_length = len(history_values)

            if pp.degenerate or pp.flat_hat is None or reference is None:
                provisional.append(
                    DetectionResult(
                        client_id=client_id,
                        round=server_round,
                        is_malicious_gt=int(item.get("is_malicious_gt", 0)),
                        attack_active=int(item.get("attack_active", 0)),
                        is_malicious_pred=1,
                        secjudge_ok=False,
                        norm_sum=0.0,
                        cosine=float("nan"),
                        threshold=float("nan"),
                        risk=1.0,
                        pass_flag=False,
                        warmup=False,
                        history_length=history_length,
                        preprocess_time=preprocess_time,
                        raw_norm=pp.norm,
                        norm_median=(
                            math.exp(norm_median_log)
                            if math.isfinite(norm_median_log)
                            else float("nan")
                        ),
                        norm_mad=norm_mad,
                        norm_zscore=float("inf"),
                        reference_threshold=reference_threshold,
                        sign_threshold=sign_threshold,
                        temporal_ok=False,
                        reference_ok=False,
                        decision_reason="degenerate_update",
                        detector_mode=self.detector_mode,
                    )
                )
                continue

            norm_sum = float(torch.dot(pp.flat_hat, pp.flat_hat))
            norm_zscore = abs(math.log(pp.norm) - norm_median_log) / norm_scale
            norm_ok = math.isfinite(norm_zscore) and norm_zscore <= self.norm_k
            reference_cosine = self._cosine_flat(pp.flat_hat, reference)
            reference_ok = reference_cosine >= reference_threshold
            reference_sign = torch.sign(reference)
            nonzero_reference = reference_sign != 0
            sign_agreement = float(
                (
                    torch.sign(pp.flat_hat)[nonzero_reference]
                    == reference_sign[nonzero_reference]
                ).to(torch.float32).mean()
            )

            previous = history["prev_g"]
            temporal_cosine = (
                self._cosine_flat(pp.flat_hat, previous)
                if previous is not None
                else float("nan")
            )
            warmup = previous is None or history_length < self.warmup_rounds
            mu = _ema(history_values, self.beta)
            sigma = _std(history_values)
            temporal_threshold = float("nan")
            temporal_ok = True
            if not warmup:
                temporal_threshold = max(-1.0, min(0.95, mu - self.k * max(sigma, 0.02)))
                temporal_ok = temporal_cosine >= temporal_threshold

            # A synchronized benign trajectory change may lower every temporal
            # score. Cohort-consistent clients are retained in that case.
            cohort_consistent = reference_cosine >= reference_median - reference_scale
            if self.detector_mode == "v1_temporal":
                pass_flag = bool(temporal_ok)
            elif self.detector_mode == "v2_reference_only":
                pass_flag = bool(reference_ok)
            elif self.detector_mode == "v2_norm_only":
                pass_flag = bool(norm_ok)
            else:
                pass_flag = bool(
                    norm_ok and reference_ok and (temporal_ok or cohort_consistent)
                )
            reasons = []
            if self.detector_mode in {"v2_norm_only", "v2_full"} and not norm_ok:
                reasons.append("raw_norm_outlier")
            if (
                self.detector_mode in {"v2_reference_only", "v2_full"}
                and not reference_ok
            ):
                reasons.append("round_direction_outlier")
            if self.detector_mode == "v1_temporal" and not temporal_ok:
                reasons.append("temporal_outlier")
            elif (
                self.detector_mode == "v2_full"
                and not temporal_ok
                and not cohort_consistent
            ):
                reasons.append("temporal_outlier")
            if not reasons:
                reasons.append("accepted")

            temporal_risk = (
                _clip01((1.0 - temporal_cosine) / 2.0)
                if math.isfinite(temporal_cosine)
                else 0.0
            )
            risk = _clip01(
                0.45 * _clip01((1.0 - reference_cosine) / 2.0)
                + 0.25 * temporal_risk
                + 0.20 * _clip01(norm_zscore / self.norm_k)
                + 0.10 * (1.0 - sign_agreement)
            )
            decision_time = time.perf_counter() - decision_started
            provisional.append(
                DetectionResult(
                    client_id=client_id,
                    round=server_round,
                    is_malicious_gt=int(item.get("is_malicious_gt", 0)),
                    attack_active=int(item.get("attack_active", 0)),
                    is_malicious_pred=0 if pass_flag else 1,
                    secjudge_ok=norm_ok,
                    norm_sum=norm_sum,
                    cosine=temporal_cosine,
                    threshold=temporal_threshold,
                    risk=risk,
                    pass_flag=pass_flag,
                    warmup=warmup,
                    history_length=history_length,
                    mu=mu,
                    sigma=sigma,
                    preprocess_time=preprocess_time,
                    decision_time=decision_time,
                    detection_time=decision_time,
                    total_ssdf_time=preprocess_time + decision_time,
                    logical_as1_time=preprocess_time,
                    logical_as2_time=decision_time,
                    raw_norm=pp.norm,
                    norm_median=math.exp(norm_median_log),
                    norm_mad=norm_mad,
                    norm_zscore=norm_zscore,
                    reference_cosine=reference_cosine,
                    reference_threshold=reference_threshold,
                    sign_agreement=sign_agreement,
                    sign_threshold=sign_threshold,
                    temporal_ok=temporal_ok,
                    reference_ok=reference_ok,
                    decision_reason=";".join(reasons),
                    detector_mode=self.detector_mode,
                )
            )

        if self.detector_mode == "v1_temporal":
            # Compatibility baseline: the original temporal-only detector
            # advanced history even for rejected trajectories.
            pp_by_client = {int(item["client_id"]): pp for item, pp, _ in prepared}
            for result in provisional:
                pp = pp_by_client[result.client_id]
                if pp.degenerate or pp.flat_hat is None:
                    continue
                history = self._hist(result.client_id)
                if math.isfinite(result.cosine):
                    history["cos"].append(float(result.cosine))
                    history["cos"] = history["cos"][-self.history_limit :]
                history["prev_g"] = pp.flat_hat
                result.history_updated = True
            return provisional

        # Rejected trajectories never become trusted history. If the entire
        # cohort is rejected, no detector state is advanced.
        if any(result.pass_flag for result in provisional):
            pp_by_client = {int(item["client_id"]): pp for item, pp, _ in prepared}
            for result in provisional:
                if not result.pass_flag:
                    continue
                history = self._hist(result.client_id)
                if math.isfinite(result.cosine):
                    history["cos"].append(float(result.cosine))
                    history["cos"] = history["cos"][-self.history_limit :]
                history["prev_g"] = pp_by_client[result.client_id].flat_hat
                result.history_updated = True
        return provisional

    def detect(
        self,
        client_id: int,
        server_round: int,
        delta: dict[str, torch.Tensor],
        is_malicious_gt: int,
        attack_active: int,
    ) -> DetectionResult:
        """Compatibility wrapper; round-aware callers should use detect_round."""
        return self.detect_round(
            [
                {
                    "client_id": client_id,
                    "server_round": server_round,
                    "delta": delta,
                    "is_malicious_gt": is_malicious_gt,
                    "attack_active": attack_active,
                }
            ]
        )[0]
