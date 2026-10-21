"""Independent Flower-compatible port of FedDMC's default code path.

The author's runner defaults to ``pca_hdbscan_b``: PCA features, HDBSCAN
clustering, the largest cluster as benign, and cumulative majority voting to
stabilize client labels. Centering in PCA makes local parameters and their
same-round deltas equivalent detector inputs.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]

from collections import defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from .recovery import flatten_state


class FedDMCDetector:
    """Detect suspicious client updates and average the retained deltas."""

    def __init__(self, pca_components: int = 10, min_cluster_size: int = 3) -> None:
        if pca_components < 1 or min_cluster_size < 2:
            raise ValueError("FedDMC parameters are outside their valid range")
        self.pca_components = pca_components
        self.min_cluster_size = min_cluster_size
        self.votes: dict[int, int] = defaultdict(int)
        self.observations: dict[int, int] = defaultdict(int)

    def detect(self, client_ids: list[int], client_updates: list[dict]):
        if len(client_ids) != len(client_updates):
            raise ValueError("client_ids and client_updates must have equal length")
        n_clients = len(client_updates)
        if n_clients < 2:
            raise ValueError("FedDMC requires at least two client updates")

        flattened = [flatten_state(update)[0] for update in client_updates]
        feature_matrix = np.stack(flattened)
        n_components = min(
            self.pca_components,
            n_clients - 1,
            feature_matrix.shape[1],
        )
        if n_components < 1 or not np.isfinite(feature_matrix).all():
            raise ValueError("FedDMC received empty or non-finite client updates")

        projected = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=0,
        ).fit_transform(feature_matrix)

        if n_clients < self.min_cluster_size:
            labels = np.zeros(n_clients, dtype=int)
            cluster_label = 0
            clustering_status = "insufficient_clients"
        else:
            labels = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                copy=True,
            ).fit_predict(projected)
            counts = np.bincount(labels - labels.min())
            cluster_label = int(np.argmax(counts) + labels.min())
            clustering_status = "ok"

        raw_suspect_idx = {
            idx for idx, label in enumerate(labels) if int(label) != cluster_label
        }
        suspected: list[int] = []
        vote_counts: dict[int, int] = {}
        observed_counts: dict[int, int] = {}
        for idx, client_id in enumerate(client_ids):
            cid = int(client_id)
            self.observations[cid] += 1
            self.votes[cid] += int(idx in raw_suspect_idx)
            vote_counts[cid] = self.votes[cid]
            observed_counts[cid] = self.observations[cid]
            if self.votes[cid] > self.observations[cid] / 2:
                suspected.append(cid)

        suspected_set = set(suspected)
        retained = [
            idx for idx, client_id in enumerate(client_ids)
            if int(client_id) not in suspected_set
        ]
        # Match the author implementation's all-filtered safeguard: average all
        # updates rather than leaving the global model unchanged.
        fallback_all = not retained
        if fallback_all:
            retained = list(range(n_clients))
        aggregate = np.mean(feature_matrix[retained], axis=0)

        diag = {
            "method": "feddmc",
            "aggregation_status": "ok",
            "detector": "pca_hdbscan_cumulative_majority",
            "clustering_status": clustering_status,
            "pca_components": int(n_components),
            "min_cluster_size": self.min_cluster_size,
            "benign_cluster_label": cluster_label,
            "client_ids": [int(cid) for cid in client_ids],
            "raw_suspect_ids": [int(client_ids[i]) for i in sorted(raw_suspect_idx)],
            "suspected_client_ids": sorted(suspected),
            "cluster_labels": {
                int(client_ids[i]): int(labels[i]) for i in range(n_clients)
            },
            "history_vote_counts": vote_counts,
            "history_observation_counts": observed_counts,
            "retained_client_ids": [int(client_ids[i]) for i in retained],
            "fallback_all_updates": fallback_all,
            "effective_clients": len(retained),
        }
        return aggregate, diag
