
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
from pytorchexample.experiment.logical_servers import LogicalTwoServerStore
from pytorchexample.experiment.recovery import RecoveryStateStore


def test_round_robin_routing_preserves_share_order():
    store = LogicalTwoServerStore()
    store.provision(["s0", "s1", "s2", "s3", "s4"])

    assert store.c1.shares == ["s0", "s2", "s4"]
    assert store.c2.shares == ["s1", "s3"]
    assert store.combined() == ["s0", "s1", "s2", "s3", "s4"]
    assert store.audit() == {
        "server_topology": "logical_two_server",
        "server_names": ["C1", "C2"],
        "independent_server_processes": False,
        "c1_share_count": 3,
        "c2_share_count": 2,
        "total_share_count": 5,
    }


def test_recovery_store_exposes_logical_server_boundary():
    store = RecoveryStateStore()
    shares = ["s0", "s1", "s2", "s3"]
    store.put(3, 4, "A", shares)
    assert store.get(3, 4, "A") == shares
    audit = store.get_logical_server_audit(3, 4, "A")
    assert audit["server_names"] == ["C1", "C2"]
    assert audit["independent_server_processes"] is False
    assert audit["c1_share_count"] == 2
    assert audit["c2_share_count"] == 2
