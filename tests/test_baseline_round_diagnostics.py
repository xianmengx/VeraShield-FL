"""Regression checks for diagnostic coverage without changing aggregation."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import csv
from types import SimpleNamespace

import pytest
import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg

from pytorchexample.experiment.aggregation_strategy import AggregationStrategy
from pytorchexample.experiment.aggregation_diagnostics import validate_aggregation_record
from pytorchexample.experiment.recorded_fedavg import RecordingFedAvg
from pytorchexample.experiment.recorder import ExperimentRecorder


def reply(cid=0, value=1., invalid=0):
    return SimpleNamespace(content={
        'arrays': ArrayRecord({'w': torch.tensor([value])}),
        'metrics': {'client_partition_id': cid, 'invalid_update': invalid, 'num-examples': 1},
    })


@pytest.mark.parametrize('method', ['fedavg', 'krum', 'trimmed_mean', 'feddmc'])
def test_rejected_updates_record_scheduled_tamper_without_applying_it(monkeypatch, tmp_path, method):
    monkeypatch.setattr(FedAvg, '_check_and_log_replies', lambda *args: ([reply(invalid=1), reply(value=float('nan'))], []))
    strategy = AggregationStrategy(aggregation_method=method, vl_tamper_rounds=[20])
    strategy._current_arrays = ArrayRecord({'w': torch.tensor([0.])})
    assert strategy.aggregate_train(20, []) == (None, None)
    row = strategy.aggregation_records[20]
    assert row['aggregation_status'] == 'aggregation_failed_no_valid_replies'
    assert row['received_replies'] == row['invalid_update_replies'] == 2
    assert row['tamper_scheduled'] == 1 and row['tamper_applied'] == 0
    assert row['dropout_failures'] == row['runtime_failures'] == 0
    assert validate_aggregation_record(row) == []
    recorder = ExperimentRecorder(str(tmp_path))
    recorder._write_aggregation(strategy.aggregation_records)
    with (tmp_path / 'aggregation_diagnostics.csv').open() as stream:
        saved = next(csv.DictReader(stream))
    assert saved['invalid_update_replies'] == '2'
    assert saved['tamper_applied'] == '0'


def test_all_post_share_unavailable_is_distinct_from_invalid_reply(monkeypatch):
    monkeypatch.setattr(FedAvg, '_check_and_log_replies', lambda *args: ([reply()], []))
    strategy = AggregationStrategy(unavailable_by_round={21: [0]}, vl_tamper_rounds=[21])
    strategy._current_arrays = ArrayRecord({'w': torch.tensor([0.])})
    assert strategy.aggregate_train(21, []) == (None, None)
    row = strategy.aggregation_records[21]
    assert row['aggregation_status'] == 'aggregation_failed_due_to_insufficient_clients'
    assert row['invalid_update_replies'] == 0
    assert row['dropped_clients'] == 1 and row['total_clients'] == 1
    assert row['tamper_applied'] == 0


def test_krum_insufficient_inputs_keeps_f_and_records_unapplied_tamper(monkeypatch):
    monkeypatch.setattr(FedAvg, '_check_and_log_replies', lambda *args: ([reply(i) for i in range(6)], []))
    strategy = AggregationStrategy(aggregation_method='krum', krum_f=2, vl_tamper_rounds=[40])
    strategy._current_arrays = ArrayRecord({'w': torch.tensor([0.])})
    assert strategy.aggregate_train(40, []) == (None, None)
    row = strategy.aggregation_records[40]
    assert row['krum_f'] == 2 and row['krum_valid'] is False
    assert row['tamper_scheduled'] == 1 and row['tamper_applied'] == 0


def test_clean_fedavg_output_is_unchanged(monkeypatch):
    monkeypatch.setattr(FedAvg, '_check_and_log_replies', lambda *args: ([reply(0, 2.), reply(1, 4.)], []))
    strategy = AggregationStrategy()
    strategy._current_arrays = ArrayRecord({'w': torch.tensor([1.])})
    strategy.train_metrics_aggr_fn = lambda *args: None
    arrays, _ = strategy.aggregate_train(1, [])
    assert torch.equal(arrays.to_torch_state_dict()['w'], torch.tensor([3.]))
    assert strategy.aggregation_records[1]['effective_clients'] == 2


@pytest.mark.parametrize('n,normal,emergency', [(20,16,12), (10,8,6)])
def test_recovery_metadata_uses_actual_parameters(tmp_path, n, normal, emergency):
    recorder = ExperimentRecorder(str(tmp_path))
    config = recorder._build_config(
        {'share-n': n, 'normal-threshold': normal, 'emergency-threshold': emergency},
        {}, [{'num_clients': n}], 4)
    assert f'A=Shamir({n},{normal})' in config['d2_v2_semantics']
    assert f'B=Shamir({n},{emergency})' in config['d2_v2_semantics']
    assert config['failure_source'] != 'dropout_only'
