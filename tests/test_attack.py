
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import numpy as np
import torch

from pytorchexample.experiment.attack import apply_update_attack, update_l2_norm
from pytorchexample.task import add_backdoor_trigger


def test_random_gradient_replaces_direction_and_controls_norm():
    delta = {
        "weight": torch.tensor([[1.0, -2.0], [0.5, 3.0]]),
        "bias": torch.tensor([0.25, -0.75]),
    }
    attacked = apply_update_attack(
        "random_gradient", delta, strength=5.0,
        noise_rng=np.random.default_rng([7, 4, 17]),
    )
    assert np.isclose(update_l2_norm(attacked), 5.0 * update_l2_norm(delta), rtol=1e-6)
    original = torch.cat([value.reshape(-1) for value in delta.values()])
    replacement = torch.cat([value.reshape(-1) for value in attacked.values()])
    cosine = torch.nn.functional.cosine_similarity(original, replacement, dim=0)
    assert abs(float(cosine)) < 0.95


def test_random_gradient_shared_seed_produces_shared_direction():
    first = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    second = {"weight": torch.tensor([4.0, 5.0, 6.0])}
    seed = [21, 8, 17]
    attacked_first = apply_update_attack(
        "random_gradient", first, strength=3.0,
        noise_rng=np.random.default_rng(seed),
    )
    attacked_second = apply_update_attack(
        "random_gradient", second, strength=3.0,
        noise_rng=np.random.default_rng(seed),
    )
    cosine = torch.nn.functional.cosine_similarity(
        attacked_first["weight"], attacked_second["weight"], dim=0
    )
    assert np.isclose(float(cosine), 1.0, atol=1e-6)


def test_coordinated_gaussian_retains_update_and_has_relative_noise_norm():
    delta = {"weight": torch.tensor([1.0, -2.0, 0.5, 3.0])}
    attacked = apply_update_attack(
        "coordinated_gaussian", delta, strength=1.0,
        noise_rng=np.random.default_rng([7, 4, 17]),
    )
    noise = {key: attacked[key] - delta[key] for key in delta}
    assert np.isclose(update_l2_norm(noise), update_l2_norm(delta), rtol=1e-6)
    assert not torch.equal(attacked["weight"], noise["weight"])


def test_coordinated_gaussian_uses_shared_noise_direction():
    first = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    second = {"weight": torch.tensor([4.0, 5.0, 6.0])}
    seed = [21, 8, 17]
    attacked_first = apply_update_attack(
        "coordinated_gaussian", first, strength=1.0,
        noise_rng=np.random.default_rng(seed),
    )
    attacked_second = apply_update_attack(
        "coordinated_gaussian", second, strength=1.0,
        noise_rng=np.random.default_rng(seed),
    )
    noise_first = attacked_first["weight"] - first["weight"]
    noise_second = attacked_second["weight"] - second["weight"]
    cosine = torch.nn.functional.cosine_similarity(noise_first, noise_second, dim=0)
    assert np.isclose(float(cosine), 1.0, atol=1e-6)


def test_backdoor_model_replacement_scales_poisoned_update():
    delta = {"weight": torch.tensor([1.0, -2.0, 0.5])}
    attacked = apply_update_attack("backdoor", delta, strength=5.0)
    assert torch.equal(attacked["weight"], 5.0 * delta["weight"])


def test_backdoor_trigger_only_changes_bottom_right_patch():
    images = torch.zeros((2, 3, 8, 8))
    triggered = add_backdoor_trigger(images, trigger_size=3)
    assert torch.equal(images, torch.zeros_like(images))
    assert torch.all(triggered[..., -3:, -3:] == 1.0)
    assert torch.all(triggered[..., :-3, :] == 0.0)
    assert torch.all(triggered[..., :, :-3] == 0.0)
