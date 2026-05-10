from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")


def assert_tree_finite(tree):
    for leaf in jax.tree.leaves(tree):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_presets_and_schedule_formulas():
    from bpc.config import SELECTED_PRESET, make_presets
    from bpc.updates.minibatch_updates import compute_kappa, compute_scale

    presets = make_presets()
    cfg = presets[SELECTED_PRESET]
    assert cfg.name == "paper_svi_delay5000_cap005_3hidden_zeroone"
    assert cfg.hidden_layers == 3
    assert cfg.update_mode == "svi_scaled"
    assert cfg.hidden_init == "zeros"

    assert compute_kappa(12, 2, cfg) == 0.05
    assert compute_scale(60000, 128, cfg) == 468.75

    literal = presets["paper_literal_batch_local"]
    assert compute_kappa(12, 2, literal) == pytest.approx(12.0 ** -0.25)
    assert compute_scale(60000, 128, literal) == 1.0

    power = presets["svi_power_half_delay5000_cap005_2hidden_std"]
    assert compute_scale(60000, 128, power) == pytest.approx(math.sqrt(468.75))


def test_prior_and_posterior_initialization_invariants():
    from bpc.config import DTYPE
    from bpc.priors.weight_prior import init_posterior, init_prior

    layer_dims = (2, 3, 2)
    prior = init_prior(layer_dims)

    assert len(prior) == 2
    np.testing.assert_allclose(np.asarray(prior[0].V_inv), 0.1 * np.eye(3), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(prior[0].MV), np.zeros((3, 3)), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(prior[0].S), 0.001 * np.eye(3), atol=0, rtol=0)
    assert float(prior[0].nu_shift) == 4.0

    np.testing.assert_allclose(np.asarray(prior[1].V_inv), 0.1 * np.eye(4), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(prior[1].MV), np.zeros((2, 4)), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(prior[1].S), 0.001 * np.eye(2), atol=0, rtol=0)
    assert float(prior[1].nu_shift) == 5.0

    key = jax.random.PRNGKey(0)
    params_a = init_posterior(layer_dims, key)
    params_b = init_posterior(layer_dims, key)
    for eta_a, eta_b in zip(params_a, params_b):
        np.testing.assert_allclose(np.asarray(eta_a.V_inv), np.asarray(eta_b.V_inv), atol=0, rtol=0)
        np.testing.assert_allclose(np.asarray(eta_a.MV), np.asarray(eta_b.MV), atol=0, rtol=0)
        np.testing.assert_allclose(np.asarray(eta_a.S), np.asarray(eta_b.S), atol=0, rtol=0)
        assert eta_a.V_inv.dtype == DTYPE
        np.testing.assert_allclose(np.asarray(eta_a.S), np.asarray(eta_a.S.T), atol=1e-12, rtol=1e-12)
    assert_tree_finite(params_a)


def test_matrix_normal_wishart_prior_moments_are_well_formed():
    from bpc.config import BPCConfig, JITTER
    from bpc.distributions.matrix_normal_wishart import natural_to_moment, posterior_moments
    from bpc.priors.weight_prior import init_prior

    layer_dims = (2, 3, 2)
    prior = init_prior(layer_dims)
    M, V, Psi, nu, psi_inv_raw = natural_to_moment(prior[0], d_y=3, d_x=3)

    np.testing.assert_allclose(np.asarray(M), np.zeros((3, 3)), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(V), (1.0 / (0.1 + 2.0 * JITTER)) * np.eye(3), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(psi_inv_raw), 0.001 * np.eye(3), atol=0, rtol=0)
    np.testing.assert_allclose(np.asarray(Psi), (1.0 / (0.001 + JITTER)) * np.eye(3), rtol=1e-12)
    assert float(nu) == 5.0

    moms = posterior_moments(prior, layer_dims, BPCConfig(name="moments"))
    assert len(moms) == 2
    assert moms[0].E_prec.shape == (3, 3)
    assert moms[0].E_prec_W.shape == (3, 3)
    assert moms[0].E_Wt_prec_W.shape == (3, 3)
    assert_tree_finite(moms)


def test_sufficient_statistics_known_values():
    from bpc.config import BPCConfig, DTYPE
    from bpc.updates.hebbian_updates import sufficient_statistics

    cfg = BPCConfig(name="stats")
    layer_dims = (2, 2, 2)
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    hidden = (jnp.asarray([[0.5, -0.5], [1.0, 2.0]], dtype=DTYPE),)
    y = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)

    stats = sufficient_statistics(hidden, x, y, layer_dims, cfg)

    np.testing.assert_allclose(np.asarray(stats[0].T1), np.array([[10.0, 14.0, 4.0], [14.0, 20.0, 6.0], [4.0, 6.0, 2.0]]))
    np.testing.assert_allclose(np.asarray(stats[0].T2), np.array([[3.5, 5.0, 1.5], [5.5, 7.0, 1.5]]))
    np.testing.assert_allclose(np.asarray(stats[0].T3), np.array([[1.25, 1.75], [1.75, 4.25]]))
    assert float(stats[0].T4) == 2.0

    np.testing.assert_allclose(np.asarray(stats[1].T1), np.array([[1.25, 2.0, 1.5], [2.0, 4.0, 2.0], [1.5, 2.0, 2.0]]))
    np.testing.assert_allclose(np.asarray(stats[1].T2), np.array([[0.5, 0.0, 1.0], [1.0, 2.0, 1.0]]))
    np.testing.assert_allclose(np.asarray(stats[1].T3), np.eye(2))
    assert float(stats[1].T4) == 2.0


def test_hidden_inference_update_and_prediction_smoke():
    from bpc.config import BPCConfig, DTYPE
    from bpc.distributions.matrix_normal_wishart import posterior_moments
    from bpc.inference.gradients import bpc_hidden_gradients
    from bpc.inference.hidden_state_inference import adam_latent_inference_diag
    from bpc.priors.hidden_state_prior import hidden_init
    from bpc.priors.weight_prior import init_posterior, init_prior
    from bpc.training.metrics import classification_accuracy
    from bpc.updates.hebbian_updates import sufficient_statistics
    from bpc.updates.posterior_updates import compute_mstep_diagnostics, compute_stats_diagnostics, interpolate_eta, stats_to_eta

    cfg = BPCConfig(name="tiny", hidden=3, hidden_layers=1, latent_steps=3, hidden_init="feedforward", eval_batch_size=2)
    layer_dims = (2, 3, 2)
    key = jax.random.PRNGKey(1)
    x = jnp.asarray([[0.2, -0.4], [1.0, 0.5], [-0.7, 0.3], [0.0, 0.9]], dtype=DTYPE)
    y = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)

    params = init_posterior(layer_dims, key)
    moms = posterior_moments(params, layer_dims, cfg)
    h0 = hidden_init(params, layer_dims, x, cfg, key)
    grads = bpc_hidden_gradients(h0, moms, x, y, cfg)

    assert len(h0) == 1
    assert h0[0].shape == (4, 3)
    assert grads[0].shape == (4, 3)
    assert_tree_finite(grads)

    hidden, diag = adam_latent_inference_diag(h0, moms, x, y, cfg)
    assert hidden[0].shape == (4, 3)
    assert diag.grad_norm_per_step.shape == (3, 1)
    assert diag.energy_per_step.shape == (3, 2)
    assert_tree_finite((hidden, diag))

    stats = sufficient_statistics(hidden, x, y, layer_dims, cfg)
    target = stats_to_eta(init_prior(layer_dims), stats, jnp.asarray(1.5, dtype=DTYPE))
    updated = interpolate_eta(params, target, jnp.asarray(0.25, dtype=DTYPE))
    assert_tree_finite((compute_stats_diagnostics(stats), compute_mstep_diagnostics(params, updated, layer_dims)))

    prior_params = init_prior((2, 2))
    acc = classification_accuracy(prior_params, (2, 2), x[:, :2], y, cfg)
    assert acc == 0.5


def test_mnist_npz_loader_standardize_scalar(tmp_path):
    from bpc.config import BPCConfig
    from bpc.data import load_mnist

    x_train_raw = np.arange(2 * 28 * 28, dtype=np.float64).reshape(2, 28, 28)
    x_test_raw = np.arange(2 * 28 * 28, 3 * 28 * 28, dtype=np.float64).reshape(1, 28, 28)
    y_train = np.array([1, 3])
    y_test = np.array([3])
    path = tmp_path / "mnist_fake.npz"
    np.savez(path, x_train=x_train_raw, y_train=y_train, x_test=x_test_raw, y_test=y_test)

    cfg = BPCConfig(name="npz", mnist_source="npz", mnist_npz=str(path), normalize="standardize_scalar")
    x_train, y_train_oh, x_test, y_test_oh = load_mnist(cfg)

    x_train_scaled = x_train_raw.reshape((-1, 784)) / 255.0
    x_test_scaled = x_test_raw.reshape((-1, 784)) / 255.0
    mu = float(x_train_scaled.mean())
    sd = float(x_train_scaled.std() + 1e-6)

    np.testing.assert_allclose(x_train, (x_train_scaled - mu) / sd)
    np.testing.assert_allclose(x_test, (x_test_scaled - mu) / sd)
    np.testing.assert_allclose(y_train_oh, np.eye(10)[[1, 3]])
    np.testing.assert_allclose(y_test_oh, np.eye(10)[[3]])
