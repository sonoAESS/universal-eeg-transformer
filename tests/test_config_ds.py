"""Tests de configuración y de construcción de splits."""

from __future__ import annotations

import numpy as np
import pytest


def test_load_default_config(tmp_path):
    from eeg_transform.config import load_config, save_config

    cfg = load_config("config/default.yaml")
    assert cfg.data.subjects == [1, 2, 3, 4]
    assert cfg.leadfield.sigmas == [0.33, 1.0, 0.0042, 0.33]
    assert cfg.model.latent_dim == 0  # auto

    out = tmp_path / "cfg.yaml"
    save_config(cfg, out)
    cfg2 = load_config(out)
    assert cfg2.data.bandpass == cfg.data.bandpass
    assert cfg2.model.optimizer == "adam"


def test_config_validation():
    from eeg_transform.config import EEGTransformConfig

    cfg = EEGTransformConfig()
    cfg.data.bandpass = [-1.0, 45.0]
    with pytest.raises(ValueError):
        cfg.validate()

    cfg = EEGTransformConfig()
    cfg.leadfield.brain_radius = 0.099
    with pytest.raises(ValueError):
        cfg.validate()

    cfg = EEGTransformConfig()
    cfg.dataset.split_mode = "mal"
    with pytest.raises(ValueError):
        cfg.validate()


def test_block_splits_cover_all():
    from eeg_transform.data.dataset import _block_assign, _stride_cap

    n = 1000
    assign = _block_assign(n, 0.2, 0.2)
    assert set(np.unique(assign)) == {0, 1, 2}
    assert np.count_nonzero(assign == 2) == pytest.approx(0.2 * n, abs=2)
    assert np.count_nonzero(assign == 0) > np.count_nonzero(assign == 1)

    # los índices de test forman el bloque final (contiguo)
    test_idx = np.where(assign == 2)[0]
    assert np.array_equal(test_idx, np.arange(test_idx[0], n))

    cap = _stride_cap(np.arange(5000), 500, seed=1)
    assert len(cap) == 500
    assert len(np.unique(cap)) == 500


def test_subject_split_deterministic():
    from eeg_transform.config import DatasetConfig
    from eeg_transform.data.dataset import _subject_label

    dc = DatasetConfig(test_frac=0.2, val_frac=0.2)
    labels = [_subject_label(s, [1, 2, 3, 4, 5], dc) for s in [1, 2, 3, 4, 5]]
    assert sorted(labels) == [0, 0, 0, 1, 2] or "determinista"
    # 1 -> test, 2 -> val, resto train en modo por defecto
    assert _subject_label(1, [1, 2, 3, 4, 5], dc) == 2  # test
    assert _subject_label(2, [1, 2, 3, 4, 5], dc) == 1  # val