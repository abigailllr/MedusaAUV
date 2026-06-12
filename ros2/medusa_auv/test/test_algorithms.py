import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "medusa_auv"))

import algorithms as a


def test_haversine_zero():
    assert a.haversine((47.36, 8.54), (47.36, 8.54)) == 0.0


def test_haversine_known_distance():
    d = a.haversine((47.0, 8.0), (47.0, 8.01))
    assert 750 < d < 770


def test_bearing_east():
    assert abs(a.bearing((47.0, 8.0), (47.0, 8.1)) - 90.0) < 1.0


def test_bearing_north():
    assert a.bearing((47.0, 8.0), (47.1, 8.0)) < 1.0 or a.bearing((47.0, 8.0), (47.1, 8.0)) > 359.0


def test_temp_score_optimal():
    assert a.temp_score(26.0, 25.0, 27.0, 5.0) == 1.0


def test_temp_score_falloff():
    assert a.temp_score(30.0, 25.0, 27.0, 5.0) == 0.4


def test_temp_score_far():
    assert a.temp_score(10.0, 25.0, 27.0, 5.0) == 0.0


def test_bloom_forecast_range():
    f = a.bloom_forecast(26.0, 1.0, 10.0, 25.0, 27.0, 5.0, 10.0)
    assert 0.0 <= f <= 1.0
    assert f == 1.0


def test_bloom_forecast_cold_dark():
    assert a.bloom_forecast(5.0, 0.0, 0.0, 25.0, 27.0, 5.0, 10.0) == 0.0


def test_bloom_severity():
    assert a.bloom_severity(0, 3, 10) == 0
    assert a.bloom_severity(5, 3, 10) == 1
    assert a.bloom_severity(12, 3, 10) == 2


def test_fuse_confidence():
    assert a.fuse_confidence(1.0, 0.0, 0.5) == 0.5
    assert a.fuse_confidence(0.8, 0.8, 0.7) == 0.8
    assert a.fuse_confidence(1.0, 1.0, 0.5) == 1.0


def test_band_ratio_detects_pulse():
    fps = 20.0
    t = np.arange(64) / fps
    pulsing = np.sin(2 * np.pi * 0.6 * t)
    noise = np.random.default_rng(0).normal(0, 0.01, 64)
    assert a.band_ratio(pulsing, fps, 0.25, 1.0) > a.band_ratio(noise, fps, 0.25, 1.0)


def test_band_ratio_bounds():
    r = a.band_ratio([1.0] * 64, 20.0, 0.25, 1.0)
    assert 0.0 <= r <= 1.0


def test_confusion_counts():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]
    assert a.confusion_counts(y_true, y_pred) == (2, 1, 1, 1)


def test_precision_recall_f1_perfect():
    m = a.precision_recall_f1([1, 0, 1, 0], [1, 0, 1, 0])
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_precision_recall_f1_values():
    m = a.precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["accuracy"] == 0.5


def test_precision_recall_f1_empty_pred():
    m = a.precision_recall_f1([1, 1, 0], [0, 0, 0])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_gray_world_neutralizes():
    img = np.zeros((4, 4, 3), dtype=float)
    img[:, :, 0] = 50.0
    img[:, :, 1] = 100.0
    img[:, :, 2] = 150.0
    out = a.gray_world(img)
    means = out.reshape(-1, 3).mean(axis=0)
    assert abs(means[0] - means[2]) < abs(50.0 - 150.0)


if __name__ == "__main__":
    import inspect
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError:
                failed += 1
                print(f"FAIL {name}")
    sys.exit(1 if failed else 0)
