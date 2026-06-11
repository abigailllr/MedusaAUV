import math
import numpy as np


def gray_world(image):
    means = image.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scaled = image * (gray / (means + 1e-6))
    return np.clip(scaled, 0.0, 255.0)


def haversine(a, b):
    r = 6371000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def bearing(a, b):
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def temp_score(temperature, opt_min, opt_max, falloff):
    if opt_min <= temperature <= opt_max:
        return 1.0
    distance = min(abs(temperature - opt_min), abs(temperature - opt_max))
    return max(0.0, 1.0 - distance / falloff)


def bloom_forecast(temperature, light, density, opt_min, opt_max, falloff, density_reference):
    density_score = min(density / max(density_reference, 1e-6), 1.0)
    score = 0.5 * temp_score(temperature, opt_min, opt_max, falloff) + 0.3 * light + 0.2 * density_score
    return max(0.0, min(1.0, score))


def bloom_severity(count, low, high):
    if count >= high:
        return 2
    if count >= low:
        return 1
    return 0


def band_ratio(motion, fps, band_min, band_max):
    signal = np.asarray(motion, dtype=float)
    signal = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    band = spectrum[(freqs >= band_min) & (freqs <= band_max)].sum()
    return float(min(band / (spectrum.sum() + 1e-6), 1.0))


def fuse_confidence(model_conf, pulse_conf, model_weight):
    return max(0.0, min(1.0, model_weight * model_conf + (1.0 - model_weight) * pulse_conf))
