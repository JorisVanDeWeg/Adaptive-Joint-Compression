import numpy as np
from scipy.stats import skew, kurtosis, entropy


def compute_features(window):
    rms = np.sqrt(np.mean(np.square(window)))
    clearance_factor = np.max(window) / (np.mean(np.abs(window)) ** 2)
    crest_factor = np.max(window) / rms if rms != 0 else 0

    return [
        np.mean(window),
        np.std(window),
        np.min(window),
        np.max(window),
        np.max(window) - np.mean(window),
        np.max(window) - np.min(window),
        rms,
        clearance_factor,
        crest_factor,
        kurtosis(window),
        skew(window)]
    

def extract_window_features(X, window_size):
    n_samples, n_timesteps = X.shape
    feature_list = []

    for i in range(n_samples):
        features = []
        for j in range(0, n_timesteps - window_size + 1, window_size):
            window = X[i, j:j + window_size]
            features.append(compute_features(window))
        feature_list.append(features)

    return np.array(feature_list)
