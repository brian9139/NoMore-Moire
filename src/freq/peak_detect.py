import json, os
import numpy as np
from scipy.ndimage import maximum_filter

from src.freq.fft2d import load_specpack_npz

def save_peaks_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_peaks_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def find_local_maximum(log_mag, neighborhood_size=9):
    data_max = maximum_filter(log_mag, neighborhood_size)
    maximum = (log_mag == data_max)
    
    h, w = log_mag.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    peak_y = y_coords[maximum]
    peak_x = x_coords[maximum]
    peak_values = log_mag[maximum]
    
    peaks = list(zip(peak_x, peak_y, peak_values))
    return peaks

def filter_peaks(peaks, center, r_min, max_pairs):
    cx, cy = center
    filtered_peaks = []
    
    for px, py, value in peaks:
        dx = px - cx
        dy = py - cy
        distance = np.sqrt(dx**2 + dy**2)

        if distance < r_min:
            continue
            
        filtered_peaks.append((px, py, value, distance))

    filtered_peaks.sort(key=lambda x: x[2], reverse=True)
    
    peak_pairs = []
    used_peaks = set()
    
    for i, (px1, py1, val1, dist1) in enumerate(filtered_peaks):
        if i in used_peaks:
            continue
            
        sym_px = 2*cx - px1
        sym_py = 2*cy - py1
        
        best_match = None
        min_dist = float('inf')
        
        for j, (px2, py2, val2, dist2) in enumerate(filtered_peaks):
            if j in used_peaks or j == i:
                continue
                
            sym_dist = np.sqrt((px2 - sym_px)**2 + (py2 - sym_py)**2)
            
            if sym_dist < 3.0 and sym_dist < min_dist:
                best_match = j
                min_dist = sym_dist
        
        if best_match is not None:
            px2, py2, val2, dist2 = filtered_peaks[best_match]
            
            dx1 = px1 - cx
            dy1 = py1 - cy
            angle = np.arctan2(dy1, dx1) * 180 / np.pi
            
            avg_strength = (val1 + val2) / 2
            radius = (dist1 + dist2) / 2
            
            peak_pairs.append({
                'peak1': {'x': int(px1), 'y': int(py1), 'strength': float(val1)},
                'peak2': {'x': int(px2), 'y': int(py2), 'strength': float(val2)},
                'radius': float(radius),
                'angle': float(angle),
                'avg_strength': float(avg_strength)
            })
            
            used_peaks.add(i)
            used_peaks.add(best_match)
    
    peak_pairs.sort(key=lambda x: x['avg_strength'], reverse=True)
    return peak_pairs[:max_pairs]

def _detect_peaks(log_mag, r_min, max_pairs): #detect peaks from given logmag and save as json
    h, w = log_mag.shape
    center = (w // 2, h // 2)
    
    all_peaks = find_local_maximum(log_mag)
    
    peak_pairs = filter_peaks(all_peaks, center, r_min=r_min, max_pairs=max_pairs)
    
    return peak_pairs

def detect_peaks(r_min, max_pairs, path = './out'): #loads from npz and detects peaks and save as json
    categories = ('real', 'synth')
    
    for category in categories:
        paths = os.listdir(os.path.join(path, category))
        for name in paths:
            if os.path.isdir(os.path.join(path, category, name)):
                specpak_path = os.path.join(path, category, name, 'specpack.npz')
                mag, phase, log_mag = load_specpack_npz(specpak_path)
                
                peak_pairs = _detect_peaks(log_mag, r_min=r_min, max_pairs=max_pairs)
                
                save_peaks_json(os.path.join(path, category, name, 'peaks.json'), peak_pairs)