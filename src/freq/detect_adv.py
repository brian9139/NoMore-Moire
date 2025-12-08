# freq/detect_adv.py
import numpy as np
import json
import os
import glob
import cv2
from pathlib import Path

from src.freq.window import apply_window
from src.freq.logpolar import cart2logpolar
from src.freq.orient_energy import compute_orientation_energy
from src.freq.peak_detect import save_peaks_json
from src.freq.orient_energy import non_maximum_suppression, filter_peaks_by_threshold

def _load_images(category):
    SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    image_list = []
    image_dir = Path('./data') / category
    
    for ext in SUPPORTED_EXTS:
        image_paths = glob.glob(str(image_dir / f'*{ext}'))
        for img_path in image_paths:
            name = Path(img_path).stem
            if name.endswith('_gt'):
                continue
            
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                image_list.append((name, img))
    
    return image_list

def _process_image_advanced(image):
    if len(image.shape) == 3:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)
        gray = img_yuv[:, :, 0]
    else:
        gray = image.astype(np.float32)
    
    # gray = apply_window(gray)
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    logmag = np.log(magnitude + 1e-8)
    
    logpolar = cart2logpolar(logmag)
    
    orient_energy, _ = compute_orientation_energy(
        logpolar,
        rho_width=10,
        theta_width=30
    )
    
    return {
        'mag': magnitude,
        'phase': phase,
        'logmag': logmag,
        'logpolar': logpolar,
        'orient_energy': orient_energy
    }

def _save_advanced_specpack(spec_data, filepath):
    np.savez_compressed(
        filepath,
        mag=spec_data['mag'],
        phase=spec_data['phase'],
        logmag=spec_data['logmag'],
        logpolar=spec_data['logpolar'],
        orient_energy=spec_data['orient_energy']
    )

def specpack_adv(loader=None):
    if loader is not None:
        return

    categories = ['synth', 'real']
    
    print("Starting advanced FFT processing...")
    
    for category in categories:
        images = _load_images(category)
        
        if not images:
            print(f"No images found in category: {category}")
            continue
        
        print(f"Processing category: {category}, Number of images: {len(images)}")
        
        processed_count = 0
        for name, img in images:
            try:
                print(f"[spec_adv] Processing image: {name}")
                
                spec_data = _process_image_advanced(img)
                
                save_path = Path('./out') / category / name
                save_path.mkdir(parents=True, exist_ok=True)
                
                npz_path = save_path / 'specpack_adv.npz'
                _save_advanced_specpack(spec_data, str(npz_path))
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                continue
        
        print(f"Category '{category}': Processed {processed_count}/{len(images)} images\n")
    
    print("Advanced spectral analysis completed!")

def detect_peaks_adv(r_min=5, max_pairs=10, path='./out'):
    categories = ['synth', 'real']
    
    print("Starting advanced peak detection...")
    
    for category in categories:
        specpack_files = _scan_specpack_files(path, category)
        
        if not specpack_files:
            print(f"No specpack_adv.npz found in category: {category}")
            continue
        
        print(f"Processing category: {category}, {len(specpack_files)} specpacks")
        
        processed_count = 0
        for specpack_path, image_name in specpack_files:
            try:
                print(f"[peaks_adv] Processing: {category}/{image_name}")
                
                data = np.load(specpack_path)
                
                logpolar = data['logpolar']
                orient_energy = data['orient_energy']
                
                nms_map, nms_positions = non_maximum_suppression(
                    orient_energy,
                    neighborhood_size=(3, 3)
                )
                
                filtered_positions = filter_peaks_by_threshold(
                    nms_positions,
                    min_energy_ratio=0.0
                )
                
                peaks = _convert_to_peaks(
                    filtered_positions,
                    data['logmag'].shape,
                    r_min=r_min,
                    max_pairs=max_pairs
                )
                
                output_dir = Path(specpack_path).parent
                peaks_path = output_dir / 'peaks.json'
                save_peaks_json(str(peaks_path), peaks)
                
                processed_count += 1
            except Exception as e:
                print(f"  Error processing {image_name}: {str(e)}")
                continue
        
        print(f"Category '{category}': Processed {processed_count}/{len(specpack_files)} specpacks\n")
    
    print("Advanced peak detection completed!")

def _scan_specpack_files(output_root, category):
    specpack_files = []
    category_dir = Path(output_root) / category
    
    for spec_path in glob.glob(str(category_dir / '*' / 'specpack_adv.npz')):
        image_name = Path(spec_path).parent.name
        specpack_files.append((spec_path, image_name))
    
    return specpack_files

def _convert_to_peaks(filtered_positions, image_shape, logpolar_shape=None, 
                     r_min=5, max_pairs=10, debug_name=""):

    if not filtered_positions:
        print(f"[DEBUG: {debug_name}] No filtered positions provided to convert to peaks")
        return []
    
    h, w = image_shape
    cx, cy = w / 2, h / 2

    max_radius = np.sqrt(cx * cx + cy * cy) 

    if logpolar_shape is None:
        logpolar_shape = image_shape 
    
    Hp, Wp = logpolar_shape

    M = Hp / np.log(max_radius)
    peaks = []
    
    for i, pos in enumerate(filtered_positions[:max_pairs * 2]):
        rho_idx = pos['rho_idx']
        theta_idx = pos['theta_idx']
        energy = pos['energy']
        

        if rho_idx >= Hp:
            rho_idx = Hp - 1

        log_r = rho_idx / M
        radius = np.exp(log_r)
        
        theta_deg = (theta_idx / Wp) * 360.0
        verify_log_r = np.log(radius)
        verify_rho_idx = M * verify_log_r
        
        if radius < r_min:
            continue
        
        if radius > max_radius * 1.2:
            print(f"  WARNING: radius {radius:.2f} > max_radius {max_radius:.1f}")
        
        peak = {
            "r": float(radius),
            "theta_deg": float(theta_deg),
            "strength": float(energy),
            "energy": float(energy)
        }
        
        peaks.append(peak)
    
    peaks.sort(key=lambda x: x['energy'], reverse=True)
    result = peaks[:max_pairs * 2]    
    return result