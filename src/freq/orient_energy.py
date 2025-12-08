import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from typing import List, Dict, Tuple

def create_fan_window(rho_size: int, theta_size: int, rho_center: float, 
                      theta_center: float, rho_width: int = 3, theta_width: float = 10) -> np.ndarray:
    rho_grid, theta_grid = np.mgrid[0:rho_size, 0:theta_size]
    
    theta_grid_deg = theta_grid * 360.0 / theta_size
    
    theta_diff = np.abs(theta_grid_deg - theta_center)
    theta_diff = np.minimum(theta_diff, 360 - theta_diff)
    
    rho_diff = np.abs(rho_grid - rho_center)
    
    rho_mask = rho_diff <= rho_width
    theta_mask = theta_diff <= theta_width
    window_mask = rho_mask & theta_mask
    
    rho_dist = rho_diff / max(rho_width, 1e-8)
    theta_dist = theta_diff / max(theta_width, 1e-8)
    dist_sq = rho_dist ** 2 + theta_dist ** 2
    gaussian_weights = np.exp(-dist_sq / 2)
    
    window = window_mask.astype(np.float32) * gaussian_weights
    
    return window


def _compute_orientation_energy(logpolar_img: np.ndarray, 
                               rho_bin_step: int = 1, 
                               theta_bin_step: int = 5,
                               rho_width: int = 3, 
                               theta_width: float = 15) -> Tuple[np.ndarray, List[Dict]]:
    
    rho_size, theta_size = logpolar_img.shape
    
    rho_centers = np.arange(rho_width, rho_size - rho_width, rho_bin_step)
    theta_indices = np.arange(0, theta_size, theta_bin_step)
    
    energy_map = np.zeros_like(logpolar_img)
    positions = []
    
    theta_to_deg_factor = 360.0 / theta_size
    
    for rho_center in rho_centers:
        rho_grid = np.arange(rho_size)
        rho_diff = np.abs(rho_grid - rho_center)
        rho_mask_full = rho_diff <= rho_width
        
        rho_dist = rho_diff / max(rho_width, 1e-8)
        rho_gaussian = np.exp(-rho_dist ** 2 / 2)
        
        for theta_idx in theta_indices:
            theta_center_deg = theta_idx * theta_to_deg_factor
            
            theta_grid_deg = np.arange(theta_size) * theta_to_deg_factor
            theta_diff = np.abs(theta_grid_deg - theta_center_deg)
            theta_diff = np.minimum(theta_diff, 360 - theta_diff)
            
            theta_mask = theta_diff <= theta_width
            
            theta_dist = theta_diff / max(theta_width, 1e-8)
            theta_gaussian = np.exp(-theta_dist ** 2 / 2)
            
            rho_mask_2d = rho_mask_full[:, np.newaxis]
            theta_mask_2d = theta_mask[np.newaxis, :]
            window_mask = rho_mask_2d & theta_mask_2d
            
            rho_gauss_2d = rho_gaussian[:, np.newaxis]
            theta_gauss_2d = theta_gaussian[np.newaxis, :]
            gaussian_weights = rho_gauss_2d * theta_gauss_2d
            
            window = window_mask.astype(np.float32) * gaussian_weights
            
            energy_sum = np.sum(logpolar_img * window)
            
            energy_map[rho_center, theta_idx] = energy_sum
            
            positions.append({
                'rho_idx': int(rho_center),
                'theta_idx': int(theta_idx),
                'theta_deg': float(theta_center_deg),
                'energy': float(energy_sum),
                'window_area': float(np.sum(window))
            })
    
    return energy_map, positions


def _compute_orientation_energy_fast(logpolar_img: np.ndarray,
                                     rho_width: int = 3,
                                     theta_width: float = 15) -> Tuple[np.ndarray, List[Dict]]:
    rho_size, theta_size = logpolar_img.shape
    
    kernel_rho = 2 * rho_width + 1
    kernel_theta = int(theta_width * theta_size / 180.0)
    
    y, x = np.mgrid[-rho_width:rho_width+1, -kernel_theta//2:kernel_theta//2+1]
    kernel = np.exp(-(x**2/(2*(kernel_theta/4)**2) + y**2/(2*rho_width**2)))
    kernel = kernel / kernel.sum()
    
    energy_map = fftconvolve(logpolar_img, kernel, mode='same')
    
    positions = []
    rho_centers = np.arange(rho_width, rho_size - rho_width)
    theta_indices = np.arange(0, theta_size)
    
    for rho_center in rho_centers:
        for theta_idx in theta_indices:
            energy = energy_map[rho_center, theta_idx]
            if energy > 0:
                theta_center_deg = theta_idx * 360.0 / theta_size
                positions.append({
                    'rho_idx': int(rho_center),
                    'theta_idx': int(theta_idx),
                    'theta_deg': float(theta_center_deg),
                    'energy': float(energy),
                    'window_area': float(np.sum(kernel)) 
                })
    
    return energy_map, positions


def compute_orientation_energy(logpolar_img: np.ndarray,
                                         rho_bin_step: int = 1,
                                         theta_bin_step: int = 5,
                                         rho_width: int = 3,
                                         theta_width: float = 15,
                                         fast_mode: bool = True) -> Tuple[np.ndarray, List[Dict]]:

    if fast_mode and logpolar_img.shape[0] > 50 and logpolar_img.shape[1] > 50:
        #print("Using fast orientation energy computation")
        return _compute_orientation_energy_fast(logpolar_img, rho_width, theta_width)
    else:
        return _compute_orientation_energy(logpolar_img, rho_bin_step, 
                                          theta_bin_step, rho_width, theta_width)


def non_maximum_suppression(energy_map: np.ndarray, 
                            neighborhood_size: Tuple[int, int] = (5, 15)) -> Tuple[np.ndarray, List[Dict]]:
    rho_neigh, theta_neigh = neighborhood_size
    
    data_max = maximum_filter(energy_map, size=(rho_neigh, theta_neigh))
    
    maxima = (energy_map == data_max) & (energy_map > 0)
    
    peak_indices = np.where(maxima)
    peak_energies = energy_map[peak_indices]
    
    sorted_indices = np.argsort(peak_energies)[::-1]
    
    peak_positions = []
    for idx in sorted_indices:
        rho_idx = peak_indices[0][idx]
        theta_idx = peak_indices[1][idx]
        energy = peak_energies[idx]
        
        peak_positions.append({
            'rho_idx': int(rho_idx),
            'theta_idx': int(theta_idx),
            'energy': float(energy)
        })
    
    nms_map = np.zeros_like(energy_map)
    nms_map[maxima] = energy_map[maxima]
    
    return nms_map, peak_positions

def filter_peaks_by_threshold(peak_positions: List[Dict], 
                              min_energy_ratio: float = 0.1) -> List[Dict]:
    if not peak_positions:
        return []
    
    max_energy = max(p['energy'] for p in peak_positions)
    threshold = max_energy * min_energy_ratio
    
    return [p for p in peak_positions if p['energy'] >= threshold]

def create_orientation_visualization(logpolar_img: np.ndarray, 
                                     energy_map: np.ndarray, 
                                     nms_map: np.ndarray,
                                     peak_positions: List[Dict], 
                                     output_path: str = None):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    im1 = axes[0, 0].imshow(logpolar_img, cmap='hot', aspect='auto')
    axes[0, 0].set_title('Log-Polar Image')
    axes[0, 0].set_xlabel('Angle (bins)')
    axes[0, 0].set_ylabel('Log Radius (bins)')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    im2 = axes[0, 1].imshow(energy_map, cmap='hot', aspect='auto')
    axes[0, 1].set_title('Orientation Energy Map')
    axes[0, 1].set_xlabel('Angle (bins)')
    axes[0, 1].set_ylabel('Log Radius (bins)')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    im3 = axes[1, 0].imshow(nms_map, cmap='hot', aspect='auto')
    axes[1, 0].set_title('NMS Filtered Peaks')
    axes[1, 0].set_xlabel('Angle (bins)')
    axes[1, 0].set_ylabel('Log Radius (bins)')
    
    for peak in peak_positions[:10]:
        axes[1, 0].plot(peak['theta_idx'], peak['rho_idx'], 'wo', markersize=8)
    
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    if peak_positions:
        energies = [p['energy'] for p in peak_positions]
        angles = [p['theta_idx'] * 360.0 / logpolar_img.shape[1] for p in peak_positions]
        radii = [p['rho_idx'] for p in peak_positions]
        
        sc = axes[1, 1].scatter(angles, radii, c=energies, cmap='hot',
                                s=50, alpha=0.7, edgecolors='k')
        axes[1, 1].set_title('Peak Distribution (Angle vs Log Radius)')
        axes[1, 1].set_xlabel('Angle (degrees)')
        axes[1, 1].set_ylabel('Log Radius (bins)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.colorbar(sc, ax=axes[1, 1], shrink=0.8)
    else:
        axes[1, 1].text(0.5, 0.5, 'No peaks detected',
                        ha='center', va='center')
        axes[1, 1].set_title('Peak Distribution')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Orientation energy visualization saved to: {output_path}")
    
    return fig