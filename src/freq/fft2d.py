import numpy as np
import cv2
import os, glob, cv2

# import src.io.artifact
# import src.io.loader
from src.freq.window import apply_window
from src.io.loader import DataLoader

def load_image(category):
    SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    image_list = []
    image_dir = os.path.join('./data', category)

    for ext in SUPPORTED_EXTS:
        image_paths = glob.glob(os.path.join(image_dir, f'*{ext}'))
        for img_path in image_paths:
            name, ext = os.path.splitext(os.path.basename(img_path))
            if name.endswith('_gt'):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                image_list.append((name, img))
    return image_list

def save_specpack_npz(path, mag, log_magnitude, phase):
    np.savez(path, mag=mag, log_magnitude=log_magnitude, phase=phase)
    return

def load_specpack_npz(path):
    data = np.load(path)
    mag = data['mag']
    log_magnitude = data['log_magnitude']
    phase = data['phase']
    return mag, phase, log_magnitude

def fft2d(loader: DataLoader = None):
    categories = ('real', 'synth')
    if loader is not None:
        print("Starting FFT processing with DataLoader...")
        for category in categories:
            print(f"Processing category: {category}")
            loader.scan_images(category)
            for x_name, batch_x, _, _ in loader:
                for i in range(batch_x.shape[0]):
                    img = batch_x[i]
                    name = x_name[i]
                    print(f"[spec] Processing image: {name}")
                    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32)
                    img_gray = img_yuv[:, :, 0]
                    # img_gray = apply_window(img_gray)
                    f = np.fft.fft2(img_gray)
                    fshift = np.fft.fftshift(f)
                    mag = np.abs(fshift)
                    phase = np.angle(fshift)
                    log_mag = np.log(mag + 1e-8)
                    save_path = os.path.join('./out', category, name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_specpack_npz(path = os.path.join(save_path, 'specpack.npz'),
                                mag=mag,
                                log_magnitude=log_mag,
                                phase=phase)
                    np.savez(os.path.join(save_path, 'yuv.npz'), y=img_yuv[:, :, 0], u=img_yuv[:, :, 1], v=img_yuv[:, :, 2])
    else:
        print("Starting FFT processing...")
        for category in categories:
            images = load_image(category)
            print(f"Processing category: {category}, Number of images: {len(images)}")
            for name, img in images:
                print(f"[spec] Processing image: {name}")
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32)
                img = img_yuv[:, :, 0]
                # img = apply_window(img)
                f = np.fft.fft2(img)
                fshift = np.fft.fftshift(f)
                mag = np.abs(fshift)
                phase = np.angle(fshift)
                log_mag = np.log(mag + 1e-8)
                save_path = os.path.join('./out', category, name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_specpack_npz(path = os.path.join(save_path, 'specpack.npz'),
                            mag=mag,
                            log_magnitude=log_mag,
                            phase=phase)
                np.savez(os.path.join(save_path, 'yuv.npz'), y=img_yuv[:, :, 0], u=img_yuv[:, :, 1], v=img_yuv[:, :, 2])
