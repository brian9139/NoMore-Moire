import glob, os, re, cv2
import numpy as np

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

class DataLoader:
    def __init__(self, root, batch_size, color_mode, category):
        self.root = root
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.category = category
        self.pairs = self.scan_images(category)

    def scan_images(self, category):
        files = []
        for ext in SUPPORTED_EXTS:
            files += glob.glob(os.path.join(self.root, category, f"*{ext}"), recursive=True)

        def extract_number(path):
            base = os.path.basename(path)
            m = re.search(r"\d+", base)
            return int(m.group()) if m else 0

        sorted(files, key=extract_number)

        pairs = []

        for f in files:
            name, ext = os.path.splitext(f)
            if name.endswith("_gt"):
                continue

            gt = name + "_gt" + ext
            if os.path.exists(gt):
                pairs.append((f, gt))
            else:
                print(f"[Warning] GT not found for: {f}")

        return pairs

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if self.color_mode == 'gray':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.astype(np.float32)
        return image

    def __iter__(self):
        batch_x = []
        batch_y = []

        for x_path, y_path in self.pairs:
            x_image = self.load_image(x_path)
            y_image = self.load_image(y_path)

            batch_x.append(x_image)
            batch_y.append(y_image)

            if len(batch_x) == self.batch_size:
                yield np.stack(batch_x), np.stack(batch_y)
                batch_x, batch_y = [], []

        if batch_x:
            yield np.stack(batch_x), np.stack(batch_y)

# loader = DataLoader('../../data', 4, 'gray', 'synth')

# for x, y in loader:
#     print('input batch:', x.shape)
#     print('GT batch:', y.shape)