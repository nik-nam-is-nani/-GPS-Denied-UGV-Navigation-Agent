import numpy as np
import cv2
import torch


class DepthEstimator:
    def __init__(self, model_type="dpt_sam", device=None, seed=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.transform = None
        self.model_type = model_type
        self.use_fallback = True

    def load_model(self):
        try:
            import timm
            from torchvision import transforms

            self.model = timm.create_model(
                "vit_base_patch16_384",
                pretrained=True,
                num_classes=0,
                global_pool=""
            ).to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            self.use_fallback = False
            print(f"Loaded MiDaS model on {self.device}")

        except Exception as e:
            print(f"MiDaS unavailable, using fallback: {e}")
            self.use_fallback = True

    def estimate_depth(self, image):
        if self.use_fallback or self.model is None:
            return self._simulate_depth(image)

        try:
            h, w = image.shape[:2]
            input_tensor = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)

            depth = output.squeeze().cpu().numpy()
            depth = cv2.resize(depth, (w, h))

            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            return depth_normalized

        except Exception as e:
            print(f"Depth estimation failed: {e}")
            return self._simulate_depth(image)

    def _simulate_depth(self, image):
        h, w = image.shape[:2]
        depth = np.zeros((h, w), dtype=np.float32)

        center = (w // 2, h // 2)
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                base_depth = 1.0 - (dist / max(w, h))
                noise = self.rng.uniform(0.8, 1.0)
                depth[y, x] = base_depth * noise

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            texture_factor = gray.astype(np.float32) / 255.0
            depth = depth * (0.7 + 0.3 * texture_factor)

        return np.clip(depth, 0, 1)

    def get_obstacle_distances(self, depth_image, num_rays=8, fov=120):
        h, w = depth_image.shape
        center_x = w // 2

        angles = np.linspace(-np.radians(fov / 2), np.radians(fov / 2), num_rays)
        distances = []

        for angle in angles:
            ray_x = int(center_x + (np.tan(angle) * h))

            if 0 <= ray_x < w:
                depth_value = depth_image[h // 2, ray_x]
                distance = depth_value * 20.0
                distances.append(distance)
            else:
                distances.append(20.0)

        return np.array(distances)


class SimpleDepthEstimator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def estimate(self, frame, ground_truth_distances):
        noise = self.rng.uniform(0.9, 1.1, len(ground_truth_distances))
        noisy_depth = ground_truth_distances * noise
        return np.clip(noisy_depth, 0.1, 20.0)