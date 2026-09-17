from typing import List
import numpy as np
from matplotlib import colors
from PIL import Image
from sklearn.cluster import KMeans

class ColorExtractor:
    def __init__(self, n_colors=3):
        self.n_colors = n_colors
        self.xkcd_colors = {
            name.replace("xkcd:", ""): tuple(int(c * 255) for c in colors.to_rgb(val))
            for name, val in colors.XKCD_COLORS.items()
        }

    def get_dominant_colors(self, pil_img):
        img = pil_img.convert("RGB").resize((50, 50))  # 降低计算量
        pixels = np.array(img).reshape(-1, 3)

        # 使用 KMeans 聚类
        kmeans = KMeans(n_clusters=self.n_colors, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(int)

        # 按聚类中心出现频率排序
        counts = np.bincount(labels)
        sorted_idxs = np.argsort(-counts)

        return [tuple(centers[i]) for i in sorted_idxs]

    def rgb_to_xkcd_name(self, rgb):
        def dist(c1, c2):
            return sum((a - b) ** 2 for a, b in zip(c1, c2))
        min_dist = float("inf")
        closest_name = None
        for name, xkcd_rgb in self.xkcd_colors.items():
            d = dist(rgb, xkcd_rgb)
            if d < min_dist:
                min_dist = d
                closest_name = name
        return closest_name

    def get_top_color_names(self, pil_img, topk=3):
        dominant_rgbs = self.get_dominant_colors(pil_img)
        return [self.rgb_to_xkcd_name(rgb) for rgb in dominant_rgbs[:topk]]

    # 新增批量接口
    def get_top_color_names_batch(self, pil_imgs: List[Image.Image], topk=3):
        """
        支持批量输入PIL图像，返回对应的颜色名称列表。
        返回形如：[['red', 'blue', 'green'], ['black', 'white', 'gray'], ...]
        """
        results = []
        for pil_img in pil_imgs:
            try:
                names = self.get_top_color_names(pil_img, topk=topk)
            except Exception as e:
                names = []
                print(f"ColorExtractor batch error: {e}")
            results.append(names)
        return results
