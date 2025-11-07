import cv2
import numpy as np
import time

def box_filter_integral(img: np.ndarray, radius: int) -> np.ndarray:
    """
    Tính trung bình vùng (2r+1)x(2r+1) bằng integral image với padding replicate.
    Thời gian O(1) theo bán kính cho mỗi pixel.
    """
    if radius <= 0:
        return img.astype(np.float32)

    img = img.astype(np.float32)
    h, w = img.shape

    # Pad theo biên để cửa sổ luôn "full" ở mọi pixel
    pad = radius
    img_pad = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE
    )

    # Integral có kích thước (H_pad+1, W_pad+1)
    ii = cv2.integral(img_pad)  # float64 mặc định
    # Các lát cắt 4 góc của cửa sổ (vector hóa toàn ảnh)
    # Với ảnh gốc (h, w), sau padding: với mỗi (i,j) gốc
    # - top-left trong ảnh pad là (i, j)
    # - bottom-right là (i+2r, j+2r)
    # Trong integral cần +1 offset
    A = ii[2 * pad + 1 : 2 * pad + 1 + h, 2 * pad + 1 : 2 * pad + 1 + w]  # bottom-right
    B = ii[0 : 0 + h, 2 * pad + 1 : 2 * pad + 1 + w]  # top-right
    C = ii[2 * pad + 1 : 2 * pad + 1 + h, 0 : 0 + w]  # bottom-left
    D = ii[0 : 0 + h, 0 : 0 + w]  # top-left

    S = A - B - C + D  # tổng cửa sổ
    area = float((2 * pad + 1) * (2 * pad + 1))
    return (S / area).astype(np.float32)


def bilateral_pbf_ic_gray(
    img: np.ndarray, radius: int = 5, sigma_r: float = 0.1, num_samples: int = 8
) -> np.ndarray:
    """
    Constant-time (O(1) theo bán kính) Bilateral Filtering cho ảnh xám (PBFIC):
    - gs: dùng BOX filter O(1) (integral image)
    - gr: Gaussian theo cường độ
    - Nội suy tuyến tính giữa 2 mẫu gần nhất
    """
    assert num_samples >= 2, "num_samples phải >= 2"
    img = img.astype(np.float32) / 255.0
    H, W = img.shape
    samples = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)

    # Tính J_k và W_k cho từng mẫu k
    J_list = []
    W_list = []
    for k in samples:
        rw = np.exp(-0.5 * ((img - k) ** 2) / (sigma_r**2)).astype(np.float32)  # (H,W)
        Jk = box_filter_integral(img * rw, radius)  # (H,W)
        Wk = box_filter_integral(rw, radius)  # (H,W)
        J_list.append(Jk)
        W_list.append(Wk)

    J = np.stack(J_list, axis=0)  # (N,H,W)
    W = np.stack(W_list, axis=0)  # (N,H,W)

    # Tìm hai mẫu k1, k2 kề nhau để nội suy cho từng pixel
    # idx in [1..N-1]
    idx = np.searchsorted(samples, img, side="right")
    idx = np.clip(idx, 1, num_samples - 1)

    k1 = samples[idx - 1]
    k2 = samples[idx]
    a = (img - k1) / (k2 - k1 + 1e-8)  # hệ số nội suy (H,W)

    # Lấy đúng "lát" (N,H,W) theo idx cho từng pixel bằng take_along_axis
    idx1 = (idx - 1)[None, ...]  # (1,H,W)
    idx2 = idx[None, ...]  # (1,H,W)

    # Giá trị đã lọc (J/W) ở hai mẫu
    I1 = np.take_along_axis(J, idx1, axis=0)[0] / (
        np.take_along_axis(W, idx1, axis=0)[0] + 1e-8
    )  # (H,W)
    I2 = np.take_along_axis(J, idx2, axis=0)[0] / (
        np.take_along_axis(W, idx2, axis=0)[0] + 1e-8
    )  # (H,W)

    I_res = (1.0 - a) * I1 + a * I2
    return np.clip(I_res * 255.0, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    path = "image/anh1.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Không tìm thấy file ảnh")

    for r in [3, 5, 9, 15, 25]:
        t0 = time.perf_counter()
        _ = bilateral_pbf_ic_gray(img, radius=r, sigma_r=0.1, num_samples=8)
        t1 = time.perf_counter()
        print(f"radius={r}: {t1 - t0:.3f}s")

    # So sánh chất lượng với OpenCV bilateral
    I_ref = cv2.bilateralFilter(img, d=15, sigmaColor=25, sigmaSpace=5)
    I_pbf = bilateral_pbf_ic_gray(img, radius=5, sigma_r=25 / 255.0, num_samples=8)
    psnr = cv2.PSNR(I_ref, I_pbf)
    print(f"PSNR vs OpenCV = {psnr:.2f} dB")
    
    cv2.imshow("Original", img)
    cv2.imshow("OpenCV Bilateral", I_ref)
    cv2.imshow("PBFIC O(1)", I_pbf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
