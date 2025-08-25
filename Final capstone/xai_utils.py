import numpy as np

def normalize_importances(values):
    """Chuẩn hóa về [-1, 1]"""
    values = np.array(values, dtype=np.float32)
    if values.max() == values.min():  # tránh chia cho 0
        return np.zeros_like(values)
    normed = 2 * (values - values.min()) / (values.max() - values.min()) - 1
    return normed

def combine_xai_results(method1_vals, method2_vals):
    """
    Nhận 2 vector importances (ví dụ từ LIME và SHAP).
    Chuẩn hóa riêng từng vector về [-1,1],
    rồi lấy trung bình cộng.
    """
    norm1 = normalize_importances(method1_vals)
    norm2 = normalize_importances(method2_vals)
    combined = (norm1 + norm2) / 2.0
    return combined
