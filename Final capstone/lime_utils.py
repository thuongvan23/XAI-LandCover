import torch
import numpy as np
from sklearn.linear_model import Ridge

def lime_multi_band_importance(model, image_tensor, target_class, num_samples=1000, method='zero', apply_sigmoid=True):
    """
    Giải thích dự đoán theo kiểu LIME cho ảnh nhiều bands.
    
    Args:
        model: Mô hình torch đã huấn luyện.
        image_tensor: Tensor ảnh đầu vào, shape (12, H, W)
        target_class: Lớp cần giải thích (int)
        num_samples: Số mẫu mask để sinh.
        method: Cách che band ('zero' hoặc 'mean')
        apply_sigmoid: Có dùng sigmoid sau model không (EfficientNet thì có, ViT thì không)
    
    Returns:
        importances: Mảng 12 số thể hiện tầm quan trọng từng band.
    """
    model.eval()
    image = image_tensor.to(torch.float32)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        if apply_sigmoid:
            base_prob = torch.sigmoid(output)[0, target_class].item()
        else:
            base_prob = output[0, target_class].item()

    C = image.shape[0]
    H, W = image.shape[1], image.shape[2]
    masks = []
    probs = []

    for _ in range(num_samples):
        band_mask = np.random.randint(0, 2, size=C)
        masked_image = image.clone()

        for i in range(C):
            if band_mask[i] == 0:
                if method == 'zero':
                    masked_image[i] = 0.0
                elif method == 'mean':
                    masked_image[i] = image[i].mean()

        with torch.no_grad():
            output = model(masked_image.unsqueeze(0))
            if apply_sigmoid:
                prob = torch.sigmoid(output)[0, target_class].item()
            else:
                prob = output[0, target_class].item()

        masks.append(band_mask)
        probs.append(prob)

    masks = np.array(masks)
    probs = np.array(probs)

    # Ridge regression
    reg = Ridge(alpha=1.0)
    reg.fit(masks, probs)
    importances = reg.coef_

    return importances
