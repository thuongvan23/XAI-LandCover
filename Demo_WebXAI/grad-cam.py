import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def gradcam_single_band(model, image_tensor, target_class, target_layer, band_index):
    """
    Tính Grad-CAM cho 1 band trong ảnh 12 bands.
    Khi visualize: giữ band này làm kênh R, 2 kênh còn lại = 0.
    
    Args:
        model: mô hình torch đã huấn luyện
        image_tensor: tensor (1, 12, H, W) trên cùng DEVICE với model
        target_class: int
        target_layer: layer để hook Grad-CAM
        band_index: int (0-11)
    Returns:
        visualization: ảnh RGB overlay heatmap
    """

    # Tạo Grad-CAM object (phiên bản mới không dùng use_cuda)
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(target_class)]

    # Tính Grad-CAM cho ảnh full 12 channels
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # (H, W)

    # Chuẩn hóa dữ liệu của band được chọn
    band_data = image_tensor[0, band_index, :, :].detach().cpu().numpy()
    band_norm = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)

    # Tạo ảnh RGB: band chọn → kênh R, G=0, B=0
    rgb_image = np.zeros((band_norm.shape[0], band_norm.shape[1], 3), dtype=np.float32)
    rgb_image[:, :, 0] = band_norm  # R = band được chọn

    # Overlay heatmap lên ảnh RGB
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return visualization
