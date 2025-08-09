import torch
import shap
import numpy as np

class MultiLabelModel(torch.nn.Module):
    def __init__(self, base_model, class_idx):
        super().__init__()
        self.base_model = base_model
        self.class_idx = class_idx

    def forward(self, x):
        logits = self.base_model(x)
        return logits[:, self.class_idx].unsqueeze(-1)

def shap_multi_band_importance_efficientnet(
    model,
    image_tensor,
    target_class,
    background_dict_path,
    apply_sigmoid=True,
    device=None
):
    """
    Tính SHAP band importance cho 1 class cụ thể (multi-label).
    
    Args:
        model: torch model
        image_tensor: (1, C, H, W) tensor
        target_class: int
        background_dict_path: str - path tới file backgrounds_dict.pt
        apply_sigmoid: bool - có dùng sigmoid không (EfficientNet multi-label thường dùng)
        device: torch.device
    
    Return:
        np.array: (C,) - tổng SHAP value mỗi band
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    image_tensor = image_tensor.to(device)

    # Load background dict
    background_dict = torch.load(background_dict_path, map_location=device)
    if target_class not in background_dict:
        raise ValueError(f"No background found for class {target_class}")
    background_imgs = background_dict[target_class].to(device)  # (N, C, H, W)

    # Wrapper model cho class này
    class_model = MultiLabelModel(model, target_class).to(device)

    # Tạo explainer
    explainer = shap.GradientExplainer(class_model, background_imgs)

    # Tính SHAP values
    shap_values = explainer.shap_values(image_tensor)  # list with 1 tensor
    shap_array = np.squeeze(shap_values)  # (C, H, W)

    if shap_array.ndim != 3:
        raise ValueError(f"SHAP array shape is invalid: {shap_array.shape}")

    # Tổng theo từng band
    band_importance = shap_array.reshape(shap_array.shape[0], -1).sum(axis=1)

    return band_importance
