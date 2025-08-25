import torch
import torch.nn as nn
import shap
import numpy as np

class MultiLabelModel(nn.Module):
    """ để chỉ trả về logitWrapper/prob for 1 class"""
    def __init__(self, base_model, class_idx, apply_sigmoid=False):
        super().__init__()
        self.base_model = base_model
        self.class_idx = class_idx
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        logits = self.base_model(x)            
        if self.apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits[:, self.class_idx].unsqueeze(-1)


def kshap_multi_band_importance_vit(
    model,
    image_tensor,          
    target_class,
    background_dict_path,
    apply_sigmoid=True,
    device=None,
    max_background=30,
    nsamples=150
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)   # (1, C, H, W)

    image_tensor = image_tensor.to(device)

    # Load background (dict: class_id -> (N, C, H, W))
    background_dict = torch.load(background_dict_path, map_location=device)
    if target_class not in background_dict:
        raise ValueError(f"No background found for class {target_class}")

    background_full = background_dict[target_class].to(device)  # (N, C, H, W)

    # Giới hạn background
    if background_full.shape[0] > max_background:
        idx = torch.randperm(background_full.shape[0], device=device)[:max_background]
        background_imgs = background_full.index_select(0, idx)
    else:
        background_imgs = background_full

    # baseline = trung bình background theo band
    baseline = background_imgs.mean(dim=0, keepdim=True)  # (1, C, H, W)
    num_bands = image_tensor.shape[1]

    # Wrapper chỉ cho target class
    class_model = MultiLabelModel(model, target_class, apply_sigmoid=apply_sigmoid).to(device).eval()

    # Hàm predict với masking theo band
    def predict_with_mask(mask):
        imgs = []
        for m in mask:
            img = image_tensor.clone()
            for j in range(num_bands):
                if m[j] == 0:
                    img[:, j, :, :] = baseline[:, j, :, :]   # luôn giữ batch dim
            imgs.append(img)
        batch = torch.cat(imgs, dim=0).to(device)  # (n_samples, C, H, W)
        with torch.no_grad():
            preds = class_model(batch)  # (n_samples, 1)
        return preds[:, 0].cpu().numpy()

    # KernelExplainer với baseline all-zero mask
    explainer = shap.KernelExplainer(
        predict_with_mask,
        np.zeros((1, num_bands))
    )

    # Tính SHAP cho input "all-one mask" (tức giữ nguyên ảnh)
    shap_values = explainer.shap_values(
        np.ones((1, num_bands)),
        nsamples=nsamples
    )

    shap_array = np.array(shap_values).squeeze()   # (num_bands,)
    return shap_array
