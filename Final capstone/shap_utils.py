import torch
import shap
import numpy as np
import torch.nn.functional as F


class MultiLabelModel(torch.nn.Module):
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


def _maybe_downscale_for_vit(tensor, target_hw=(224, 224)):
    """
    Resize tensor to target_hw nếu cần.
    """
    n, c, h, w = tensor.shape
    th, tw = target_hw
    if (h, w) == (th, tw):
        return tensor
    return F.interpolate(tensor, size=(th, tw), mode="bilinear", align_corners=False)


def _sample_patches(tensor, num_patches=64, patch_size=16):
    """
    Giảm số patch đầu vào cho ViT bằng cách chỉ giữ lại num_patches patch ngẫu nhiên.
    tensor: (N, C, H, W)
    """
    n, c, h, w = tensor.shape
    patches_h = h // patch_size
    patches_w = w // patch_size
    total_patches = patches_h * patches_w

    if num_patches >= total_patches:
        return tensor  # giữ nguyên nếu số patch yêu cầu >= tổng patch

    # chọn ngẫu nhiên patch index
    idx = torch.randperm(total_patches)[:num_patches]
    mask = torch.zeros(total_patches, dtype=torch.bool, device=tensor.device)
    mask[idx] = True
    mask = mask.view(patches_h, patches_w)

    # set patch không chọn về 0 để giảm tải
    tensor_masked = tensor.clone()
    for i in range(patches_h):
        for j in range(patches_w):
            if not mask[i, j]:
                tensor_masked[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 0
    return tensor_masked


def shap_multi_band_importance_efficientnet(
    model,
    image_tensor,
    target_class,
    background_dict_path,
    apply_sigmoid=True,
    device=None,
    model_type="EfficientNet",
    max_samples=20,
    nsamples=30,          #  thêm nsamples để giảm số lần sampling
    vit_patch_subset=64    #  số patch giữ lại khi tính SHAP cho ViT
):
    """
    Tính SHAP band-importance cho 1 class.
    - Với EfficientNet: giữ nguyên input.
    - Với ViT: downscale + chỉ lấy subset patch để tránh OOM.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    # Load background (dict: class_id -> (N, C, H, W))
    background_dict = torch.load(background_dict_path, map_location=device)
    if target_class not in background_dict:
        raise ValueError(f"No background found for class {target_class}")

    background_full = background_dict[target_class].to(device)  # (N, C, H, W)

    # Giới hạn background
    if model_type == "ViT":
        max_samples = int(min(max_samples, 20))  # chặt chẽ hơn cho ViT
    if background_full.shape[0] > max_samples:
        idx = torch.randperm(background_full.shape[0], device=device)[:max_samples]
        background_imgs = background_full.index_select(0, idx)
    else:
        background_imgs = background_full

    # Nếu là ViT → downscale + patch sampling
    use_autocast = (device.type == "cuda")
    if model_type == "ViT":
        background_imgs = _maybe_downscale_for_vit(background_imgs, target_hw=(224, 224))
        image_tensor = _maybe_downscale_for_vit(image_tensor, target_hw=(224, 224))

        background_imgs = _sample_patches(background_imgs, num_patches=vit_patch_subset)
        image_tensor = _sample_patches(image_tensor, num_patches=vit_patch_subset)

    # Wrapper model cho class
    class_model = MultiLabelModel(model, target_class, apply_sigmoid=apply_sigmoid).to(device).eval()

    # GradientExplainer cho PyTorch models
    explainer = shap.GradientExplainer(class_model, background_imgs)

    # Tính SHAP
    if use_autocast:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            shap_values = explainer.shap_values(image_tensor, nsamples=nsamples)
    else:
        shap_values = explainer.shap_values(image_tensor, nsamples=nsamples)

    # Chuyển về numpy
    shap_array = np.squeeze(shap_values)  # (C, H, W)
    if shap_array.ndim != 3:
        raise ValueError(f"SHAP array shape is invalid: {shap_array.shape}")

    # Tổng theo band
    band_importance = shap_array.reshape(shap_array.shape[0], -1).sum(axis=1)

    # Dọn VRAM/Cache
    if device.type == "cuda":
        del shap_values
        torch.cuda.empty_cache()

    return band_importance

