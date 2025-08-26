import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import rasterio
import albumentations as A

from lime_utils import lime_multi_band_importance 
from shap_utils import shap_multi_band_importance_efficientnet
from kshap_utils import kshap_multi_band_importance_vit
from xai_utils import combine_xai_results
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.serialization import add_safe_globals
from timm.models.vision_transformer import VisionTransformer


# ----------------- Config / Labels -----------------
class_names = [
    'Arable land', 'Mixed forest', 'Coniferous forest', 'Transitional woodland, shrub',
    'Broad-leaved forest', 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Complex cultivation patterns', 'Pastures', 'Urban fabric', 'Inland waters',
    'Marine waters', 'Agro-forestry areas', 'Permanent crops', 'Inland wetlands',
    'Moors, heathland and sclerophyllous vegetation', 'Natural grassland and sparsely vegetated areas',
    'Industrial or commercial units', 'Coastal wetlands', 'Beaches, dunes, sands'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Model loaders -----------------
@st.cache_resource
def load_efficientnet_model(path="t-capstone-process-bigearth-nhan-efficientnet-v3.pth"):
    # assume file was saved by torch.save(model, ...)
    model = torch.load(path, map_location=DEVICE, weights_only=False)
    model.eval()
    return model

@st.cache_resource
def load_vit_model(path="my_pytorch_model_bigearth_1.pth"):
    # allow VisionTransformer global if model pickle needs it
    add_safe_globals({'VisionTransformer': VisionTransformer})
    model = torch.load(path, map_location=DEVICE, weights_only=False)
    model.eval()
    return model

# ----------------- Helpers -----------------
def find_last_conv_layer(model):
    """
    Tìm layer Conv2d cuối cùng để dùng làm target layer cho Grad-CAM.
    Trả về module hoặc None.
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def preprocess_image(uploaded_file, file_type, model_type):
    """
    Trả về:
      tensor_img: torch.FloatTensor shape (1, C, H, W) trên DEVICE
      np_img_for_display: numpy array dùng để hiển thị RGB (C,H,W hoặc H,W,C)
    """
    # Load input
    if file_type == 'npy':
        np_img = np.load(uploaded_file)  # (C, H, W)
    elif file_type == 'tif':
        with rasterio.open(uploaded_file) as src:
            np_img = src.read()  # (bands, H, W)
    else:
        raise ValueError("Chỉ hỗ trợ .npy hoặc .tif")

    if np_img.shape[0] != 12:
        raise ValueError("Ảnh phải có đúng 12 bands (12, H, W).")

    # Normalize
    np_img = np_img.astype(np.float32) / 10000.0

    # If model_type == "ViT": resize to 224x224 (H,W)
    if model_type == "ViT":
        # albumentations expects HWC
        hwc = np.transpose(np_img, (1, 2, 0))  # (H, W, C)
        transform = A.Compose([A.Resize(224, 224)])
        hwc = transform(image=hwc)["image"]
        chw = np.transpose(hwc, (2, 0, 1))  # (C, H, W)
        np_img_proc = chw
    else:
        np_img_proc = np_img  # (C, H, W)

    tensor_img = torch.tensor(np_img_proc, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    return tensor_img, np_img  # return original np_img for display (bands, H, W)

def safe_predict(model, image_tensor, apply_sigmoid: bool):
    model.eval()
    with torch.no_grad():
        out = model(image_tensor.to(DEVICE))
        out = out.cpu()
        if apply_sigmoid:
            probs = torch.sigmoid(out).squeeze().numpy()
        else:
            probs = out.squeeze().numpy()
    return probs

# Grad-CAM per-band visualization
def gradcam_single_band(model, image_tensor, target_class, target_layer, band_index):
    """
    - image_tensor: (1, C=12, H, W) on DEVICE
    - target_layer: module to use with GradCAM (should be conv)
    - band_index: 0-based index of band to visualize
    """
    # create cam
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(int(target_class))]
    # compute cam on full input
    grayscale_cam = cam(input_tensor=image_tensor.to(DEVICE), targets=targets)  # (N, H, W)
    cam_map = grayscale_cam[0]  # (H, W)

    # take the chosen band (on CPU numpy) from image_tensor (use original scale)
    band = image_tensor.detach().cpu().numpy()[0, band_index]  # (H, W)
    # normalize band to 0-1 for visualization background
    bmin, bmax = band.min(), band.max()
    if bmax - bmin > 1e-8:
        band_norm = (band - bmin) / (bmax - bmin)
    else:
        band_norm = np.zeros_like(band)

    # create RGB image where R=band_norm, G=0, B=0
    rgb_background = np.zeros((band_norm.shape[0], band_norm.shape[1], 3), dtype=np.float32)
    rgb_background[:, :, 0] = band_norm

    visualization = show_cam_on_image(rgb_background, cam_map, use_rgb=True)
    return visualization  # uint8 RGB image

# ----------------- Streamlit UI -----------------
st.set_page_config(layout="wide", page_title="BigEarthNet XAI (LIME + Grad-CAM)")
st.title("🛰️ BigEarthNet — Classification + LIME + Grad-CAM")

# Sidebar: model choice
st.sidebar.header("Setting")
model_choice = st.sidebar.radio("Sellect model:", ("EfficientNet", "ViT"))

# Load chosen model
try:
    if model_choice == "EfficientNet":
        model = load_efficientnet_model()
    else:
        model = load_vit_model()
    st.sidebar.success(f"{model_choice} loaded")
except Exception as e:
    st.sidebar.error(f"Unable to load model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload satellite image (.npy or .tif)", type=["npy", "tif", "tiff"])

if uploaded_file is None:
    st.info("Please upload .npy (12 bands) or .tif (12 bands) file.")
    st.stop()

# Process upload and inference
try:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "tiff":
        file_type = "tif"

    image_tensor, np_img_orig = preprocess_image(uploaded_file, file_type, model_choice)
    st.success("Image processed")

    # Show RGB using original np_img_orig (bands, H, W) -> band indices 4-3-2 = idx [3,2,1]
    try:
        rgb = np_img_orig[[3, 2, 1], :, :]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb_vis = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        st.image(rgb_vis, caption="RGB emulation (band 4-3-2)",  width=500)
    except Exception:
        st.write("Unable to display RGB preview")

    # Predict
    apply_sigmoid = (model_choice == "EfficientNet")
    probs = safe_predict(model, image_tensor, apply_sigmoid)
    # Ensure probs is 1D length = num_classes
    probs = np.array(probs).reshape(-1)

    # Display predictions
    st.subheader("Probabilities")
    threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
    cols = st.columns(2)
    with cols[0]:
        for i, p in enumerate(probs):
            mark = "✅" if p > threshold else ""
            st.write(f"{i:02d} {class_names[i]}: {p:.3f} {mark}")

    # # XAI: LIME / Kernel SHAP band importance
    positive_indices = [i for i, p in enumerate(probs) if p > threshold]
    # -----------------------------
    st.subheader("📊 XAI Band Importance (Combined)")

    if len(positive_indices) == 0:
        st.info("No class exceeds the threshold for calculating XAI.")
    else:
        chosen_for_xai = st.selectbox(
            "Select class to calculate band-importance",
            positive_indices,
            format_func=lambda x: f"{x} - {class_names[x]} ({probs[x]:.3f})"
        )

        if st.button("Calculate XAI for class " + str(chosen_for_xai)):
            try:
                with st.spinner("Calculating XAI..."):
                    if model_choice == "EfficientNet":
                        # ---------------- LIME ----------------
                        lime_vals = lime_multi_band_importance(
                            model=model,
                            image_tensor=image_tensor.squeeze(0),
                            target_class=chosen_for_xai,
                            num_samples=1000,
                            method='zero',
                            apply_sigmoid=apply_sigmoid
                        )

                        # ---------------- SHAP ----------------
                        shap_vals = shap_multi_band_importance_efficientnet(
                            model=model,
                            image_tensor=image_tensor,
                            target_class=chosen_for_xai,
                            background_dict_path= "backgrounds_dict_eff_shap.pt",
                            apply_sigmoid=apply_sigmoid,
                            device=DEVICE,
                            model_type=model_choice,
                            max_samples= 50
                        )

                        combined_vals = combine_xai_results(lime_vals, shap_vals)
                        method_label = "LIME + SHAP (avg)"

                    else:  # ViT
                        # ---------------- Kernel SHAP ----------------
                        kshap_vals = kshap_multi_band_importance_vit(
                            model=model,
                            image_tensor=image_tensor,
                            target_class=chosen_for_xai,
                            nsamples= 150,
                            apply_sigmoid=apply_sigmoid,
                            background_dict_path= "backgrounds_dict_vit_shap.pt",
                            device=DEVICE
                        )

                        # ---------------- SHAP ----------------
                        shap_vals = shap_multi_band_importance_efficientnet(
                            model=model,
                            image_tensor=image_tensor,
                            target_class=chosen_for_xai,
                            background_dict_path = "backgrounds_dict_vit_shap.pt",
                            apply_sigmoid=apply_sigmoid,
                            device=DEVICE,
                            model_type=model_choice,
                            max_samples= 20
                        )

                        combined_vals = combine_xai_results(kshap_vals, shap_vals)
                        method_label = "Kernel SHAP + SHAP (avg)"

                    st.session_state['xai_result'] = {
                        'class_id': chosen_for_xai,
                        'importances': combined_vals,
                        'method': method_label
                    }

            except Exception as e:
                st.error(f"Error in calculating XAI: {e}")

        # Show XAI result
        if 'xai_result' in st.session_state:
            xai_res = st.session_state['xai_result']
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(np.arange(1, len(xai_res['importances']) + 1), xai_res['importances'])
            ax.set_xlabel("Band")
            ax.set_ylabel("Importance (avg normalized)")
            ax.set_title(f"{xai_res['method']} for class {xai_res['class_id']} - {class_names[xai_res['class_id']]}")
            st.pyplot(fig)

    # Grad-CAM block
    st.subheader("🔥 Grad-CAM per-band (pixel-level)")
    if len(positive_indices) == 0:
        st.info("No class exceeds the threshold for calculating XAI.")
    else:
        chosen_for_grad = st.selectbox("Select class to calculate Grad-CAM", positive_indices,
                                       format_func=lambda x: f"{x} - {class_names[x]} ({probs[x]:.3f})",
                                       key="grad_class")
        chosen_band = st.slider("Select band to visualize (1..12)", 1, 12, 1)
        if st.button("Calculate Grad-CAM for class & band"):
            target_layer = find_last_conv_layer(model)
            if target_layer is None:
                st.error("No Conv2d layer found in the model to do Grad-CAM.")
            else:
                with st.spinner("Calculating Grad-CAM..."):
                    vis = gradcam_single_band(
                        model=model,
                        image_tensor=image_tensor,
                        target_class=chosen_for_grad,
                        target_layer=target_layer,
                        band_index=chosen_band - 1
                    )
                    st.session_state['gradcam_result'] = {
                        'class_id': chosen_for_grad,
                        'band_id': chosen_band,
                        'image': vis
                    }

        # Re-display if Grad-CAM is present
        if 'gradcam_result' in st.session_state:
            g = st.session_state['gradcam_result']

            # 1. RGB base (bands 4-3-2)
            try:
                rgb_orig = np_img_orig[[3, 2, 1], :, :]
                rgb_orig = np.transpose(rgb_orig, (1, 2, 0))
                rgb_orig_vis = (rgb_orig - rgb_orig.min()) / (rgb_orig.max() - rgb_orig.min() + 1e-8)
            except Exception:
                rgb_orig_vis = None

            # 2. Selected band (grayscale)
            band_idx = g['band_id'] - 1
            band_img = np_img_orig[band_idx, :, :]
            band_img_norm = (band_img - band_img.min()) / (band_img.max() - band_img.min() + 1e-8)

            # 3. Grad-CAM heatmap
            gradcam_img = g['image']

            col1, col2, col3 = st.columns(3)

            with col1:
                if rgb_orig_vis is not None:
                    st.image(rgb_orig_vis, caption="Original RGB (bands 4-3-2)", use_container_width=True)
                else:
                    st.write("Unable to display original RGB")

            with col2:
                # Use clamp and gray colormap
                st.image(band_img_norm, caption=f"Single-band grayscale (band {g['band_id']})", 
                        use_container_width=True, channels="GRAY")

            with col3:
                st.image(
                    gradcam_img,
                    caption=f"Grad-CAM class {g['class_id']} ({class_names[g['class_id']]}), band {g['band_id']}",
                    use_container_width=True
                )





except Exception as e:
    st.error(f"Error while processing image: {e}")




