# core/predict.py
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from core.config import TORCH_DEVICE, IMG_SIZE
from core.utils_logging import log

def predict_image(model, router, memory_bank, img_path: str, class_names: List[str], img_size=IMG_SIZE) -> dict:
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(TORCH_DEVICE) # type: ignore
    model.eval()
    with torch.no_grad():
        stage_activations, logits = model.forward_with_cache(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    nn_pred_idx = int(np.argmax(probs))
    nn_pred_label = class_names[nn_pred_idx]
    nn_confidence = float(probs[nn_pred_idx])

    pathway_tag, router_score = router.route(stage_activations, nn_predicted_class=nn_pred_label)

    if pathway_tag and router_score >= router.threshold:
        pathway_info = memory_bank.get_pathway_info(pathway_tag)
        final_prediction = pathway_info.get("majority_class", nn_pred_label)
        final_confidence = router_score
        prediction_source = "pathway"
        log(f"✅ Pathway prediction: {final_prediction} (confidence: {final_confidence:.3f})")
    else:
        final_prediction = nn_pred_label
        final_confidence = nn_confidence
        prediction_source = "neural_network"
        log(f"🧠 Neural network prediction: {final_prediction} (confidence: {final_confidence:.3f})")

    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "source": prediction_source,
        "nn_prediction": nn_pred_label,
        "nn_confidence": nn_confidence,
        "pathway_tag": pathway_tag,
        "pathway_score": float(router_score),
        "all_probs": {class_names[i]: float(probs[i]) for i in range(len(probs))}
    }
