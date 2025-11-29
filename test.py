# Test.py
"""
Complete BitBrain Test harness.
"""
from core.fast_learner import MobileNetFastLearner
from core.memory_bank import MemoryBank
from core.router import Router
from core.predict import predict_image
from core.utils_data import get_dataloaders
from core.config import TORCH_DEVICE, ROUTER_THRESHOLD
import torch
import os

print("Initializing BitBrain (test)...")
train_loader, _ = get_dataloaders("data", batch_size=1)
class_names = train_loader.dataset.classes # type: ignore
model = MobileNetFastLearner(num_classes=len(class_names), pretrained=False, capture_stages=5)
ckpt = "checkpoints/model_best.pt"
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.to(TORCH_DEVICE)
model.eval()

memory_bank = MemoryBank("memory_bank")
router = Router(memory_bank, threshold=ROUTER_THRESHOLD)

print(f"Memory bank: {len(memory_bank.nodes)} nodes, {len(memory_bank.pathways)} pathways")

test_cases = [
    # change these paths to your images
    (r"C:\Users\Grey\Downloads\archive (2)\cats_set\cat.4311.jpg", "cat"),
    (r"C:\Users\Grey\Downloads\archive (2)\dogs_set\dog.4013.jpg", "dog"),
    (r"C:\Users\Grey\Downloads\archive (3)\Birds\train\train_2556.jpg", "bird"),
    # (r"C:\Users\Grey\Downloads\archive (4)\banana_classification\test\overripe\musa-acuminata-mold-e61f3fe3-1d0a-11ec-b881-d8c4975e38aa_jpg.rf.6658b435fdfbc7604a51fc29e402e7fa.jpg","banana"),

    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\watermelon\img_1341.jpeg","watermelon"),
    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\kiwi\img_1311.jpeg","kiwi"),
    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\mango\img_1161.jpeg","mango"),
    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\orange\img_901.jpeg","orange"),
    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\pinenapple\img_161.jpeg","pinenapple"),
    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\stawberries\img_91.jpeg","stawberries"),
    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\avocado\img_311.jpeg","avocado"),
    # (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\cherry\img_291.jpeg","cherry"),
    (r"C:\Users\Grey\Downloads\archive (5)\MY_data\test\apple\img_131.jpeg","apple")



]

pathway_used = 0
nn_used = 0
pathway_correct = 0
nn_correct = 0

for img_path, expected in test_cases:
    print("\n" + "-"*40)
    print("Image:", os.path.basename(img_path))
    result = predict_image(model, router, memory_bank, img_path, class_names)
    print("Predicted:", result["prediction"], "Source:", result["source"], "Confidence:", result["confidence"])
    correct = (result["prediction"] == expected)
    if result["source"] == "pathway":
        pathway_used += 1
        if correct:
            pathway_correct += 1
    else:
        nn_used += 1
        if correct:
            nn_correct += 1

print("\nSummary")
print("Pathway used:", pathway_used)
print("Pathway accuracy:", pathway_correct)
print("NN used:", nn_used)
print("NN accuracy:", nn_correct)
print("Router threshold:", router.threshold)
if pathway_used == 0:
    print("\n⚠️ No pathway predictions; check memory_bank/ and router threshold.")
