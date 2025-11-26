# inference/run_inference.py
import argparse
from core.fast_learner import MobileNetFastLearner
from core.memory_bank import MemoryBank
from core.router import Router
from core.predict import predict_image
from core.utils_data import get_dataloaders
from core.config import TORCH_DEVICE
import torch

def run_demo(model_ckpt: str, img_path: str, data_dir: str = "data"):
    train_loader, _ = get_dataloaders(data_dir, batch_size=1)
    class_names = train_loader.dataset.classes # type: ignore
    model = MobileNetFastLearner(num_classes=len(class_names), pretrained=True)
    model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
    model.to(TORCH_DEVICE)
    model.eval()
    mb = MemoryBank("memory_bank")
    router = Router(mb)
    result = predict_image(model, router, mb, img_path, class_names)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--data", type=str, default="data")
    args = parser.parse_args()
    run_demo(args.ckpt, args.img, args.data)
