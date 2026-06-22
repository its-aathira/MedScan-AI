# ─────────────────────────────────────────────
#  backend/api/model_registry.py
#  Manages model loading and caching
# ─────────────────────────────────────────────

from pathlib import Path
from typing  import Dict, Optional

from configs.dataset_config        import MODULES, BASE_MODEL_DIR
from backend.inference.predictor   import MedicalPredictor


class ModelRegistry:
    """
    Loads and caches MedicalPredictor instances for each module.
    Only loads modules whose checkpoint files exist on disk.
    """

    def __init__(self, weights_dir: str = BASE_MODEL_DIR):
        self.weights_dir  = Path(weights_dir)
        self.predictors   : Dict[str, MedicalPredictor] = {}

    def load_available(self):
        from huggingface_hub import hf_hub_download
        import os
    
        print(f"  Downloading model weights from HuggingFace...")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
    
        for module_key in MODULES:
            filename = f"{module_key}_best.pt"
            ckpt = self.weights_dir / filename
        
            if not ckpt.exists():
                try:
                    print(f"  ⬇  Downloading {filename}...")
                    hf_hub_download(
                    repo_id="aathix/medscan-ai-weights",
                    filename=filename,
                    local_dir=str(self.weights_dir),
                    token=os.getenv("HF_TOKEN"),
                    )
                    print(f"  ✓  {filename} downloaded")
                except Exception as e:
                    print(f"  ⚠  Failed to download {module_key}: {e}")
                    continue
        
            try:
                self.predictors[module_key] = MedicalPredictor(
                    module_key=module_key,
                    checkpoint_path=str(ckpt),
                )
            except Exception as e:
                print(f"  ⚠  Failed to load {module_key}: {e}")
    
        print(f"\n  Loaded modules: {list(self.predictors.keys())}")

    def get(self, module_key: str) -> Optional[MedicalPredictor]:
        return self.predictors.get(module_key)

    def available_modules(self) -> list:
        return list(self.predictors.keys())

    def module_info(self) -> list:
        info = []
        for key, predictor in self.predictors.items():
            cfg = MODULES[key]
            info.append({
                "module_key"  : key,
                "name"        : cfg["name"],
                "num_classes" : cfg["num_classes"],
                "classes"     : cfg["classes"],
                "dataset"     : cfg["dataset"],
            })
        return info