"""Load the promoted model and classify documents. Used by the app and the batch script."""
import json
from pathlib import Path

import joblib
import sklearn

from . import config
from .extraction import get_text

MANUAL_REVIEW = "Manual Review"


class Classifier:
    def __init__(self, model_path=None, meta_path=None):
        model_path = Path(model_path or config.LATEST_MODEL)
        meta_path = Path(meta_path or (model_path.with_suffix(".json")))
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run: python train_model.py")
        self.model = joblib.load(model_path)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.classes = [str(c) for c in self.model.classes_]
        self.mode = self.meta.get("extraction_mode", "auto")
        self.warnings = []
        trained_with = self.meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            self.warnings.append(f"Model was trained with scikit-learn {trained_with} but "
                                 f"{sklearn.__version__} is installed. Retrain or pin the version.")

    @property
    def missing_classes(self):
        """Official classes this model has never seen examples of."""
        return [c for c in config.CLASSES if c not in self.classes]

    def predict_text(self, text, source="unknown", threshold=None):
        threshold = config.REVIEW_THRESHOLD if threshold is None else threshold
        if len(text.strip()) < config.MIN_TEXT_CHARS:
            return {"predicted": None, "confidence": 0.0, "runner_up": None, "runner_up_confidence": 0.0,
                    "final": MANUAL_REVIEW, "cargas_document_type": None, "text_source": source,
                    "reason": "Too little readable text"}
        proba = self.model.predict_proba([text])[0]
        order = proba.argsort()[::-1]
        best, second = order[0], (order[1] if len(order) > 1 else None)
        label, conf = self.classes[best], float(proba[best])
        accepted = conf >= threshold
        return {
            "predicted": label,
            "confidence": round(conf, 4),
            "runner_up": self.classes[second] if second is not None else None,
            "runner_up_confidence": round(float(proba[second]), 4) if second is not None else 0.0,
            "final": label if accepted else MANUAL_REVIEW,
            "cargas_document_type": config.CARGAS_DOCUMENT_TYPE.get(label) if accepted else None,
            "text_source": source,
            "reason": "" if accepted else f"Confidence below {threshold:.2f}",
        }

    def predict_file(self, path, threshold=None, use_cache=False):
        res = get_text(path, self.mode, use_cache)
        if res["source"] == "error":
            return {"predicted": None, "confidence": 0.0, "runner_up": None, "runner_up_confidence": 0.0,
                    "final": MANUAL_REVIEW, "cargas_document_type": None, "text_source": "error",
                    "reason": f"Could not read file: {res['error']}"}
        return self.predict_text(res["text"], res["source"], threshold)
