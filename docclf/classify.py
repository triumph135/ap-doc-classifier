"""Load the promoted model and classify documents. Used by the app and the batch script."""
import json
from pathlib import Path

import joblib
import numpy as np
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

    def predict_file(self, path, threshold=None, use_cache=False, keep_text=False):
        """Classify one file. keep_text=True adds the extracted text under the key "text"."""
        res = get_text(path, self.mode, use_cache)
        if res["source"] == "error":
            out = {"predicted": None, "confidence": 0.0, "runner_up": None, "runner_up_confidence": 0.0,
                   "final": MANUAL_REVIEW, "cargas_document_type": None, "text_source": "error",
                   "reason": f"Could not read file: {res['error']}"}
        else:
            out = self.predict_text(res["text"], res["source"], threshold)
        if keep_text:
            out["text"] = res["text"]
        return out

    def explain_text(self, text, top=12):
        """Why the model scored this text the way it did.

        familiarity is the share of the document's distinct words that appear in the
        model's training vocabulary. A low share usually means a new vendor or format.
        The word lists are the words that pushed hardest toward the top class and toward
        the runner up."""
        features = self.model.named_steps["features"]
        words = dict(features.transformer_list)["words"]
        coef = self.model.named_steps["clf"].coef_
        proba = self.model.predict_proba([text])[0]
        order = proba.argsort()[::-1]
        best, second = int(order[0]), int(order[1])

        tokens = {t for t in words.build_analyzer()(text) if " " not in t}
        known = sum(1 for t in tokens if t in words.vocabulary_)

        # binary models store one weight row, so split it evenly between the two classes
        weights = coef if coef.shape[0] > 1 else np.vstack([-coef[0] / 2, coef[0] / 2])
        x = features.transform([text])
        contrib = np.asarray(x.multiply(weights[best] - weights[second]).todense()).ravel()
        names = features.get_feature_names_out()
        is_word = np.array([n.startswith("words__") for n in names])
        idx = np.argsort(contrib)
        label = lambda i: names[i].split("__", 1)[1]
        return {
            "top_class": self.classes[best],
            "runner_up": self.classes[second],
            "known_words": known,
            "total_words": len(tokens),
            "familiarity": known / max(1, len(tokens)),
            "words_for_top": [label(i) for i in idx[::-1] if is_word[i] and contrib[i] > 0][:top],
            "words_for_runner_up": [label(i) for i in idx if is_word[i] and contrib[i] < 0][:top],
        }
