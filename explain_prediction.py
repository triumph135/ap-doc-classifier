"""Explain one prediction: the scores, which words pushed the result, and how
familiar the document's wording is to the model.

    python explain_prediction.py "C:\\path\\to\\document.pdf"
    python explain_prediction.py "C:\\path\\to\\document.pdf" --show-text
"""
import argparse

from docclf.classify import Classifier


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--top", type=int, default=12, help="how many words to list")
    ap.add_argument("--show-text", action="store_true", help="print the extracted text")
    args = ap.parse_args()

    clf = Classifier()
    result = clf.predict_file(args.file, threshold=0.0, keep_text=True)
    if result["predicted"] is None:
        raise SystemExit(result["reason"])
    info = clf.explain_text(result["text"], args.top)

    print(f"Text source: {result['text_source']}   characters read: {len(result['text'])}")
    print(f"Top class: {result['predicted']} ({result['confidence']:.2f})   "
          f"Runner up: {result['runner_up']} ({result['runner_up_confidence']:.2f})")
    print(f"Familiarity: {info['known_words']} of {info['total_words']} distinct words "
          f"({info['familiarity']:.0%}) appear in the model's training vocabulary")
    print(f"\nWords pushing toward {info['top_class']}:    " + (", ".join(info["words_for_top"]) or "none"))
    print(f"Words pushing toward {info['runner_up']}:    " + (", ".join(info["words_for_runner_up"]) or "none"))
    if args.show_text:
        print("\n--- extracted text ---\n" + result["text"])


if __name__ == "__main__":
    main()
