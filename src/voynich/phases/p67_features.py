"""
Phase 67, Track 3: Feature-Based Prediction from Confirmed Triples
====================================================================
Learn a mapping from stroke features (first_stroke, last_stroke,
glyph_class) to syllable values using the 12 confirmed triples,
then predict the 13 unresolved triples.

With only 12 training samples and ~20 target classes, accuracy will
be low.  The primary value is in soft probability rankings rather
than hard predictions.

Dependency chain:
    results/combined_refine.json      (Phase 15)
    results/triple_tiers.json         (Phase 28/53)
    EVA_VISUAL_COMPONENTS             (reference.py)
        -> results/p67_features.json
"""

import json
import os
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.reference import EVA_VISUAL_COMPONENTS


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Confirmed / unresolved triple separation
# ---------------------------------------------------------------------------

def _get_confirmed_and_unresolved(
    rd: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (confirmed_12, unresolved_13).  Only truly CONFIRMED triples."""
    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    tier_data = _safe_load(os.path.join(rd, 'triple_tiers.json'))
    confirmed_keys: Set[str] = set()

    if tier_data and 'tiers' in tier_data:
        tiers = tier_data['tiers']
        if isinstance(tiers, dict):
            for entry in tiers.get('CONFIRMED', []):
                confirmed_keys.add(entry.get('triple_key', ''))
        elif isinstance(tiers, list):
            for entry in tiers:
                if entry.get('tier', '') == 'CONFIRMED':
                    confirmed_keys.add(entry.get('triple_key', ''))

    if not confirmed_keys:
        return dict(assignment), {}

    confirmed = {k: v for k, v in assignment.items() if k in confirmed_keys}
    unresolved = {k: v for k, v in assignment.items() if k not in confirmed_keys}
    return confirmed, unresolved


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictionDetail:
    triple_key: str
    predicted_syllable: str
    confidence: float
    top_candidates: List[Dict[str, Any]]  # [{syllable, probability}, ...]
    current_tp15: str


@dataclass
class FeaturePredictionResult:
    phase: str = "67"
    step: str = "67.3"
    experiment: str = "feature_prediction"
    n_confirmed: int = 0
    n_unresolved: int = 0
    feature_dim: int = 0
    n_classes: int = 0
    # Feature vocabulary
    first_strokes: List[str] = field(default_factory=list)
    last_strokes: List[str] = field(default_factory=list)
    glyph_classes: List[str] = field(default_factory=list)
    # LOO-CV results per classifier
    loo_results: Dict[str, float] = field(default_factory=dict)
    best_classifier: str = ""
    best_loo_accuracy: float = 0.0
    # Predictions
    predictions: List[PredictionDetail] = field(default_factory=list)
    n_in_costamagna: int = 0
    n_agree_with_tp15: int = 0
    # Gates
    g1_onset_loo: bool = False       # P1: onset accuracy > 50%
    g2_vowel_loo: bool = False       # P2: vowel accuracy > 40%
    g3_in_costamagna: bool = False   # P3: >= 8/13 in inventory
    g4_agree_freq: bool = False      # P4: >= 3 agree with Track 2
    gates_passed: int = 0
    gate_passed: bool = False
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _build_feature_vocabulary() -> Tuple[List[str], List[str], List[str]]:
    """Extract all stroke and class values from EVA_VISUAL_COMPONENTS."""
    first_strokes: Set[str] = set()
    last_strokes: Set[str] = set()
    glyph_classes: Set[str] = set()

    for components in EVA_VISUAL_COMPONENTS.values():
        first_strokes.add(components['first_stroke'])
        last_strokes.add(components['last_stroke'])
        glyph_classes.add(components['glyph_class'])

    return sorted(first_strokes), sorted(last_strokes), sorted(glyph_classes)


def _encode_triple(
    triple_key: str,
    first_vocab: List[str],
    last_vocab: List[str],
    class_vocab: List[str],
) -> List[int]:
    """One-hot encode a triple key into a binary feature vector."""
    parts = [p.strip() for p in triple_key.split(',')]
    if len(parts) != 3:
        return [0] * (len(first_vocab) + len(last_vocab) + len(class_vocab))

    first_stroke, last_stroke, glyph_class = parts

    vec = []
    for s in first_vocab:
        vec.append(1 if first_stroke == s else 0)
    for s in last_vocab:
        vec.append(1 if last_stroke == s else 0)
    for s in class_vocab:
        vec.append(1 if glyph_class == s else 0)

    return vec


def _extract_onset_vowel(syllable: str) -> Tuple[str, str]:
    """Extract onset consonant and nucleus vowel from a syllable."""
    vowels = set('aeiou')
    onset = ''
    vowel = ''

    for i, c in enumerate(syllable):
        if c in vowels:
            onset = syllable[:i]
            # Collect all vowels (for diphthongs)
            vowel = c
            break
    else:
        # No vowel found — treat whole thing as onset
        onset = syllable
        vowel = ''

    return onset, vowel


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------

def _build_training_data(
    triples: Dict[str, str],
    first_vocab: List[str],
    last_vocab: List[str],
    class_vocab: List[str],
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Build X matrix and y labels from triple assignments.

    Returns (X, y_syllable, y_onset, y_vowel).
    """
    keys = sorted(triples.keys())
    X = np.array([_encode_triple(k, first_vocab, last_vocab, class_vocab)
                   for k in keys])
    y_syllable = [triples[k] for k in keys]
    y_onset = [_extract_onset_vowel(triples[k])[0] for k in keys]
    y_vowel = [_extract_onset_vowel(triples[k])[1] for k in keys]

    return X, y_syllable, y_onset, y_vowel


def _run_loo_cv(
    X: np.ndarray,
    y: List[str],
    classifiers: Dict,
) -> Dict[str, float]:
    """Leave-one-out cross-validation for each classifier.

    Returns {clf_name: accuracy}.
    """
    n = len(y)
    results = {}

    for clf_name, clf_factory in classifiers.items():
        correct = 0
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            X_test = X[i:i+1]
            y_train = [y[j] for j in range(n) if j != i]

            try:
                clf = clf_factory()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)
                if pred[0] == y[i]:
                    correct += 1
            except Exception:
                pass

        results[clf_name] = correct / n if n > 0 else 0.0

    return results


def _predict_with_probabilities(
    X_train: np.ndarray,
    y_train: List[str],
    X_test: np.ndarray,
    test_keys: List[str],
    classifiers: Dict,
) -> List[Dict[str, Any]]:
    """Train all classifiers on full training set, predict test set.

    For each test sample, collect probability-like scores from all classifiers
    and produce a weighted ensemble.

    Returns list of {triple_key, predicted, confidence, top_candidates}.
    """
    all_classes = sorted(set(y_train))
    n_test = X_test.shape[0]

    # Collect predictions from each classifier
    clf_predictions = {}
    for clf_name, clf_factory in classifiers.items():
        try:
            clf = clf_factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_train, y_train)

                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(X_test)  # (n_test, n_classes)
                    # Map to all_classes ordering
                    clf_classes = list(clf.classes_)
                    full_proba = np.zeros((n_test, len(all_classes)))
                    for ci, c in enumerate(clf_classes):
                        if c in all_classes:
                            idx = all_classes.index(c)
                            full_proba[:, idx] = proba[:, ci]
                    clf_predictions[clf_name] = full_proba
                else:
                    pred = clf.predict(X_test)
                    # Convert hard prediction to one-hot
                    full_proba = np.zeros((n_test, len(all_classes)))
                    for i, p in enumerate(pred):
                        if p in all_classes:
                            idx = all_classes.index(p)
                            full_proba[i, idx] = 1.0
                    clf_predictions[clf_name] = full_proba
        except Exception:
            clf_predictions[clf_name] = np.zeros((n_test, len(all_classes)))

    # Ensemble: average probabilities
    ensemble = np.zeros((n_test, len(all_classes)))
    for proba in clf_predictions.values():
        ensemble += proba
    n_clfs = len(clf_predictions)
    if n_clfs > 0:
        ensemble /= n_clfs

    # Build results
    results = []
    for i in range(n_test):
        sorted_indices = np.argsort(ensemble[i])[::-1]
        top_candidates = []
        for idx in sorted_indices[:5]:
            if ensemble[i, idx] > 0:
                top_candidates.append({
                    'syllable': all_classes[idx],
                    'probability': round(float(ensemble[i, idx]), 4),
                })

        best_idx = sorted_indices[0]
        results.append({
            'triple_key': test_keys[i],
            'predicted': all_classes[best_idx],
            'confidence': round(float(ensemble[i, best_idx]), 4),
            'top_candidates': top_candidates,
        })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_feat_predict():
    """Track 3: Feature-based prediction of unresolved triples."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 67.3 — Feature-Based Prediction")
    print("=" * 42)

    # --- Load data ---
    confirmed, unresolved = _get_confirmed_and_unresolved(rd)
    print(f"  Confirmed triples: {len(confirmed)}")
    print(f"  Unresolved triples: {len(unresolved)}")

    if len(confirmed) < 3:
        print("  ERROR: Too few confirmed triples for prediction.")
        result = FeaturePredictionResult(
            n_confirmed=len(confirmed),
            n_unresolved=len(unresolved),
            runtime_seconds=round(time.time() - t0, 1),
        )
        _save_json(rd, 'p67_features.json', result)
        return

    # --- Feature vocabulary ---
    first_vocab, last_vocab, class_vocab = _build_feature_vocabulary()
    feature_dim = len(first_vocab) + len(last_vocab) + len(class_vocab)
    print(f"  Feature dim: {feature_dim} "
          f"({len(first_vocab)} first + {len(last_vocab)} last + {len(class_vocab)} class)")

    # --- Build training data ---
    X_train, y_syl, y_onset, y_vowel = _build_training_data(
        confirmed, first_vocab, last_vocab, class_vocab)
    n_classes = len(set(y_syl))
    print(f"  Classes (syllables): {n_classes}")
    print(f"  Onset classes: {len(set(y_onset))}")
    print(f"  Vowel classes: {len(set(y_vowel))}")

    # --- Classifiers ---
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    k_neighbors = min(3, len(confirmed) - 1)

    classifiers = {
        'knn': lambda: KNeighborsClassifier(n_neighbors=max(1, k_neighbors)),
        'tree': lambda: DecisionTreeClassifier(max_depth=2, random_state=42),
        'lr': lambda: LogisticRegression(
            max_iter=2000, random_state=42, C=0.1,
            solver='lbfgs', multi_class='multinomial'),
        'nb': lambda: GaussianNB(),
    }

    # --- LOO-CV for syllable ---
    print("\n  LOO-CV (syllable)...")
    loo_syl = _run_loo_cv(X_train, y_syl, classifiers)
    for name, acc in sorted(loo_syl.items()):
        print(f"    {name}: {acc:.1%}")

    # --- LOO-CV for onset ---
    print("  LOO-CV (onset)...")
    loo_onset = _run_loo_cv(X_train, y_onset, classifiers)
    for name, acc in sorted(loo_onset.items()):
        print(f"    {name}: {acc:.1%}")

    # --- LOO-CV for vowel ---
    print("  LOO-CV (vowel)...")
    loo_vowel = _run_loo_cv(X_train, y_vowel, classifiers)
    for name, acc in sorted(loo_vowel.items()):
        print(f"    {name}: {acc:.1%}")

    best_clf = max(loo_syl, key=loo_syl.get)
    best_acc = loo_syl[best_clf]
    best_onset_acc = max(loo_onset.values())
    best_vowel_acc = max(loo_vowel.values())

    # --- Predict unresolved ---
    print(f"\n  Predicting {len(unresolved)} unresolved triples...")
    unresolved_keys = sorted(unresolved.keys())
    X_test = np.array([_encode_triple(k, first_vocab, last_vocab, class_vocab)
                        for k in unresolved_keys])

    pred_results = _predict_with_probabilities(
        X_train, y_syl, X_test, unresolved_keys, classifiers)

    # Load Costamagna inventory for validation
    from voynich.phases.costamagna_csp import _load_costamagna_inventory
    cv_set, cvc_set, all_set = _load_costamagna_inventory()
    costamagna_all = cv_set | cvc_set

    predictions = []
    n_in_costamagna = 0
    n_agree_tp15 = 0

    for pr in pred_results:
        triple_key = pr['triple_key']
        predicted = pr['predicted']
        tp15_val = unresolved[triple_key]

        in_costa = predicted in costamagna_all
        if in_costa:
            n_in_costamagna += 1
        if predicted == tp15_val:
            n_agree_tp15 += 1

        predictions.append(PredictionDetail(
            triple_key=triple_key,
            predicted_syllable=predicted,
            confidence=pr['confidence'],
            top_candidates=pr['top_candidates'],
            current_tp15=tp15_val,
        ))

        marker = "✓ Costa" if in_costa else "✗"
        tp15_mark = " =T_P15" if predicted == tp15_val else ""
        print(f"    {triple_key}: {predicted} (conf={pr['confidence']:.2f}) "
              f"{marker}{tp15_mark}")

    # Check Track 2 agreement if available
    freq_data = _safe_load(os.path.join(rd, 'p67_frequency.json'))
    n_agree_freq = 0
    if freq_data and 'domains' in freq_data:
        freq_domains = {d['triple_key']: d['candidates']
                        for d in freq_data['domains']}
        for pred in predictions:
            if pred.triple_key in freq_domains:
                if pred.predicted_syllable in freq_domains[pred.triple_key]:
                    n_agree_freq += 1

    # --- Gates ---
    g1 = best_onset_acc > 0.50
    g2 = best_vowel_acc > 0.40
    g3 = n_in_costamagna >= 8
    g4 = n_agree_freq >= 3
    gates_passed = sum([g1, g2, g3, g4])

    result = FeaturePredictionResult(
        n_confirmed=len(confirmed),
        n_unresolved=len(unresolved),
        feature_dim=feature_dim,
        n_classes=n_classes,
        first_strokes=first_vocab,
        last_strokes=last_vocab,
        glyph_classes=class_vocab,
        loo_results={
            'syllable': loo_syl,
            'onset': loo_onset,
            'vowel': loo_vowel,
        },
        best_classifier=best_clf,
        best_loo_accuracy=round(best_acc, 4),
        predictions=predictions,
        n_in_costamagna=n_in_costamagna,
        n_agree_with_tp15=n_agree_tp15,
        g1_onset_loo=g1,
        g2_vowel_loo=g2,
        g3_in_costamagna=g3,
        g4_agree_freq=g4,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 2,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p67_features.json', result)

    # --- Summary ---
    print(f"\n  Summary")
    print(f"  -------")
    print(f"  Best classifier:   {best_clf} ({best_acc:.1%} syllable LOO)")
    print(f"  Best onset LOO:    {best_onset_acc:.1%} "
          f"({'PASS' if g1 else 'FAIL'} > 50%)")
    print(f"  Best vowel LOO:    {best_vowel_acc:.1%} "
          f"({'PASS' if g2 else 'FAIL'} > 40%)")
    print(f"  In Costamagna:     {n_in_costamagna}/13 "
          f"({'PASS' if g3 else 'FAIL'} >= 8)")
    print(f"  Agree w/ Track 2:  {n_agree_freq} "
          f"({'PASS' if g4 else 'FAIL'} >= 3)")
    print(f"  Gates: {gates_passed}/4")
    print(f"  Saved: {path}")
    print(f"  Time: {result.runtime_seconds:.1f}s")
