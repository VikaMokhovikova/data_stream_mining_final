"""Leakage-safe hyperparameter search for the PA44 heat-stress stream models.

Protocol
--------
* Selection uses ONLY the 2016-2017 warm-up period:
  learn on 2016, then predict -> score -> learn on every 2017 work hour.
  Imputation medians and scaling for this inner loop are fitted on 2016 only.
* One candidate set is sampled per model (seeded) and scored on both cities
  and both representations; the winner for each (model, representation) is
  the candidate with the highest mean 2017 macro-F1 across the two cities.
* The winners are then replayed under the unchanged official protocol
  (learn 2016-2017, predict -> evaluate -> learn 2018-2025, warm-up scaling),
  next to the notebook baseline, so the 2018-2025 stream is never used to choose.

Inputs are the processed paired model streams written by the main notebook
(data/processed/city_*_model_stream.csv). Outputs go to results/tuning/.

Usage:  python src/tune_hyperparameters.py [--trials N] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUT_DIR = PROJECT_ROOT / 'results' / 'tuning'

CITY_FILES = {'Portimão': 'city_A_model_stream.csv', 'Covilhã': 'city_B_model_stream.csv'}
WARMUP_END = pd.Timestamp('2017-12-31 23:00:00', tz='UTC')
INNER_TRAIN_END = pd.Timestamp('2016-12-31 23:00:00', tz='UTC')
RANDOM_SEED = 42

RAW = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m']
REP_A = [f'lag1_{v}' for v in RAW]
REP_B = REP_A + ['temperature_change_1h', 'temperature_mean_6h', 'temperature_std_6h',
                 'humidity_mean_6h', 'rainy_hours_24h', 'shift_hour_sin', 'shift_hour_cos']
REPS = {'A': REP_A, 'B': REP_B}

# Notebook configuration (cell "MODEL_PARAMETERS"), expressed in the same
# parameter vocabulary as the search spaces below.
BASE_TREE = dict(grace_period=50, delta=1e-3, tau=0.1, leaf_prediction='mc',
                 split_criterion='info_gain')
BASELINE = {
    'Gaussian Naive Bayes': {},
    'Online Logistic Regression': dict(optimizer='SGD', lr=0.05, l2=0.0, intercept_lr=0.01),
    'Hoeffding Tree': dict(BASE_TREE),
    'Online Bagging': dict(n_models=5, **BASE_TREE),
    'Adaptive Random Forest': dict(n_models=5, max_features='sqrt', lambda_value=6, **BASE_TREE),
    'UnimPA': dict(max_rules=5, alpha_add=0.3, tau_merge=0.7, lambda_sim=0.5, e=0.5,
                   act_min_update=0.3, dist_safe_factor=3.0, C=1.0,
                   buffer_size_separability=200),
    'UnimSafe': dict(max_rules=5, alpha_add=0.3, tau_merge=0.7, e=0.5, act_min_update=0.3,
                     dist_safe_factor=3.0, forget_counts=0.99, min_rules_for_merge=5,
                     min_support_for_merge=10, buffer_size_separability=200),
}
MODEL_NAMES = list(BASELINE)


# ----------------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------------

def loguniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_tree(rng):
    return dict(
        grace_period=rng.choice([10, 25, 50, 100, 200, 400]),
        delta=loguniform(rng, 1e-9, 1e-1),
        tau=rng.choice([0.01, 0.05, 0.1, 0.2]),
        leaf_prediction=rng.choice(['mc', 'nb', 'nba']),
        split_criterion=rng.choice(['info_gain', 'gini', 'hellinger']),
    )


def sample(model_name, rng):
    if model_name == 'Online Logistic Regression':
        return dict(optimizer=rng.choice(['SGD', 'Adam', 'AdaGrad']),
                    lr=loguniform(rng, 1e-3, 0.5),
                    l2=rng.choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
                    intercept_lr=rng.choice([0.001, 0.01, 0.1]))
    if model_name == 'Hoeffding Tree':
        return sample_tree(rng)
    if model_name == 'Online Bagging':
        return dict(n_models=rng.choice([5, 10, 15, 20, 30]), **sample_tree(rng))
    if model_name == 'Adaptive Random Forest':
        return dict(n_models=rng.choice([5, 10, 15, 20, 30]),
                    max_features=rng.choice(['sqrt', 'log2', 0.3, 0.5, 0.8]),
                    lambda_value=rng.choice([1, 3, 6, 10]), **sample_tree(rng))
    if model_name == 'UnimPA':
        return dict(max_rules=rng.choice([2, 3, 5, 8, 12, 20]),
                    alpha_add=rng.uniform(0.05, 0.8), tau_merge=rng.uniform(0.5, 0.95),
                    lambda_sim=rng.uniform(0.1, 0.9), e=rng.uniform(0.2, 0.8),
                    act_min_update=rng.uniform(0.05, 0.6),
                    dist_safe_factor=rng.uniform(1.5, 6.0), C=loguniform(rng, 0.01, 10.0),
                    buffer_size_separability=rng.choice([100, 200, 500]))
    if model_name == 'UnimSafe':
        return dict(max_rules=rng.choice([2, 3, 5, 8, 12, 20]),
                    alpha_add=rng.uniform(0.05, 0.8), tau_merge=rng.uniform(0.5, 0.95),
                    e=rng.uniform(0.2, 0.8), act_min_update=rng.uniform(0.05, 0.6),
                    dist_safe_factor=rng.uniform(1.5, 6.0),
                    forget_counts=rng.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]),
                    min_rules_for_merge=rng.choice([2, 3, 5, 8]),
                    min_support_for_merge=rng.choice([5, 10, 20, 50]),
                    buffer_size_separability=rng.choice([100, 200, 500]))
    raise KeyError(model_name)


TRIAL_WEIGHT = {  # relative share of the trial budget (cheap models get more)
    'Online Logistic Regression': 1.0, 'Hoeffding Tree': 1.5, 'Online Bagging': 1.0,
    'Adaptive Random Forest': 1.0, 'UnimPA': 1.5, 'UnimSafe': 1.5,
}


def candidate_sets(trials):
    out = {'Gaussian Naive Bayes': [BASELINE['Gaussian Naive Bayes']]}  # no hyperparameters
    for name, weight in TRIAL_WEIGHT.items():
        rng = random.Random(f'{RANDOM_SEED}-{name}')
        out[name] = [dict(BASELINE[name])] + [sample(name, rng)
                                              for _ in range(int(trials * weight) - 1)]
    return out


# ----------------------------------------------------------------------------
# Model construction (mirrors the notebook's new_model)
# ----------------------------------------------------------------------------

def make_tree(p):
    from river import tree
    return tree.HoeffdingTreeClassifier(grace_period=p['grace_period'], delta=p['delta'],
                                        tau=p['tau'], leaf_prediction=p['leaf_prediction'],
                                        split_criterion=p['split_criterion'])


class FuzzyAdapter:
    def __init__(self, cls, n_features, params):
        self.learner = cls(n_features=n_features, n_classes=2, random_state=RANDOM_SEED, **params)

    def predict_row(self, row):
        return int(self.learner.predict(row[None, :])[0])

    def learn_row(self, row, y):
        self.learner.partial_fit(row[None, :], np.array([y]))


class RiverAdapter:
    def __init__(self, model, columns):
        self.model, self.columns = model, columns

    def predict_row(self, row):
        p = self.model.predict_one(dict(zip(self.columns, row.tolist())))
        return 0 if p is None else int(p)

    def learn_row(self, row, y):
        self.model.learn_one(dict(zip(self.columns, row.tolist())), y)


def build(model_name, params, columns):
    from river import ensemble, forest, linear_model, naive_bayes, optim
    if model_name == 'UnimPA':
        from ENF_UnimNeuron_PA import EvolvingNeuroFuzzyUnimPA
        return FuzzyAdapter(EvolvingNeuroFuzzyUnimPA, len(columns), params)
    if model_name == 'UnimSafe':
        from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
        return FuzzyAdapter(EvolvingNeuroFuzzyUnimSafe, len(columns), params)
    if model_name == 'Gaussian Naive Bayes':
        model = naive_bayes.GaussianNB()
    elif model_name == 'Online Logistic Regression':
        opt = getattr(optim, params['optimizer'])(params['lr'])
        model = linear_model.LogisticRegression(optimizer=opt, l2=params['l2'],
                                                intercept_lr=params['intercept_lr'])
    elif model_name == 'Hoeffding Tree':
        model = make_tree(params)
    elif model_name == 'Online Bagging':
        model = ensemble.BaggingClassifier(model=make_tree(params), n_models=params['n_models'],
                                           seed=RANDOM_SEED)
    elif model_name == 'Adaptive Random Forest':
        model = forest.ARFClassifier(
            n_models=params['n_models'], max_features=params['max_features'],
            lambda_value=params['lambda_value'], grace_period=params['grace_period'],
            delta=params['delta'], tau=params['tau'], leaf_prediction=params['leaf_prediction'],
            split_criterion=params['split_criterion'], seed=RANDOM_SEED)
    else:
        raise KeyError(model_name)
    return RiverAdapter(model, columns)


# ----------------------------------------------------------------------------
# Data preparation and prequential replay
# ----------------------------------------------------------------------------

def load_streams():
    streams = {}
    for city, file in CITY_FILES.items():
        # round_trip parsing reproduces the notebook's float64 values bit-for-bit.
        frame = pd.read_csv(PROCESSED_DIR / file, index_col='timestamp_utc',
                            float_precision='round_trip')
        frame.index = pd.to_datetime(frame.index, utc=True)
        streams[city] = frame.sort_index()
    index = streams['Portimão'].index
    assert all(f.index.equals(index) for f in streams.values()), 'cities must be paired'
    return streams


def prepare(streams, stats_end):
    """Median-impute and standardise with statistics from rows <= stats_end only."""
    pooled = pd.concat([f.loc[f.index <= stats_end, REP_B] for f in streams.values()],
                       ignore_index=True)
    medians = pooled.median(numeric_only=True)
    arrays = {}
    for rep, cols in REPS.items():
        scaler = StandardScaler().fit(pooled[cols].fillna(medians))
        for city, frame in streams.items():
            arrays[(city, rep)] = scaler.transform(frame[cols].fillna(medians))
    labels = {city: f['y_event'].astype(int).to_numpy() for city, f in streams.items()}
    return arrays, labels


def macro_f1(cm):
    (tn, fp), (fn, tp) = cm
    def f1(p_num, p_den, r_den):
        p = p_num / p_den if p_den else 0.0
        r = p_num / r_den if r_den else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0
    f1_event = f1(tp, tp + fp, tp + fn)
    f1_none = f1(tn, tn + fn, tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n = tn + fp + fn + tp
    return dict(macro_f1=(f1_event + f1_none) / 2, accuracy=(tn + tp) / n,
                precision_event=precision, recall_event=recall,
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def replay(model, X, y, n_learn):
    for i in range(n_learn):
        model.learn_row(X[i], int(y[i]))
    cm = np.zeros((2, 2), dtype=int)
    for i in range(n_learn, len(y)):
        cm[y[i], model.predict_row(X[i])] += 1
        model.learn_row(X[i], int(y[i]))
    return macro_f1(cm)


# Worker state (populated once per process).
_DATA = {}


def _init_worker():
    streams = load_streams()
    warm = {c: f.loc[f.index <= WARMUP_END] for c, f in streams.items()}
    inner_arrays, inner_labels = prepare(warm, INNER_TRAIN_END)
    full_arrays, full_labels = prepare(streams, WARMUP_END)
    index = streams['Portimão'].index
    _DATA['validation'] = (inner_arrays, inner_labels, int((index <= INNER_TRAIN_END).sum()))
    _DATA['official'] = (full_arrays, full_labels, int((index <= WARMUP_END).sum()))


def run_job(job):
    phase, model_name, cand_id, params, city, rep = job
    arrays, labels, n_learn = _DATA[phase]
    start = time.perf_counter()
    model = build(model_name, params, REPS[rep])
    result = replay(model, arrays[(city, rep)], labels[city], n_learn)
    result.update(phase=phase, model=model_name, candidate=cand_id, city=city,
                  representation=rep, seconds=time.perf_counter() - start,
                  params=json.dumps(params))
    return result


def run_parallel(jobs, workers, label):
    rows, done, t0 = [], 0, time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f'  [{label}] {done}/{len(jobs)} runs, {time.perf_counter() - t0:.0f}s',
                      flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=80,
                        help='base candidates per model (scaled by TRIAL_WEIGHT)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = candidate_sets(args.trials)
    print('Candidates per model:', {k: len(v) for k, v in candidates.items()})

    # ---- Stage 1: selection on the warm-up (learn 2016, prequential 2017) ----
    jobs = [('validation', m, i, p, city, rep)
            for m, cands in candidates.items() for i, p in enumerate(cands)
            for city in CITY_FILES for rep in REPS]
    val = run_parallel(jobs, args.workers, 'validation 2017')
    val.to_csv(OUT_DIR / 'validation_trials.csv', index=False)

    mean_val = (val.groupby(['model', 'representation', 'candidate', 'params'])['macro_f1']
                .mean().rename('val_macro_f1_mean').reset_index())
    best_rows = mean_val.loc[mean_val.groupby(['model', 'representation'])
                             ['val_macro_f1_mean'].idxmax()]
    base_rows = mean_val[mean_val.candidate == 0].set_index(['model', 'representation'])
    best = {}
    for row in best_rows.itertuples():
        best.setdefault(row.model, {})[row.representation] = {
            'candidate': int(row.candidate), 'params': json.loads(row.params),
            'validation_macro_f1_mean': row.val_macro_f1_mean,
            'baseline_validation_macro_f1_mean':
                float(base_rows.loc[(row.model, row.representation), 'val_macro_f1_mean']),
        }
    (OUT_DIR / 'best_parameters.json').write_text(
        json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- Stage 2: official protocol for baseline and selected configurations ----
    jobs = []
    for m in MODEL_NAMES:
        for rep in REPS:
            sel = best[m][rep]
            for city in CITY_FILES:
                jobs.append(('official', m, 0, BASELINE[m], city, rep))
                if sel['candidate'] != 0:
                    jobs.append(('official', m, sel['candidate'], sel['params'], city, rep))
    off = run_parallel(jobs, args.workers, 'official 2018-2025')
    off['config'] = np.where(off.candidate == 0, 'baseline', 'tuned')
    off.to_csv(OUT_DIR / 'official_runs.csv', index=False)

    # baseline rows double as 'tuned' when the baseline won the selection
    tuned = off[off.config == 'tuned']
    keep_base = off[off.config == 'baseline'].merge(
        tuned[['model', 'city', 'representation']], how='left', indicator=True,
        on=['model', 'city', 'representation'])
    keep_base = keep_base[keep_base._merge == 'left_only'].drop(columns='_merge')
    tuned_all = pd.concat([tuned, keep_base.assign(config='tuned')], ignore_index=True)
    base = off[off.config == 'baseline']
    key = ['city', 'representation', 'model']
    comparison = base[key + ['macro_f1', 'recall_event', 'precision_event']].merge(
        tuned_all[key + ['macro_f1', 'recall_event', 'precision_event', 'candidate']],
        on=key, suffixes=('_baseline', '_tuned'))
    comparison['delta_macro_f1'] = comparison.macro_f1_tuned - comparison.macro_f1_baseline
    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    comparison = comparison.sort_values(['city', 'representation', 'model'],
                                        key=lambda s: s.map(order) if s.name == 'model' else s)
    comparison.to_csv(OUT_DIR / 'official_comparison.csv', index=False)

    pd.set_option('display.width', 200)
    print('\nOfficial 2018-2025 macro-F1 (selection used 2016-2017 only):')
    print(comparison[key + ['macro_f1_baseline', 'macro_f1_tuned', 'delta_macro_f1']]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
