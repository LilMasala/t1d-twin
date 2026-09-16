# Personal type 1 diabetes twins

Fit a UVA/Padova simulator to one person's CGM, insulin and context data, then ask it what would have happened under different pump settings.

The model is mechanistic, so its parameters are physiological quantities (insulin sensitivity, absorption speed, endogenous glucose production) rather than network weights, and you can change a carb ratio and re-run the day. It is written in torch, so those parameters are fitted by gradient rather than by grid search.

On held-out data it forecasts about as well as [T1DSim_AI](https://github.com/Mosqueralopez/T1DSim_AI), a neural digital twin published in *Neural Computing and Applications*, and predicts hypoglycaemia events considerably better. Numbers and caveats are below.

## What it fits

Per person, from their own records:

| Group | Parameters |
| --- | --- |
| Core physiology | insulin sensitivity, endogenous glucose production, insulin absorption and action speed, carb absorption speed and bioavailability, hypoglycaemic counter-regulation |
| Context | menstrual cycle phase (or an inferred 28-day rhythm), exercise, sleep debt, dawn phenomenon, infusion site age, stress, a 24-hour sensitivity rhythm |
| Response shape | 10 carb and 10 insulin response coefficients in 30-minute bins over 5 hours |
| Per day | insulin sensitivity and EGP drift |
| Per meal | carb-count error and timing error, because logged carbs are not trustworthy |
| Unexplained | a signed glucose disturbance in 30-minute blocks, with a heavy-tailed prior |
| Noise | CGM residual spread |

Meals that were never logged are picked up from carb-free boluses, food photos and CGM rises, then fitted as free amounts. Missing signals are inferred instead of switched off: a person with no recorded cycle days gets a free 28-day rhythm, a pump with no logged site changes gets changes inferred from suspensions.

Fitting runs in two stages: a 12-start batched MAP with CGM noise held fixed, then stochastic variational inference with a full-rank guide over the global parameters. Both optimise in prior-standardised units.

## Results

### Against T1DSim_AI

Both models get the same split (our fitted days train, later days held out), the same five-hour evaluation sequences (chosen by T1DSim_AI's own `SequenceSelection`), and the same metrics. T1DSim_AI is trained with its authors' published recipe, and also with a tuned learning rate that scores better than the published one. Our harness reproduces its published example twin exactly (test RMSE 23.96).

Pooled per-sequence RMSE, mg/dL:

| Model | HUPA-UCM (17 people, 217 seq) | T1D-UOM (4 pump users, 56 seq) |
| --- | --- | --- |
| T1DSim_AI population model | 53.4 | 57.2 |
| T1DSim_AI personal twin | 49.7 | 46.3 |
| T1DSim_AI personal twin, tuned | 48.4 | 45.3 |
| This twin | 48.0 | 39.0 |
| This twin, v2 priors | 47.5 | 41.8 |

Paired per-sequence differences against the tuned baseline are −0.4 to −0.9 mg/dL on HUPA and −3.5 to −6.3 on UOM, none of them significant (Wilcoxon p between 0.28 and 0.91). Against the un-personalised population model the difference is significant on both cohorts. The fair summary is that forecast accuracy is comparable to a fine-tuned neural twin, not better.

### Hypoglycaemia

A low event is at least three consecutive readings below 70 mg/dL inside the five-hour horizon. This twin gives each sequence a probability from a 32-draw posterior ensemble; T1DSim_AI is deterministic, so it also gets scored with our CGM noise model added, which turns its answers into probabilities.

HUPA-UCM, 50 low events in 217 sequences:

| Model | AUROC | Brier | Sensitivity | PPV |
| --- | --- | --- | --- | --- |
| T1DSim_AI personal twin, tuned | 0.55 | 0.272 | 0.22 | 0.36 |
| Tuned, plus our CGM noise | 0.60 | 0.252 | 0.22 | 0.33 |
| This twin, v2 priors | 0.74 | 0.156 | 0.40 | 0.74 |

Against the noise-matched baseline the AUROC difference is +0.142 (95% CI +0.058 to +0.222) and the Brier difference is −0.096 (CI −0.143 to −0.048), by participant-cluster bootstrap. At a threshold of 0.5 this twin flags 27 sequences and is right on 20; the tuned baseline flags 31 and is right on 11.

The UOM cohort has 7 low events in 56 sequences, too few to say anything, and no claim rests on it.

### Whole-day calibration

Mean absolute error of each person's time in range, averaged over people:

| Cohort | Fitted days | Held-out days | Held-out bias |
| --- | --- | --- | --- |
| T1D-UOM (n = 9) | 1.2 pp | 8.2 pp | +1.0 |
| HUPA-UCM (n = 18) | 1.3 pp | 19.3 pp | +9.2 |
| HUPA-UCM, v2 priors | 1.8 pp | 16.6 pp | +7.2 |

On days it was fitted to, the twin reproduces time in range to about a percentage point. Forecasting a day it has never seen is much harder, because it does not know what the person will eat. HUPA records are short (4–10 fitted days), which leaves parameters loosely determined and makes the twin predict a calmer day than the person actually has.

### Where it loses

On T1DSim_AI's own bundled example participant, their twin wins: RMSE 24.0 against 29.3 (paired p = 0.013). That person is 94.5% in range with smooth glucose, which suits a learned model.

Two things that did not work, kept here so nobody repeats them:

- **Assimilating the recent CGM at forecast start.** Fitting the disturbance over the three hours before the forecast tracks that window well (RMSE 32 → 7 mg/dL) but does not improve the forecast: +1.3 mg/dL on held-out UOM, nothing on HUPA. These datasets have little usable short-horizon momentum, and simple trend extrapolation loses to persistence even at 15 minutes. The code is in `t1d_twin/forecast.py` and off by default.
- **Risk-space fitting.** Weighting residuals by glycaemic risk, so low-range errors count more, gave no measurable gain.

### The v2 priors

`v2` means two additions: a 24-hour insulin sensitivity rhythm, and population priors re-estimated from other people's fits, leave-one-out. Real fits move the hand-set priors consistently: insulin absorbs faster than the base patients (+0.45 in log units), carbs absorb faster (+0.27), carb effect is higher (+0.19), EGP is lower (−0.19).

They help where records are short and hurt slightly where they are long. On HUPA they improve everything. On UOM, where people have two months of data, sequence RMSE gets worse (39.0 → 41.8), because a person with that much data is better served by their own. Weighting the prior by record length is the obvious fix and is not something to tune on these test sets.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11 or newer, and simglucose for the virtual-patient parameter table, which is read at runtime rather than copied into this repo.

## Use

Fit a person from a folder of daily records:

```bash
python scripts/twin_fit.py --records path/to/raw_days --person-id alice --settings therapy_settings.json --holdout-days 5 --out artifacts/alice/twin.json
```

Check it against days it never saw:

```bash
python scripts/twin_validate.py --twin artifacts/alice/twin.json --records path/to/raw_days --out artifacts/alice/validation.json
```

Run a counterfactual settings experiment:

```bash
python scripts/twin_experiment.py --twin artifacts/alice/twin.json --records path/to/raw_days \
    --arm current:1,1,1 --arm cr_x0.9:0.9,1,1 --arm isf_x0.9:1,0.9,1 \
    --out artifacts/alice/experiment.json
```

Arms are `name:carb_ratio_mult,isf_mult,basal_mult`, and the first one is the paired reference. Pass several `--twin`/`--records` pairs with `--synthetic-people 256` to draw a population from all the fitted twins and run the same arms across it.

With several people fitted, later fits can borrow from earlier ones:

```bash
python scripts/twin_fit.py --records path/to/raw_days --person-id bob --population-fits "artifacts/*/twin.json" --out artifacts/bob/twin.json
```

`scripts/twin_recovery.py` fits a synthetic person whose true parameters are known, which is the quickest way to see what the fitter can and cannot recover.

### Input format

Records are one JSON file per local day (`insite.raw_day.v1`), holding dense 5-minute streams (CGM, basal rate, heart rate, sleep stage, stress) and event lists (boluses, carbs, temp basals, exercise, site changes). `t1d_twin/data.py` documents the shape and handles the awkward parts: DST, hour-mean CGM, 15-minute sensors, missing insulin days, blank meal logs. Converters for two public datasets are included:

```bash
python scripts/twin_convert_t1d_uom.py --root path/to/t1d-uom --pid 2307 --out artifacts/uom/2307
python scripts/twin_convert_hupa.py --root path/to/hupa_ucm --pid HUPA0001P --out artifacts/hupa/HUPA0001P
```

## Reproducing the benchmark

T1DSim_AI is not included here. Clone it separately into `artifacts/external/T1DSim_AI`, and install it in its own Python 3.10 environment, because it pins torch 1.13:

```bash
python -m venv .venv-t1dsim-ai && .venv-t1dsim-ai/bin/pip install -e artifacts/external/T1DSim_AI
python scripts/benchmark_t1dsimai_export.py --cohort uom --pids 2301,2307,2308,2309
.venv-t1dsim-ai/bin/python scripts/benchmark_t1dsimai_train.py --cohort uom --pids 2301,2307,2308,2309
python scripts/benchmark_t1dsimai_score.py --cohort uom --pids 2301,2307,2308,2309
python scripts/benchmark_t1dsimai_report.py --cohort uom
```

The report script recomputes every statistic quoted above from the saved score files.

## Caveats

- **Counterfactuals are not validated on real data.** The benchmark shows the twin reproduces glucose given the insulin that was actually delivered. Nothing in it tests the response to a dose that never happened, which is what a carb-ratio sweep asks for. On synthetic people the mean effects of settings changes come out right, but individual parameters are not separately identifiable (insulin sensitivity trades off against absorption speed), so a twin can forecast well while splitting those wrongly. Treat settings experiments as hypotheses.
- **Held-out day forecasts are much weaker than fitted-day fits**, because unannounced meals dominate.
- Two public datasets, 21 people between them, is a small evidence base.

## Tests

```bash
pytest tests/test_twin.py
```

26 tests, about 90 seconds. They cover parity with simglucose, the loader's handling of real-world record damage, context effects, fit serialisation, experiment direction, and the two negative results above.

## Licences and data

This code is Apache 2.0 (see `LICENSE`). The virtual-patient parameters come from simglucose (MIT) and are read from your installed copy.

T1DSim_AI is licensed for non-profit academic research only. It is used here as a benchmark, and neither its code, weights nor example data are redistributed in this repository.

T1D-UOM and HUPA-UCM are public research datasets with their own terms. No participant data is included here.
