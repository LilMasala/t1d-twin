# Personal type 1 diabetes twins

This repo fits a UVA/Padova glucose simulator to one person's CGM, insulin, and context data.

Once fitted, the simulator can be rerun under different pump settings: change a carb ratio, ISF, or basal rate and simulate the same day again.

The important part is that the fitted variables are physiological quantities rather than arbitrary network weights. The model learns things like insulin sensitivity, absorption speed, endogenous glucose production, and carb response. Everything is written in torch, so those parameters can be fitted directly by gradient descent.

On held-out data, glucose forecasting is roughly comparable to [T1DSim_AI](https://github.com/Mosqueralopez/T1DSim_AI), a neural digital twin published in *Neural Computing and Applications*. The larger difference is in hypoglycaemia prediction, where the mechanistic twin does substantially better on HUPA-UCM.

## What gets fitted

Each person gets their own set of parameters:

| Group | Parameters |
| --- | --- |
| Core physiology | insulin sensitivity, endogenous glucose production, insulin absorption and action speed, carb absorption speed and bioavailability, hypoglycaemic counter-regulation |
| Context | menstrual cycle phase (or an inferred 28-day rhythm), exercise, sleep debt, dawn phenomenon, infusion site age, stress, a 24-hour sensitivity rhythm |
| Response shape | 10 carb and 10 insulin response coefficients in 30-minute bins over 5 hours |
| Per day | insulin sensitivity and EGP drift |
| Per meal | carb-count error and timing error |
| Unexplained | a signed glucose disturbance in 30-minute blocks, with a heavy-tailed prior |
| Noise | CGM residual spread |

The meal-error terms are intentional. Logged carbohydrates are noisy enough that treating them as ground truth causes its own problems.

The model also tries not to equate missing data with absence. Unlogged meals can be inferred from carb-free boluses, food photos, and CGM rises, then fitted as free amounts. If cycle days are unavailable, it can fit a free 28-day rhythm. If site changes are not recorded, it can infer likely changes from pump suspensions.

Fitting happens in two stages. First, a 12-start batched MAP fit with CGM noise fixed. Then stochastic variational inference with a full-rank guide over the global parameters. Both operate in prior-standardised parameter space.

## Benchmark

I compared the twin against T1DSim_AI using its own evaluation setup as closely as possible.

Both models see the same train/held-out split and the same five-hour sequences selected by T1DSim_AI's `SequenceSelection`. T1DSim_AI is trained once with the published recipe and once with a tuned learning rate that performs slightly better.

As a sanity check, the benchmark harness reproduces the published T1DSim_AI example twin exactly: test RMSE 23.96 mg/dL.

### Glucose forecasting

Pooled per-sequence RMSE, mg/dL:

| Model | HUPA-UCM (17 people, 217 seq) | T1D-UOM (4 pump users, 56 seq) |
| --- | --- | --- |
| T1DSim_AI population model | 53.4 | 57.2 |
| T1DSim_AI personal twin | 49.7 | 46.3 |
| T1DSim_AI personal twin, tuned | 48.4 | 45.3 |
| This twin | 48.0 | 39.0 |
| This twin, v2 priors | 47.5 | 41.8 |

Against the tuned personal T1DSim_AI model, the paired differences are small: −0.4 to −0.9 mg/dL on HUPA and −3.5 to −6.3 mg/dL on UOM. None are significant by Wilcoxon test (p = 0.28–0.91).

So I would not claim better glucose forecasting than a fine-tuned neural twin. They are in roughly the same range.

The comparison against T1DSim_AI's unpersonalised population model is significant on both cohorts.

### Hypoglycaemia

Here the difference is larger.

A low event is defined as at least three consecutive readings below 70 mg/dL during the five-hour forecast window.

This twin produces an event probability from a 32-draw posterior ensemble. T1DSim_AI is deterministic, so for the probability-based comparison I also score it after adding the same CGM noise model used here.

HUPA-UCM has 50 low events across 217 sequences:

| Model | AUROC | Brier | Sensitivity | PPV |
| --- | --- | --- | --- | --- |
| T1DSim_AI personal twin, tuned | 0.55 | 0.272 | 0.22 | 0.36 |
| Tuned, plus our CGM noise | 0.60 | 0.252 | 0.22 | 0.33 |
| This twin, v2 priors | 0.74 | 0.156 | 0.40 | 0.74 |

Against the noise-matched baseline, AUROC improves by 0.142 (95% CI 0.058–0.222) and Brier score improves by 0.096 (95% CI 0.048–0.143), using participant-cluster bootstrap.

At a 0.5 threshold, this twin flags 27 sequences and gets 20 right. The tuned baseline flags 31 and gets 11 right.

UOM only contains 7 low events in 56 sequences, which is nowhere near enough for a useful hypoglycaemia comparison.

### Whole-day behaviour

Another useful check is whether the fitted twin reproduces each person's overall time in range.

Mean absolute error in time in range, averaged over people:

| Cohort | Fitted days | Held-out days | Held-out bias |
| --- | --- | --- | --- |
| T1D-UOM (n = 9) | 1.2 pp | 8.2 pp | +1.0 |
| HUPA-UCM (n = 18) | 1.3 pp | 19.3 pp | +9.2 |
| HUPA-UCM, v2 priors | 1.8 pp | 16.6 pp | +7.2 |

On days used for fitting, time in range is usually reproduced within about a percentage point.

Held-out days are much harder. The twin knows the person's fitted physiology; it does not know what they are going to eat tomorrow.

That is especially visible in HUPA, where most people only have 4–10 fitted days. Several parameters remain weakly determined, and the model tends to produce quieter days than the real person actually had.

## Where it loses

T1DSim_AI wins clearly on its bundled example participant: 24.0 vs 29.3 mg/dL RMSE, paired p = 0.013.

That participant spends 94.5% of the time in range and has unusually smooth glucose, which is a very friendly setting for the learned model.

I also tried two ideas that sounded useful and were not.

**Assimilating recent CGM before the forecast.**  
The model can fit a disturbance over the previous three hours extremely well: RMSE drops from 32 to 7 mg/dL. It just does not help the future. Held-out UOM gets 1.3 mg/dL worse and HUPA is essentially unchanged.

These datasets seem to contain surprisingly little usable short-horizon momentum. Even simple trend extrapolation loses to persistence at 15 minutes.

The implementation is still in `t1d_twin/forecast.py`, but it is disabled by default.

**Risk-space fitting.**  
I also tried weighting residuals by glycaemic risk so that errors in the low range mattered more during fitting. It produced no measurable improvement.

## v2 priors

`v2` adds two things:

- a 24-hour insulin-sensitivity rhythm
- population priors estimated leave-one-out from the other fitted participants

The real fits move several of the original hand-set priors in consistent directions. Relative to the base virtual patients, insulin absorption is faster (+0.45 log units), carb absorption is faster (+0.27), carb effect is higher (+0.19), and EGP is lower (−0.19).

The priors help most when data are scarce.

On HUPA, they improve the results. On UOM, where participants have roughly two months of data, sequence RMSE actually gets worse: 39.0 → 41.8 mg/dL. With that much individual data, forcing someone toward the population is not especially helpful.

The obvious next version is to weaken the population prior as record length grows. I have not tuned that rule against these test sets.

## Pump-setting experiments

The benchmark above only asks whether the twin reproduces glucose under insulin that was actually delivered. A settings experiment asks something harder: what would a dose that never happened have done?

I ran two checks on that.

### Absolute dose response

`scripts/twin_dose_response.py` gives a fitted twin one extra unit of insulin, or ten extra grams of carbs, at quiet moments and measures what happens. That gives an implied ISF and an implied carb ratio for the twin, which can be compared with how the person actually doses.

The first result was bad. Across the 17 benchmarked twins, the implied carb ratio correlated −0.12 with the grams per unit each person actually bolused, and implied ISF was about a third of what the 1800 rule suggests.

The cause is that nearly every meal comes with a bolus. A CGM trace shows the combined effect of the meal and the insulin, but not how strong each one is on its own, and the fitted disturbance term soaks up whatever is left. The ten base virtual patients differ by about 3× in carb ratio but by roughly 1% in how well they fit a given person.

`t1d_twin/dosing.py` uses the person's dosing to fill that gap. Among base patients that fit about as well as the best one, it picks the one whose carb ratio is closest to the person's, then re-centres the insulin-sensitivity prior.

On four refitted people that raised the correlation to +0.60 without hurting forecast accuracy. The twins still sit about 40% below the person's own ratio, though. Putting the ratio directly into the fitting objective did not close the gap, because the CGM data pull the other way. Part of the gap is that people give 20–30% of their bolus insulin as corrections outside meals. The rest I can't explain yet.

So I would not read absolute recommendations off these twins, like "your carb ratio should be 12".

### Relative effects

The more useful question is whether the twin gets the relative picture right: which setting matters, and in which direction.

`scripts/twin_effect_robustness.py` checks that by running the same arms on several fits of one person. The fits disagree about absolute dose response (the original fit, the dosing-informed prior, and the dosing term in the objective). If their conclusions agree anyway, the conclusions don't depend on the part the data can't pin down.

The arms scale what was actually delivered. A carb ratio of ×0.8 makes each recorded meal bolus 25% larger, ISF ×0.8 does the same to correction boluses, and basal is scaled directly. The unchanged arm is the recorded day itself.

For open-loop pumps like these, that is the literal counterfactual. For a closed loop, it leaves out how the algorithm would have reacted.

I first tried driving the arms through a generic basal-bolus controller instead (`--mode controller`). Even its unchanged arm drifted away from the real day, and some arms collapsed in ways no 20% settings change would cause.

Ten fitted days, 16 posterior samples, carb ratio, ISF and basal each at ×0.8 and ×1.2:

| Person | Time below 70 | Time above 180 | Time in range |
| --- | --- | --- | --- |
| UOM 2307 | same direction, ρ 1.00 | same direction, ρ ≥ 0.94 | same direction, ρ ≥ 0.89 |
| UOM 2309 | same direction, ρ 1.00 | same direction, ρ 1.00 | **direction differs**, ρ 0.43 |
| HUPA0001P | same direction, ρ 1.00 | same direction, ρ 1.00 | same direction, ρ 1.00 |
| HUPA0028P | same direction, ρ 1.00 | same direction, ρ 1.00 | same direction, ρ ≥ 0.94 |

ρ is the rank correlation of the arms' effects between fits.

For lows and highs, every fit of every person agrees on which way each setting moves things, and on the order of the effects.

Time in range is where they can disagree. For 2309, making meal boluses 25% larger adds 5–7.5 points below 70 and removes 4.5–5.7 points above 180 in every fit. Those nearly cancel, so whether time in range goes up or down depends on the fit, which is exactly the part the data can't identify. I would report lows and highs separately rather than time in range alone.

The results are also person-specific in ways that match the records. For HUPA0001P, basal is by far the biggest lever (×0.8 costs 17–22 points of time in range) and ISF barely matters, because only 6% of their bolus insulin is corrections. For HUPA0028P it is the reverse: carb ratio dominates and basal hardly moves anything.

A few things to keep in mind:

- Effect sizes differ between fits by up to about 2× (2307, carb ratio ×0.8: +3.1 to +6.0 points below 70), so treat magnitudes as rough.
- An ISF arm can only scale corrections that were actually given, so it reads as zero for someone who never corrects (2309).
- The unchanged arm does not always match the real days. 2309's twins spend 5% of the time below 70 where the person spent 1.5%, so their lows are overstated before any setting changes.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11 or newer is required.

The simulator also uses `simglucose` for its virtual-patient parameter table. Those values are read from the installed package at runtime rather than copied into this repository.

## Use

Fit one person from a folder of daily records:

```bash
python scripts/twin_fit.py --records path/to/raw_days --person-id alice --settings therapy_settings.json --holdout-days 5 --out artifacts/alice/twin.json
```

Check the fitted twin on days it did not see:

```bash
python scripts/twin_validate.py --twin artifacts/alice/twin.json --records path/to/raw_days --out artifacts/alice/validation.json
```

Run a counterfactual pump-settings experiment:

```bash
python scripts/twin_experiment.py --twin artifacts/alice/twin.json --records path/to/raw_days \
    --arm current:1,1,1 --arm cr_x0.9:0.9,1,1 --arm isf_x0.9:1,0.9,1 \
    --out artifacts/alice/experiment.json
```

Each arm is:

```text
name:carb_ratio_mult,isf_mult,basal_mult
```

The first arm is used as the paired reference.

You can also pass several `--twin` / `--records` pairs with `--synthetic-people 256`. This samples a synthetic population from the fitted twins and applies the same intervention arms across it.

Once several people have been fitted, a new fit can use them to build its population prior:

```bash
python scripts/twin_fit.py --records path/to/raw_days --person-id bob --population-fits "artifacts/*/twin.json" --out artifacts/bob/twin.json
```

For settings questions, I would run the arms on the recorded delivery, across more than one fit, and look at lows and highs separately:

```bash
python scripts/twin_effect_robustness.py --records path/to/raw_days \
    --twin base=artifacts/alice/twin.json --twin dosing=artifacts/alice/twin_dosing.json --out artifacts/alice/effects.json
```

For parameter recovery, `scripts/twin_recovery.py` creates and refits a synthetic person whose true parameters are known. That is the easiest way to see which physiological quantities the model can actually identify.

### Input format

The raw format is one JSON file per local day: `insite.raw_day.v1`.

Each file can contain dense five-minute streams such as CGM, basal rate, heart rate, sleep stage, and stress, plus event lists for boluses, carbs, temp basals, exercise, and site changes.

`t1d_twin/data.py` defines the format and handles the less pleasant parts of real diabetes data: DST, hourly-mean CGM, 15-minute sensors, missing insulin days, and empty meal logs.

Converters for both benchmark datasets are included:

```bash
python scripts/twin_convert_t1d_uom.py --root path/to/t1d-uom --pid 2307 --out artifacts/uom/2307
python scripts/twin_convert_hupa.py --root path/to/hupa_ucm --pid HUPA0001P --out artifacts/hupa/HUPA0001P
```

## Reproducing the T1DSim_AI benchmark

T1DSim_AI is not bundled with this repository.

Clone it separately into `artifacts/external/T1DSim_AI` and give it its own Python 3.10 environment, since it pins torch 1.13:

```bash
python -m venv .venv-t1dsim-ai && .venv-t1dsim-ai/bin/pip install -e artifacts/external/T1DSim_AI
python scripts/benchmark_t1dsimai_export.py --cohort uom --pids 2301,2307,2308,2309
.venv-t1dsim-ai/bin/python scripts/benchmark_t1dsimai_train.py --cohort uom --pids 2301,2307,2308,2309
python scripts/benchmark_t1dsimai_score.py --cohort uom --pids 2301,2307,2308,2309
python scripts/benchmark_t1dsimai_report.py --cohort uom
```

The report script recomputes the statistics in this README directly from the saved score files.

## Limitations

The main unresolved question is also the reason this project exists: counterfactuals.

The benchmark tests whether the fitted twin can reproduce glucose when given insulin that was actually delivered. It does **not** tell us whether the twin correctly predicts what would have happened under a dose that never occurred.

Synthetic experiments are encouraging at the population level: changing pump settings moves glucose in the expected direction and the mean treatment effects are recovered reasonably well.

Individual physiological parameters are less clean. Insulin sensitivity and absorption speed, for example, can trade off against each other, and on real data the strength of insulin relative to carbs is barely identified at all (see "Pump-setting experiments"). Two fitted twins can therefore forecast glucose similarly while disagreeing about why it happened.

The relative picture holds up much better than the absolute one. Across fits that disagree about dose response, the direction and ranking of effects on lows and highs stay the same. Absolute carb ratios and ISFs don't, and none of the data here includes a person whose settings actually changed, which is what a real validation would need.

For now, I would treat pump-setting experiments from the model as hypotheses rather than treatment recommendations.

There are two other limits worth keeping in mind. Held-out full-day forecasts are considerably weaker than fits to observed days, largely because future meals are unknown. And the current real-data benchmark is small: two public datasets and 21 people total.

## Tests

```bash
pytest tests/test_twin.py
```

There are currently 30 tests and the suite takes about two minutes.

They cover parity with simglucose, messy real-world input handling, context effects, fit serialisation, intervention direction, the dosing prior, the recorded-delivery experiments, and the two negative experiments above.

## Licences and data

The code in this repository is Apache 2.0; see `LICENSE`.

Virtual-patient parameters come from simglucose (MIT) and are read from the user's installed copy.

T1DSim_AI is licensed for non-profit academic research only. It is used here strictly as a benchmark. Its code, weights, and example data are not redistributed.

T1D-UOM and HUPA-UCM are public research datasets with their own terms. No participant data is included in this repository.
