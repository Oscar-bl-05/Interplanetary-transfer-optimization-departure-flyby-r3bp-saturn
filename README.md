# Earth-Saturn transfer optimization: Case I and Case II

This version includes:

- Case I direct optimization.
- Case I PCA comparison.
- Case II resonant-PCA search following the professor's suggestion.
- Guided Case II search using Earth SOI return, no-impact check, close-flyby altitude window, flyby energy diagnostics and final R_B validation.
- Fuel/energy trade-off reporting for the close-flyby candidates.
- PNG plots saved in `figures/`.

## Main commands

Run both cases with the default useful configuration:

```bash
python3 main.py
```

Run only Case II:

```bash
python3 main.py --case II
```

Run Case II without opening PNGs:

```bash
python3 main.py --case II --no-open-plots
```

Run without plots:

```bash
python3 main.py --case II --skip-plots
```

## Case II default

The automatic Case II search focuses on the useful Earth-Saturn resonance:

```text
n = 1
n_earth = 12
```

The full resonance table from `n_earth = 2` to `12` is still printed, so the professor's requested `1:2` test remains documented. Lower resonances are energetically too weak for Saturn in this setup.

## Close-flyby controls

By default, the optimizer rejects weak SOI-grazing encounters and only accepts a second Earth encounter inside this altitude window:

```text
300 km <= flyby altitude <= 350000 km
```

You can change it from the command line:

```bash
python3 main.py --case II --case-ii-min-altitude 300 --case-ii-max-altitude 100000
```

The default objective is `balanced`:

```text
fuel     -> select the valid close-flyby candidate with minimum dv_tot
energy   -> select the candidate with maximum flyby energy gain / aphelion
balanced -> select the strongest energy-gain candidate within 50 m/s of the fuel-best close flyby
```

Example:

```bash
python3 main.py --case II --case-ii-objective fuel
python3 main.py --case II --case-ii-objective energy
python3 main.py --case II --case-ii-objective balanced
```

## Physical Case II constraints

A Case II candidate is only accepted if:

```text
dv_escape_min < dv_ign_II < dv_ign_I
second Earth SOI pass exists
flyby altitude is inside the requested altitude window
flyby energy gain is positive
trajectory reaches R_B after the flyby
dv_tot = |dv_ign| + |dv_fin| is computed at R_B
```

## Useful diagnostics printed for Case II

The optimized solution prints:

```text
theta
dv_ign
dv_fin
dv_tot
t_SOI_in
t_SOI_out
MED
minimum altitude
energy before flyby
energy after flyby
delta energy
post-flyby osculating aphelion
fuel-best close flyby
energy-best close flyby
```

## Important interpretation

A closer flyby extracts more heliocentric energy, but that does not automatically minimize total delta-v. The final circularization delta-v can increase if the post-flyby velocity direction at R_B is less favorable.

Therefore, the code reports the internal trade-off between:

```text
best close flyby by fuel
best close flyby by energy gain
selected candidate according to the requested objective
```

For the report, this is useful: it shows that a stronger flyby exists and raises the post-flyby aphelion, but the assignment's final comparison must still use total delta-v.

## Longer resonances / fractional resonances

The professor's formula does not require `n = 1`. Longer resonant returns can be tested with `n > 1`:

```text
n*T(a) = n_earth*T_Earth
T(a) = (n_earth/n)*T_Earth
```

This matters for Earth-Saturn because `1:12` is just below Saturn's orbital radius, while `1:13` already needs too much ignition delta-v. Intermediate ratios such as `2:25` give `T(a)=12.5 years`, between `1:12` and `1:13`.

Recommended exploratory command:

```bash
python3 main.py --case both --n-max 2 --n-earth-min 12 --n-earth-max 25 --case-ii-dv-limit-mode optimized --case-ii-max-altitude 500000 --case-ii-objective fuel --no-open-plots
```

For a heavier final run, increase the grid and step controls:

```bash
python3 main.py --case both --n-max 2 --n-earth-min 12 --n-earth-max 25 --case-ii-grid-theta 25 --case-ii-grid-dv 16 --case-ii-refines 1 --case-ii-nstep 1200 --case-ii-dv-limit-mode optimized --case-ii-max-altitude 500000 --no-open-plots
```

Use `--case-ii-dv-limit-mode optimized` when running both cases. This uses the numerical Case-I ignition delta-v as the upper bound for Case II, instead of the stricter PCA estimate.

## Best Case-II solution found in this version

Using the default Case-II configuration:

```bash
python3 main.py --case II --skip-plots --no-open-plots
```

or, for the full comparison:

```bash
python3 main.py --case both --no-open-plots
```

the best validated close-flyby solution currently found is:

```text
resonance: n = 1, n_earth = 12
theta = -0.37405 rad
dv_ign_II = 7.269035 km/s
dv_fin_II = 5.442374 km/s
dv_tot_II = 12.711409 km/s
flyby altitude = 53451 km
delta energy across flyby = +0.4256 km^2/s^2
post-flyby osculating aphelion = 1458.3 Gm
t_fin = 18.045 years
```

Compared with the numerical Case-I optimum:

```text
dv_ign_I = 7.322993 km/s
dv_fin_I = 5.489269 km/s
dv_tot_I = 12.812261 km/s
```

the Case-II saving is:

```text
dv_tot_I - dv_tot_II = 0.100852 km/s = 100.9 m/s
relative saving = 0.787 %
```

This is the most useful result in the current code because it is not a weak SOI-grazing encounter: the flyby altitude is around 53,000 km, it gives positive heliocentric energy gain, it reaches Saturn's orbital radius and it reduces the total delta-v.

## Final-output commands

Generate the full final set of outputs:

```bash
python main.py --case both --no-open-plots
```

Generate everything except GIF animations:

```bash
python main.py --case both --no-open-plots --skip-animations
```

Only check numerical results, without generating figures:

```bash
python main.py --case both --skip-plots --no-open-plots
```

The final checklist is in `FINAL_CHECKLIST.md`.
