# Earth–Saturn Transfer Optimization in Python

Numerical optimization of an interplanetary transfer from **Earth** to the **Saturn orbital radius**, targeting a **Lagrange-point-compatible heliocentric orbit** using the **restricted three-body approximation**.

This project studies and compares two transfer strategies:

- **Case I** — direct escape and final insertion
- **Case II** — lower initial escape energy plus an additional **Earth flyby** after one heliocentric revolution

The goal is simple: **minimize total \(\Delta v\)** while still reaching the target orbit.

---

## Overview

The spacecraft starts in a circular orbit around **Earth**, in the same plane as the Sun–Earth–Saturn system. From there, the trajectory is propagated in a **heliocentric inertial frame**, while continuously accounting for:

- the gravitational attraction of the **Sun**
- the gravitational attraction of **Earth**
- neglecting Saturn's gravity during transfer, as required by the formulation

The destination is not Saturn itself, but a point at the same heliocentric radius as Saturn, compatible with the orbital radius of **L4/L5**.

This is not a patched-conics toy model. The dynamics are integrated numerically under the **restricted three-body formulation**, which makes the transfer analysis more realistic and more interesting.

---

## Mission Case

**Departure planet:** Earth  
**Target orbital radius:** Saturn  
**Language:** Python  
**Focus:** transfer optimization, numerical propagation, and trajectory analysis

---

## Problem Statement

The project compares two mission architectures:

### Case I — Direct Transfer
A first impulsive maneuver provides enough energy for escape and transfer toward Saturn’s orbital radius.  
Once the spacecraft reaches the target heliocentric distance, a second impulsive maneuver is applied to match the velocity of the destination orbit.

\[
\Delta v_{\text{tot}}^{(I)} = |\Delta v_{\text{ign}}^{(I)}| + |\Delta v_{\text{fin}}^{(I)}|
\]

### Case II — Transfer with Earth Flyby
The first impulsive maneuver is smaller than in Case I, but still enough to escape Earth.  
After one revolution around the Sun, the spacecraft returns near Earth and performs a **gravitational assist** during a second pass through Earth’s sphere of influence. A final insertion maneuver is then applied at the target orbit.

\[
\Delta v_{\text{tot}}^{(II)} = |\Delta v_{\text{ign}}^{(II)}| + |\Delta v_{\text{fin}}^{(II)}|
\]

### Main Objective
Determine which strategy gives the **lowest total \(\Delta v\)** for the **Earth–Saturn** mission scenario.

---

## Physical Model

The spacecraft motion is modeled in the **restricted three-body approximation** in a heliocentric inertial frame.

State vector:

\[
Y = [x, y, v_x, v_y]
\]

The equations of motion are:

\[
\dot{\mathbf{r}} = \mathbf{v}
\]

\[
\dot{\mathbf{v}} =
-\mu \frac{\mathbf{r}}{r^3}
-\mu_A
\left(
\frac{\mathbf{r} - \mathbf{R}(t)}{|\mathbf{r} - \mathbf{R}(t)|^3}
+
\frac{\mathbf{R}(t)}{R(t)^3}
\right)
\]

where:

- \(\mu\) is the gravitational parameter of the **Sun**
- \(\mu_A\) is the gravitational parameter of **Earth**
- \(\mathbf{R}(t)\) is the heliocentric position of Earth, approximated as a circular orbit

The planetary orbit is modeled as:

\[
\mathbf{R}(t) = R_A
\left[
\cos\left(\frac{2\pi t}{T_A}\right)\mathbf{i}
+
\sin\left(\frac{2\pi t}{T_A}\right)\mathbf{j}
\right]
\]

This formulation keeps the dynamics continuous and avoids switching between local and heliocentric conics.

---

## Initial Conditions

The spacecraft starts in a circular orbit around Earth at:

\[
\rho_0 = 1.1 \times R_{\text{Earth}}
\]

The initial state is defined by two optimization variables:

- `theta` — angular position in the initial Earth-centered circular orbit
- `dv_ign` — magnitude of the first impulsive maneuver

For the Earth-to-Saturn transfer, the useful scan range is chosen for an **outer-planet transfer**, so `theta` is typically explored in:

\[
-\frac{\pi}{2} < \theta < 0
\]

The initial ignition must satisfy the escape condition:

\[
\Delta v_{\text{ign}} > (\sqrt{2} - 1)\sqrt{\frac{\mu_A}{\rho_0}}
\]

---

## What This Repository Does

This repository provides a Python workflow to:

- define the **Earth–Saturn** physical parameters
- propagate trajectories numerically with `solve_ivp`
- estimate first-guess transfer parameters from **Hohmann-based analytical approximations**
- scan families of initial conditions
- detect when the trajectory first reaches **Saturn’s heliocentric radius**
- compute the required final insertion maneuver
- evaluate total mission cost in terms of **\(\Delta v\)**
- refine the search around promising solutions
- compare **Case I** and **Case II**
- generate plots and trajectory visualizations

---

## Methodology

### 1. Analytical First Guess
Before brute-force scanning, analytical estimates are used to define meaningful search ranges:

- Hohmann-like estimate for `dv_ign`
- Hohmann-like estimate for `dv_fin`
- flyby-compatible transfer time estimate
- estimated launch angle from hyperbolic deflection geometry

These values are not the final answer. They are just a smart way to avoid scanning nonsense.

### 2. Numerical Propagation
Each candidate pair `(theta, dv_ign)` is integrated using a numerical ODE solver in Python.

### 3. Target Detection
For every propagated trajectory, the code checks whether the spacecraft reaches:

\[
|\mathbf{r}(t)| = R_B
\]

where \(R_B\) is Saturn’s mean heliocentric orbital radius.

If the trajectory never reaches that distance, it is discarded.

### 4. Final Insertion Cost
At the first valid crossing of Saturn’s orbital radius, the final velocity correction is computed to match the target circular heliocentric motion:

\[
\Delta \mathbf{v}_{\text{fin}} = \mathbf{V}_L - \mathbf{v}(t_{\text{fin}})
\]

### 5. Optimization
The best trajectory is the one with minimum total cost:

\[
\Delta v_{\text{tot}} = |\Delta v_{\text{ign}}| + |\Delta v_{\text{fin}}|
\]

### 6. Refinement
After a coarse scan, the search is repeated with tighter parameter ranges around the best candidate.

---
