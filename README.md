# Earth–Saturn Transfer Optimization in Python

Numerical optimization of an interplanetary transfer from **Earth** to the **Saturn orbital radius**, based on the **restricted three-body approximation** and implemented in **Python**.

This project analyzes and compares two transfer strategies:

- **Case I** — direct escape and final insertion
- **Case II** — lower initial escape energy plus an additional **Earth flyby** after one heliocentric revolution

The main objective is to determine which strategy minimizes the **total mission cost in $\Delta v$**.

---

## Overview

The spacecraft starts from a circular orbit around **Earth**, in the same plane as the Sun–Earth–Saturn system. From there, the trajectory is propagated in a **heliocentric inertial frame**, continuously accounting for:

- the gravitational attraction of the **Sun**
- the gravitational attraction of **Earth**
- a target orbit located at **Saturn’s heliocentric radius**

Saturn’s gravity is neglected during transfer, following the formulation of the problem.

The destination is not Saturn itself, but a point on a circular heliocentric orbit with the same radius as Saturn, compatible with the orbital radius of **$L_4$ / $L_5$**.

This makes the project a numerical astrodynamics study rather than a simple patched-conics exercise.

---

## Mission Case

- **Departure planet:** Earth
- **Target orbital radius:** Saturn
- **Language:** Python
- **Core topics:** orbital mechanics, trajectory propagation, transfer optimization, numerical integration

---

## Problem Statement

The project compares two mission architectures.

### Case I — Direct Transfer

A first impulsive maneuver provides enough energy for escape and transfer toward Saturn’s orbital radius.

Once the spacecraft reaches the target heliocentric distance, a second impulsive maneuver is applied to match the velocity of the destination orbit.

$$
\Delta v_{\text{tot}}^{(I)} = \left|\Delta v_{\text{ign}}^{(I)}\right| + \left|\Delta v_{\text{fin}}^{(I)}\right|
$$

### Case II — Transfer with Earth Flyby

The first impulsive maneuver is smaller than in Case I, but still sufficient to escape Earth.

After one revolution around the Sun, the spacecraft returns near Earth and performs a **gravitational assist** during a second pass through Earth’s sphere of influence. A final insertion maneuver is then applied at the target orbit.

$$
\Delta v_{\text{tot}}^{(II)} = \left|\Delta v_{\text{ign}}^{(II)}\right| + \left|\Delta v_{\text{fin}}^{(II)}\right|
$$

### Main Objective

Find the trajectory that minimizes total mission cost and compare both mission architectures for the **Earth–Saturn** scenario.

---

## Physical Model

The spacecraft motion is modeled in the **restricted three-body approximation** in a heliocentric inertial frame.

State vector:

$$
\mathbf{Y} = [x,\ y,\ v_x,\ v_y]
$$

Equations of motion:

$$
\dot{\mathbf{r}} = \mathbf{v}
$$

$$
\dot{\mathbf{v}} =
-\frac{\mu}{r^3}\mathbf{r}
-\mu_A \left(
\frac{\mathbf{r} - \mathbf{R}(t)}{\lVert \mathbf{r} - \mathbf{R}(t) \rVert^3}
+
\frac{\mathbf{R}(t)}{R(t)^3}
\right)
$$

where:

- $\mu$ is the gravitational parameter of the **Sun**
- $\mu_A$ is the gravitational parameter of **Earth**
- $\mathbf{R}(t)$ is the heliocentric position of Earth, approximated as a circular orbit

Earth’s heliocentric orbit is modeled as:

$$
\mathbf{R}(t) =
R_A
\begin{bmatrix}
\cos\left(\frac{2\pi t}{T_A}\right) \\
\sin\left(\frac{2\pi t}{T_A}\right)
\end{bmatrix}
$$

This formulation keeps the dynamics continuous and avoids switching between local and heliocentric conics.

---

## Initial Conditions

The spacecraft starts in a circular orbit around Earth at:

$$
\rho_0 = 1.1\,R_{\text{Earth}}
$$

The initial state is defined through two key design variables:

- `theta` — angular position of the spacecraft in the initial Earth-centered parking orbit
- `dv_ign` — magnitude of the initial impulsive maneuver

For the Earth-to-Saturn transfer, the useful search region corresponds to an **outer-planet transfer**, so `theta` is typically explored in:

$$
-\frac{\pi}{2} < \theta < 0
$$

The initial ignition must satisfy the Earth escape condition:

$$
\Delta v_{\text{ign}} > (\sqrt{2} - 1)\sqrt{\frac{\mu_A}{\rho_0}}
$$

---

## Project Goals

This repository is built to:

- model the **Earth–Saturn transfer problem** in Python
- propagate trajectories numerically using the restricted three-body equations
- estimate first guesses from analytical transfer approximations
- scan candidate initial conditions
- detect valid crossings of Saturn’s heliocentric radius
- compute the final insertion maneuver
- evaluate the total $\Delta v$ cost
- refine the search around promising solutions
- compare **Case I** and **Case II**
- visualize trajectories and mission metrics

---

## Methodology

### 1. Analytical First Guess

Before running numerical scans, analytical approximations are used to define meaningful search windows:

- Hohmann-like estimate for initial ignition
- Hohmann-like estimate for final insertion
- approximate transfer time
- launch-angle estimates based on transfer geometry

These values are not final solutions. They are used only to reduce the search space and improve efficiency.

### 2. Numerical Propagation

Each candidate pair `(theta, dv_ign)` is propagated numerically using a Python ODE solver such as `solve_ivp`.

### 3. Target Detection

For each trajectory, the code checks whether the spacecraft reaches Saturn’s heliocentric radius:

$$
\lVert \mathbf{r}(t) \rVert = R_B
$$

where $R_B$ is Saturn’s mean heliocentric orbital radius.

If the trajectory never reaches this distance, it is discarded.

### 4. Final Insertion Cost

At the first valid crossing of Saturn’s orbital radius, the velocity correction needed to match the target circular heliocentric orbit is computed.

$$
\Delta \mathbf{v}_{\text{fin}} = \mathbf{V}_L - \mathbf{v}(t_{\text{fin}})
$$

where:

- $\mathbf{v}(t_{\text{fin}})$ is the propagated spacecraft velocity at arrival
- $\mathbf{V}_L$ is the target circular heliocentric velocity at Saturn’s orbital radius

### 5. Optimization

The best solution is the one that minimizes:

$$
\Delta v_{\text{tot}} = \left|\Delta v_{\text{ign}}\right| + \left|\Delta v_{\text{fin}}\right|
$$

### 6. Refinement

After a coarse scan, the search is repeated over a narrower region around the best candidate in order to obtain a more accurate optimum.
