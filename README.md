# Federated Remaining Useful Life Prediction Under Non-IID Industrial Data

A reusable Federated Learning (FL) framework for **Remaining Useful Life (RUL) prediction** under realistic **non-IID industrial data distributions**, evaluated using NASA's C-MAPSS turbofan engine degradation datasets.

The central objective is not simply to train an RUL model. It is to investigate **how data heterogeneity across industrial clients affects Federated Learning**, and whether FL algorithms designed to handle client drift can maintain performance as the degree of non-IIDness increases.

---

## Overview

In real industrial environments, machine data is rarely identically distributed.

Different factories, machines, operating regimes, maintenance histories, and sensor environments can produce substantially different local datasets. Sending all raw data to a central server may also be undesirable because of privacy, bandwidth, ownership, or regulatory constraints.

Federated Learning addresses this by allowing multiple clients to collaboratively train a global model without directly sharing their raw data.

However, conventional FL experiments often assume that clients have similar data distributions.

This project investigates what happens when that assumption breaks.

### Research Question

> **How does increasing non-IIDness in industrial RUL data affect Federated Learning performance, convergence, and client-level consistency, and can FedProx mitigate the resulting degradation compared with FedAvg?**

---

# Project Goals

The project has five main goals:

1. Build an end-to-end FL pipeline for predictive maintenance.
2. Establish centralized, local-only, and federated baselines.
3. Create controlled **IID and non-IID client distributions** from C-MAPSS.
4. Compare **FedAvg and FedProx** under increasing levels of heterogeneity.
5. Evaluate not only global accuracy but also **client-level performance, convergence, and communication cost**.

The resulting pipeline should be reusable for other predictive-maintenance datasets and FL experiments.

---

# Dataset

## NASA C-MAPSS

The project uses the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** turbofan engine degradation datasets.

The available subsets are:

| Dataset | Training Engines | General Characteristics |
|---|---:|---|
| FD001 | 100 | Single operating condition, single fault mode |
| FD002 | 260 | Multiple operating conditions |
| FD003 | 100 | Single operating condition, multiple fault modes |
| FD004 | 248 | Multiple operating conditions, multiple fault modes |

### Primary Dataset

**FD004** is used as the primary experimental dataset because it provides the most heterogeneous setting among the four subsets:

- Multiple operating conditions
- Multiple fault modes
- 248 training engines
- Significant variation between engine trajectories

FD001–FD003 can subsequently be used to test whether the conclusions generalize across different degradation scenarios.

---

# Problem Formulation

For each engine, sensor measurements are recorded over multiple operating
