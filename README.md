# MLOps pipeline for HiggsML Uncertainty Challenge

## Overview

This repository develops an **MLOps-oriented pipeline** for experiments
related to the **HiggsML Uncertainty Challenge**.

The goal of the project is to build a **reproducible and well-structured
machine learning workflow** for a scientific analysis scenario inspired
by high‑energy physics.\
In particular, the pipeline aims to support experimentation around:

-   classification of signal vs background collision events
-   estimation of signal strength uncertainty
-   construction and evaluation of conformal prediction intervals
-   systematic comparison of models and experimental configurations

The emphasis of this repository is **methodology and reproducibility**
rather than raw model performance.\
The project focuses on demonstrating how modern **MLOps practices** can
be applied to a scientific machine learning workflow.

The pipeline will progressively introduce components such as:

-   configuration‑driven experiments
-   artifact and experiment tracking
-   reproducible data splits and pseudo‑experiments
-   structured evaluation outputs
-   lightweight model lineage and registry concepts

This repository is intended primarily as a **research and
experimentation environment**, not as a production system.

------------------------------------------------------------------------

## Relation to `conformal-predictions`

This project is built as an extension of the following repository:

**conformal-predictions**

https://github.com/clissa/conformal-predictions

That repository provides the initial **proof‑of‑concept implementation
of conformal prediction methods** used in this project.

The present repository builds on that work by:

-   extending the experimental workflow
-   restructuring parts of the codebase
-   adding reproducibility and experiment management features
-   integrating a lightweight MLOps-oriented structure

The scientific logic and early experimentation framework originate from
the original repository, while this project focuses on evolving the
workflow into a **more structured and reproducible experimentation
pipeline**.

------------------------------------------------------------------------

## Status

This repository is currently in **active development**.

The initial goal is to incrementally transform the existing
proof‑of‑concept into a **clear, reproducible experimentation pipeline**
suitable for demonstrating MLOps practices in a scientific machine
learning context.
