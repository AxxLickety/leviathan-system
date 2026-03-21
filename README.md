# 🐳 Leviathan — Regime-Aware Downside Risk Overlay

## Overview

Leviathan is a regime-aware risk overlay designed to reduce exposure during structurally adverse macro conditions.

It is **not an alpha model** and does not attempt to predict returns.  
Instead, it functions as a **risk budgeting layer** that helps avoid disproportionate losses when the system becomes fragile.

> **Goal:** improve downside outcomes (left tail, drawdowns) while accepting a controlled tradeoff in upside participation.

---

## Core Idea

Traditional strategies focus on predicting returns:

- momentum → what will keep going up  
- value → what is mispriced  
- mean reversion → what will bounce  

Leviathan takes a different approach:

> **Instead of asking “what will go up?”, it asks “when is it dangerous to stay exposed?”**

The model identifies adverse macro regimes and reduces exposure accordingly.

---

## Model Definition (v1)

Leviathan v1 is intentionally simple and robust:

- **Signal:** macro regime (e.g., restrictive real-rate environment)  
- **Action:** reduce exposure when adverse regime persists  

### Default Policy

> **50% exposure reduction after 2 consecutive quarters of adverse regime**

This avoids reacting to short-term noise while still responding to sustained risk conditions.

---

## Why Not Alpha?

Extensive testing shows:

- Strong and consistent improvement in **downside metrics** (p05, max drawdown)
- **Sharpe improvement is conditional**, not universal
- Performance tradeoff is explicit:
  - less downside
  - less upside participation

Therefore, Leviathan is best understood as:

> **a downside-risk overlay, not a return-generating alpha**

---

## Empirical Findings

Across multiple synthetic environments:

- ✔ Downside improves in the vast majority of cases  
- ✔ Drawdowns are consistently reduced  
- ⚠ Return / Sharpe improvement depends on market regime  
- ⚠ Opportunity cost exists in strong bull environments  

The model is robust across:

- multiple independent markets  
- de-synchronized crash timing  
- noisy / adversarial conditions  
- positive-drift environments  

---

## Policy Variants

Leviathan is **not a single fixed strategy**.  
It is a configurable overlay depending on mandate:

### Conservative (Capital Preservation)
- Full exit in adverse regime  
- Maximum protection  
- High opportunity cost  

### Balanced (Default)
- **50% de-risk + 2-quarter persistence**  
- Strong protection with moderate cost  

### Growth (Return-Seeking)
- Longer persistence or lighter de-risking  
- Minimal interference with upside  
- Limited but targeted protection  

---

## What Leviathan Is (and Is Not)

### ✔ It is:
- a **regime-aware risk overlay**
- a **portfolio-level risk control tool**
- a **way to reduce exposure during fragile conditions**

### ❌ It is not:
- a stock-picking model  
- a return prediction engine  
- a standalone trading strategy  

---

## How to Use

Leviathan is designed to sit **on top of existing strategies**:

Portfolio Return = Base Strategy × Leviathan Overlay


Typical use cases:

- equity or housing exposure  
- macro-sensitive portfolios  
- trend / valuation strategies  

Its role is simple:

> **reduce exposure when staying invested becomes structurally risky**

---

## Key Insight

The project’s central finding is:

> **Downside risk in housing markets is driven primarily by macro regime, not valuation levels like DTI.**

After multiple rounds of testing:

- DTI-based signals did not provide stable incremental value  
- Regime-based filtering remained robust across all environments  

---

## Repository Structure

scripts/
oos_regime_*.py # evaluation pipelines

src/research/path_a/
build_*.py # synthetic data generators

docs/
OOS_*.md # experiment memos
LEVIATHAN_V1.md # final model definition

outputs/
oos_* # experiment results


---

## Summary

Leviathan represents a shift from alpha-seeking to risk control:

- It does not try to outperform in all conditions  
- It focuses on avoiding the worst conditions  

> **A simple rule: reduce exposure when the system becomes structurally dangerous.**

---

## Status

**Leviathan v1 is complete as a research-grade downside-risk overlay.**

### Future Work

- real-world multi-market validation  
- integration with live portfolio strategies  
- adaptive / mandate-aware deployment  
