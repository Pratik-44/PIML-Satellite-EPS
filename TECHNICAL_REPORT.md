# TECHNICAL REPORT

# Physics-Informed Machine Learning for Satellite Electrical Power System Analysis and Prediction

---

# 1. Introduction

Modern satellites rely on Electrical Power Systems (EPS) to ensure reliable operation of onboard subsystems, payloads, communication units, attitude control systems, and thermal management equipment. The EPS continuously experiences changing operating conditions due to orbital transitions, eclipse periods, payload activation, heater operation, battery charging/discharging cycles, and varying subsystem loads.

Predicting the behavior of critical EPS parameters such as voltage and power consumption can assist in system monitoring, anomaly detection, operational planning, and future onboard intelligent decision-making systems.

This project investigates the application of Machine Learning (ML), Deep Learning (DL), and Physics-Informed Machine Learning (PIML) techniques for satellite telemetry forecasting and power-system analysis. Multiple deep learning architectures were evaluated under different datasets and operating conditions to understand their predictive capability, robustness, and generalization characteristics.

---

# 2. Problem Statement

The primary objective of this work is to develop predictive models capable of estimating:

* Spacecraft bus voltage
* Total power consumption

using telemetry measurements collected from the satellite Electrical Power System.

The study aims to answer the following research questions:

1. Which machine learning architectures are most effective for satellite EPS telemetry prediction?
2. How do temporal models compare against feedforward architectures?
3. Can models generalize across different telemetry datasets?
4. What operational factors contribute most strongly to prediction errors?
5. How do eclipse conditions, payload operation, and heater activity influence spacecraft power behavior?
6. Can physics-informed concepts improve prediction quality and model interpretability?

---

# 3. Dataset Description

The datasets used in this project consist of spacecraft Electrical Power System telemetry measurements collected under varying operational conditions.

## Input Features

The following telemetry parameters were used as model inputs:

* Current
* Heater Power
* Load Current
* Eclipse Flag
* Payload Status
* AOCE Mode

These parameters collectively describe the operational state of the spacecraft power subsystem.

## Target Variables

Two prediction targets were considered:

### Voltage Prediction

Prediction of spacecraft bus voltage.

### Power Prediction

Prediction of total spacecraft power consumption.

## Operational Conditions

The datasets contain telemetry spanning:

* Eclipse periods
* Sunlight periods
* Payload activation events
* Heater operation cycles
* Dynamic load variations
* Different mission operating conditions

---

# 4. Research Methodology

The project follows a progressive experimental framework consisting of multiple model families, validation strategies, and analysis stages.

## Model Categories

### Feedforward Neural Networks

#### Dense Neural Network (DNN)

A baseline neural network architecture used to establish reference performance.

#### Deep Neural Network (DeepNN)

A deeper architecture capable of learning more complex nonlinear relationships within telemetry data.

---

### Sequential Models

#### Long Short-Term Memory (LSTM)

LSTM models were investigated due to their ability to capture temporal dependencies and long-term sequential behavior.

#### Gated Recurrent Unit (GRU)

GRU architectures were evaluated as computationally efficient alternatives to LSTM networks.

---

### Attention-Based Models

#### Transformer

Transformer models were implemented to investigate whether attention mechanisms can improve telemetry forecasting and cross-dataset generalization.

---

### Generative Models

#### Generative Adversarial Networks (GAN)

GAN-based regression frameworks were explored to determine whether adversarial learning can improve telemetry prediction quality.

---

### Physics-Informed Learning

Physics-informed concepts were incorporated to align model behavior with spacecraft power-system characteristics.

The objective was to encourage physically meaningful predictions while maintaining predictive accuracy.

---

# 5. Experimental Design

A series of experiments were conducted to evaluate model behavior under different conditions.

---

## Experiment 1: Neural Network Benchmarking

Three neural architectures were compared:

* DNN
* DeepNN
* LSTM

Evaluation metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)

Both voltage and power prediction tasks were considered.

---

## Experiment 2: Time-Wise Data Splitting

Instead of random train-test splitting, a chronological split was performed.

This approach:

* Reduces data leakage
* Better represents real deployment scenarios
* Evaluates temporal generalization

---

## Experiment 3: Operational Behavior Analysis

Satellite power-system behavior was studied under different operating conditions.

Analyses included:

### Eclipse Analysis

Investigation of spacecraft behavior during eclipse and sunlight transitions.

### Heater Influence Analysis

Evaluation of heater activity and its effect on power consumption.

### Payload Analysis

Assessment of payload operation and corresponding power-system impact.

### Power Consistency Analysis

Investigation of long-term power behavior and stability.

---

## Experiment 4: Cross-Dataset Validation

Cross-dataset experiments were performed to evaluate model robustness under distribution shift.

Training and testing were conducted using different telemetry datasets.

The objective was to determine whether learned relationships transfer across operating environments.

---

## Experiment 5: Advanced Architecture Evaluation

Additional experiments compared:

* GRU
* Transformer
* GAN

These architectures were evaluated across multiple datasets to identify situations where temporal modeling, attention mechanisms, or adversarial learning become beneficial.

---

## Experiment 6: Feature Error Analysis

Detailed model explainability studies were conducted.

Techniques included:

* Feature-error correlation analysis
* Error distribution analysis
* Error bin analysis
* Operating condition sensitivity studies

This helped identify which telemetry features contribute most strongly to prediction errors.

---

## Experiment 7: Model Improvement Studies

Several architectural modifications were explored:

* Increased model capacity
* Learning-rate scheduling
* Dropout regularization
* Improved training procedures
* Hybrid GAN loss functions

The goal was to identify improvements that enhance predictive performance and generalization.

---

# 6. Results and Discussion

## Voltage Prediction

Voltage prediction consistently emerged as the more challenging task.

The primary reasons include:

* Extremely low signal variance
* Sensitivity to distribution shifts
* Complex battery regulation behavior
* Power bus stabilization effects

Even small prediction deviations significantly impact R² because voltage values remain within a narrow operating range.

### Key Observation

Temporal models such as LSTM and GRU frequently outperform feedforward architectures when temporal information is present.

This suggests that voltage dynamics contain meaningful sequential dependencies.

---

## Power Prediction

Power prediction generally achieved substantially higher accuracy.

Reasons include:

* Greater signal variability
* Strong direct relationships with telemetry features
* Immediate dependence on spacecraft operational state

Power consumption is strongly influenced by:

* Current
* Load Current
* Heater Power
* Payload Status

These direct relationships make power prediction more suitable for feedforward architectures.

### Key Observation

DeepNN frequently achieved the strongest power-prediction performance, indicating that nonlinear instantaneous relationships dominate power behavior.

---

## Cross-Dataset Generalization

Cross-dataset validation revealed important insights regarding model robustness.

### Voltage

Voltage prediction often deteriorates significantly under distribution shift.

Models trained on one telemetry dataset frequently struggle to generalize to another.

### Power

Power prediction exhibits substantially better transferability.

Models frequently maintain strong performance when tested on unseen datasets.

This indicates that power-related relationships are more stable across operating conditions than voltage dynamics.

---

## GRU vs Transformer vs GAN

### GRU

GRU models perform strongly when temporal dependencies are present.

They consistently achieve stable and reliable results across multiple datasets.

### Transformer

Transformers demonstrate strong performance and occasionally outperform recurrent architectures.

Attention mechanisms improve the ability to capture global telemetry relationships.

### GAN

GAN-based regression proved significantly more challenging.

Initial implementations performed poorly.

However, architectural refinements and hybrid loss functions produced substantial improvements, demonstrating that GAN performance is highly dependent on training strategy.

---

# 7. Feature Error Analysis

Feature-error correlation studies revealed several recurring patterns.

## Load Current

Load current consistently exhibits the strongest relationship with prediction error.

Higher load conditions frequently correspond to larger prediction deviations.

## Current

Current also contributes significantly to model error, particularly during dynamic operating conditions.

## Heater Power

Heater activity influences prediction behavior but generally contributes less error than load-related variables.

## Eclipse and Payload Status

These features influence operational behavior but typically demonstrate weaker direct relationships with prediction error.

---

# 8. Key Findings

The project produced several important observations:

1. Voltage prediction is substantially more difficult than power prediction.

2. Temporal models outperform feedforward models when meaningful temporal dependencies exist.

3. Power prediction is largely governed by instantaneous spacecraft operating conditions.

4. Cross-dataset validation is significantly more challenging than conventional train-test evaluation.

5. Distribution shift severely impacts voltage prediction performance.

6. Load current is one of the strongest contributors to prediction error.

7. Transformer architectures can provide strong generalization capability.

8. GAN architectures require substantial modification before becoming useful for telemetry regression.

9. Model selection should depend on the physical characteristics of the target variable rather than architecture complexity alone.

10. Operational behavior analysis provides valuable insight beyond traditional prediction metrics.

---

# 9. Limitations

Several limitations were identified during the study.

### Dataset Shift

Performance may degrade when telemetry distributions differ substantially between datasets.

### Extreme Operating Conditions

Rare operating regimes remain challenging for many models.

### Voltage Sensitivity

Small voltage deviations can significantly impact evaluation metrics due to low signal variance.

### Physics Integration

The current work incorporates physics-informed concepts but does not yet implement full Physics-Informed Neural Networks (PINNs).

---

# 10. Future Work

Potential future research directions include:

* Physics-Informed Neural Networks (PINNs)
* Hybrid physics-data driven architectures
* Explainable AI for spacecraft telemetry prediction
* Transfer learning across satellites
* Multi-target telemetry forecasting
* Satellite anomaly detection systems
* Real-time onboard predictive analytics
* Foundation models for spacecraft telemetry

---

# 11. Conclusion

This project presents a comprehensive study of machine learning, deep learning, and physics-informed approaches for satellite Electrical Power System telemetry analysis and prediction.

Through extensive experimentation involving DNNs, DeepNNs, LSTMs, GRUs, Transformers, GANs, cross-dataset validation, operational behavior analysis, and feature-error investigations, the study demonstrates both the opportunities and challenges associated with spacecraft telemetry forecasting.

The results show that no single architecture dominates all scenarios. Instead, model effectiveness depends strongly on target characteristics, temporal dependencies, operating conditions, and dataset distribution. The findings highlight the importance of combining predictive modeling with domain-specific analysis to better understand spacecraft power-system behavior and improve future telemetry forecasting systems.
