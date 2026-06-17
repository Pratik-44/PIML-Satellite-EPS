# Physics-Informed Machine Learning for Satellite Electrical Power System Analysis and Prediction

Physics-Informed Machine Learning and Deep Learning models for satellite Electrical Power System (EPS) telemetry analysis, voltage prediction, power forecasting, and spacecraft operational behavior studies.

## Overview

This repository presents a research-oriented study on the application of Machine Learning (ML), Deep Learning (DL), and Physics-Informed Machine Learning (PIML) techniques for Satellite Electrical Power System (EPS) telemetry analysis and prediction.

The project investigates the behavior of critical spacecraft power-system parameters under varying operational conditions and evaluates the effectiveness of multiple deep learning architectures for predicting voltage and total power consumption. In addition to predictive modeling, the work explores system behavior during eclipse transitions, heater operation, payload activity, and changing electrical load conditions.

The repository contains implementations, experiments, analyses, visualizations, and comparative studies developed as part of a Research Practice project.

---

## Project Objectives

* Predict spacecraft bus voltage from telemetry measurements.
* Forecast total power consumption of the satellite power subsystem.
* Compare the effectiveness of multiple deep learning architectures.
* Evaluate model generalization across different telemetry datasets.
* Analyze the impact of operational parameters such as eclipse conditions, heater activity, payload operation, and electrical load variations.
* Investigate the applicability of Physics-Informed Machine Learning concepts for spacecraft telemetry prediction.

---

## Dataset and Features

The project utilizes satellite Electrical Power System telemetry data containing operational and power-system measurements.

### Input Features

* Current
* Heater Power
* Load Current
* Eclipse Flag
* Payload Status
* AOCE Mode

### Target Variables

* Voltage
* Total Power

The datasets contain telemetry collected under varying spacecraft operating conditions, including eclipse and sunlight periods, changing payload activity, and dynamic subsystem loads.

---

## Implemented Models

The following models were developed and evaluated throughout the project:

### Feedforward Models

* Dense Neural Network (DNN)
* Deep Neural Network (DeepNN)

### Sequential Models

* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU)

### Attention-Based Models

* Transformer

### Generative Models

* Generative Adversarial Network (GAN)

### Physics-Informed Approaches

* Physics-guided spacecraft power-system prediction frameworks
* Constraint-aware telemetry modeling experiments

---

## Experimental Studies

### 1. Neural Network Benchmarking

A comparative study was conducted using DNN, DeepNN, and LSTM models for voltage and power prediction.

Experiments included:

* Random train-test splits (80-20)
* Time-wise train-test splits
* Comparative evaluation using MAE, RMSE, and R² metrics

### 2. Operational Behavior Analysis

The project investigates spacecraft power-system behavior under different operating conditions:

* Eclipse vs Sunlight operation
* Heater activity analysis
* Payload activity analysis
* Voltage regulation behavior
* Power consumption trends
* Load-current effects

### 3. Cross-Dataset Validation

To evaluate model robustness, models were trained and tested across different telemetry datasets.

The objective was to determine:

* Generalization capability
* Sensitivity to distribution shifts
* Reliability under unseen operating conditions

### 4. Advanced Architecture Evaluation

Additional experiments were performed using:

* GRU
* Transformer
* GAN

These models were evaluated across multiple datasets to understand the effectiveness of recurrent, attention-based, and adversarial approaches for spacecraft telemetry forecasting.

### 5. Feature Error Analysis

Detailed error analysis was performed using:

* Feature-error correlation analysis
* Error-bin analysis
* Load sensitivity analysis
* Operational regime analysis

This helped identify the factors contributing most strongly to prediction errors.

### 6. Model Improvement Studies

Several architectural and training improvements were explored, including:

* Increased model capacity
* Learning-rate scheduling
* Dropout regularization
* Hybrid GAN loss functions
* Enhanced evaluation metrics

---

## Key Findings

### Voltage Prediction

* Voltage prediction exhibits strong sensitivity to distribution shifts between datasets.
* Temporal models such as LSTM and GRU often outperform feedforward models when temporal dependencies are present.
* Low variance in voltage signals makes cross-dataset generalization particularly challenging.

### Power Prediction

* Power prediction generally achieves significantly higher accuracy than voltage prediction.
* Deep feedforward architectures frequently outperform sequence models because power consumption is strongly influenced by instantaneous system states.
* Features such as current, load current, heater activity, and payload status show strong relationships with power consumption.

### Model Comparison

* DeepNN consistently provides strong performance for power prediction tasks.
* LSTM and GRU effectively capture temporal behavior when present in the telemetry.
* Transformer models demonstrate competitive performance on certain datasets.
* GAN-based approaches are less suitable for this regression task, although architectural improvements significantly improved performance.

### Cross-Dataset Generalization

* Distribution shifts between datasets can substantially impact prediction quality.
* Power prediction generalizes more effectively than voltage prediction.
* Robust deployment of telemetry prediction models requires careful consideration of dataset characteristics and operating conditions.

---

## Repository Structure

```text
Actual Code/
│
├── Core implementations
├── Physics-informed experiments
├── Model training pipelines
│
NN/
│
├── Voltage Prediction
├── Power Prediction
├── DNN Experiments
├── DeepNN Experiments
├── LSTM Experiments
│
NN new track/
│
├── GRU
├── Transformer
├── GAN
├── Cross-dataset Studies
│
Reports/
│
├── Analysis Documents
├── Experimental Results
├── Comparative Studies
│
Results/
│
├── Prediction Outputs
├── Evaluation Metrics
├── Visualization Plots
```

---

## Technologies Used

### Programming

* Python

### Machine Learning & Deep Learning

* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn

### Visualization & Analysis

* Matplotlib
* Seaborn
* Statistical Analysis Tools

### Research Areas

* Time-Series Forecasting
* Deep Learning
* Physics-Informed Machine Learning
* Spacecraft Telemetry Analytics
* Electrical Power Systems

---

## Dataset Availability

Some datasets, CSV files, intermediate outputs, and generated artifacts used during experimentation are not included in this public repository due to size limitations and repository management considerations.

The repository focuses on:

* Source code
* Experimental workflows
* Analysis reports
* Visualizations
* Research findings

---

## Future Work

Potential future directions include:

* Advanced Physics-Informed Neural Networks (PINNs)
* Hybrid physics-data driven architectures
* Explainable AI for spacecraft telemetry prediction
* Multi-satellite transfer learning
* Anomaly detection for spacecraft power systems
* Real-time onboard telemetry forecasting

---

## Author

**Pratik Bora**

As a part of Research Practice Project - Satellite Electrical Power System Analytics using Machine Learning, Deep Learning, and Physics-Informed Learning Techniques.
