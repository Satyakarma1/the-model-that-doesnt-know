# OOD Detection Study: Quantitative Results Summary

## 1. Closed-Set Classification Performance
- **Known Test Accuracy**: 92.02%
- **Validation Accuracy**: 92.78%
- **Best Regularization Parameter (C)**: 0.01
- **Expected Calibration Error (ECE)**: 0.77% -> 0.79% (T = 0.946)

## 2. Out-of-Distribution Detection Benchmark
| Detector Method | Evaluation Split | AUROC (Higher is Better) | FPR @ 95% TPR (Lower is Better) | Threshold |
|---|---|:---:|:---:|:---:|
| **MSP** | Near-OOD (CIFAR Held-out) | **0.7182** | **74.56%** | -0.9991 |
| **MSP** | Far-OOD (FashionMNIST) | **0.8677** | **39.76%** | -0.9743 |
| **Energy** | Near-OOD (CIFAR Held-out) | **0.7319** | **73.30%** | -8.1295 |
| **Energy** | Far-OOD (FashionMNIST) | **0.8972** | **33.60%** | -5.2923 |
| **Mahalanobis** | Near-OOD (CIFAR Held-out) | **0.6824** | **80.56%** | 19.7204 |
| **Mahalanobis** | Far-OOD (FashionMNIST) | **0.9408** | **32.06%** | 23.8758 |

## 3. Selective Classification (Risk-Coverage & AURC)
| Detector | AURC (Area Under Risk-Coverage) | Coverage at 90% Accuracy |
|---|:---:|:---:|
| **MSP** | 0.0115 | 100.0% |
| **Energy** | 0.0190 | 100.0% |
| **Mahalanobis** | 0.0717 | 100.0% |

## 4. Attractor Analysis: Per-Class AUROC (Energy Detector)
| Known Class | Near-OOD AUROC | Far-OOD AUROC |
|---|:---:|:---:|
| Airplane | 0.7286 | 0.8945 |
| Automobile | 0.8720 | 0.9597 |
| Bird | 0.6973 | 0.8790 |
| Cat | 0.6789 | 0.8704 |
| Deer | 0.6825 | 0.8822 |

## 5. Failure Analysis: OOD Misclassification at 95% TPR
Distribution of predicted classes when unknown inputs slipped past the detector:
- **Airplane**: 22 inputs
- **Automobile**: 127 inputs
- **Bird**: 0 inputs
- **Cat**: 65 inputs
- **Deer**: 35 inputs