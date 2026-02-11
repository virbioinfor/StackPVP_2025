# # StackPVP_2025 
StackPVP: A Stacked Ensemble Classification Framework for Predicting Phage Virion Proteins Using Integrated Evolutionary Features

StackPVP_2025 Version:1.0

# # Description:
We propose a novel computational approach called StackPVP, a two-layer stacked ensemble learning framework that leverages integrated evolutionary features. Specifically, we explored four comprehensive evolutionary feature descriptors—ACC-PSSM, DPC-PSSM, Pse-PSSM, and PSSM-COM—derived from the Position-Specific Scoring Matrix (PSSM), which demonstrated superior discriminative power compared to traditional sequence-derived features. We then applied three complementary feature selection methods—F-score, Variance, and Recursive Feature Elimination with Cross-Validation (RFECV)—to optimize feature subsets, reducing dimensionality while preserving critical information. Finally, we constructed the framework using 12 base machine learning classifiers, with Random Forest (RF) serving as the optimal meta-classifier. Benchmarking results confirm StackPVP’s state-of-the-art performance: on the independent test set of Charoenkwan2020_2.0, it achieved an AUC of 94.26%, accuracy (ACC) of 0.897, and a 1.58% improvement in specificity (Sp) over existing methods. On the Manavalan2018 dataset, it further demonstrated robustness, with improvements of 4.29% in ACC, 16.63% in sensitivity (Sn), and 9.21% in AUC. The source code and datasets for this work are available for download in the GitHub repository (https://github.com/virbioinfor/StackPVP_2025).
## workflow:
<img width="692" height="443" alt="image" src="https://github.com/user-attachments/assets/5adde876-4489-4df8-a842-5299404b125d" />

# # Installation:
StackPVP is implemented in with Keras library. For detail instruction of installing Keras.

## Dependencies:
Python: 3.6
keras: 2.2.2
numpy: 1.24.3
scikit-learn: 0.20.1
pandas: 1.5.3
matplotlib: 3.7.1

# # Usage:
Clone the repository or download compressed source code files.

## Install dependencies:

## Data:
All data used by experiments described in manuscript is available at Github.

## Citation: 
StackPVP: A Stacked Ensemble Classification Framework for Predicting Phage Virion Proteins Using Integrated Evolutionary Features
