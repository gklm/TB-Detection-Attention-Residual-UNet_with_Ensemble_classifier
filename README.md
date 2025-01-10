# Enhanced Tuberculosis Bacilli Detection using Attention-Residual U-Net and Ensemble Classification

This repository contains code for tuberculosis bacilli detection using an Attention-Residual U-Net for image segmentation and an ensemble classifier (SVM, Random Forest, XGBoost, and Voting Classifier) for classification.

Repository Structure

/
├── segmentation/         # Contains the segmentation code using Attention-Residual U-Net
├── classification/       # Contains the ensemble classification code (SVM, RF, XGB, and Voting Classifier)
├── utils/                # Helper scripts for data preprocessing and evaluation
└── README.md             # Instructions and details about the project

Requirements

To run the code, ensure you have the following libraries installed:

Python 3.8+

TensorFlow 2.x

NumPy

OpenCV

Scikit-learn

XGBoost

You can install the required packages using:

pip install -r requirements.txt

Usage Instructions

1. Segmentation

Navigate to the segmentation/ folder.

Run the segmentation script to generate masks for input images.

python unet_segmentation.py

2. Classification

Navigate to the classification/ folder.

Ensure your dataset is organized into folders for different classes.

Run the ensemble classification script to train the models.

python ensemble_classifier.py

3. Preprocessing and Utilities

Helper functions for data preprocessing and evaluation are provided in the utils/ folder.

Citation

If you use this code in your research, please cite our arXiv preprint:

@article{greeshma2025tb,
  title={Enhanced Tuberculosis Bacilli Detection using Attention-Residual U-Net and Ensemble Classification},
  author={Greeshma K, Vishnukumar S},
  journal={The Visual Computer (pending)},
  year={2025},
  eprint={2501.03539},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

The preprint is available at https://arxiv.org/abs/2501.03539
