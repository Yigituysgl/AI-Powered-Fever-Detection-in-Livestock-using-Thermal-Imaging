# AI-Powered-Fever-Detection-in-Livestock-using-Thermal-Imaging
This project is a  prototype demo  that uses thermal images of cows to detect signs of fever using deep learning. It combines computer vision and explainable AI techniques (Grad-CAM) to support early disease detection in livestock — a critical step for smarter, healthier, and more efficient farming.

Project Goal

The goal of this project is to build an AI-powered system that:
- Detects fever in cows from thermal images using a trained CNN (ResNet18)
- Provides interpretable Grad-CAM visualizations to highlight decision areas
- Evaluates classification metrics and model performance
- Deploys the solution via an interactive Streamlit app (coming soon)

This demo is part of a research-oriented showcase exploring AI in agriculture and veterinary health monitoring.


 Dataset

- Dataset: ~1237 paired thermal images of cows labeled as either `Normal (0)` or `Fever (1)`
- Image type: RGB thermal images
- Labels: Binary (0 = Normal, 1 = Fever)
- Source: [Holstein Thermal Dataset] (internal)

 Model

- Architecture: `ResNet18` (pretrained on ImageNet)
- Fine-tuned on thermal dataset
- Image input shape: `224x224`
- Final layer modified for binary classification
- Optimizer: Adam, Loss: BCEWithLogitsLoss

 Evaluation

After training, the model was evaluated on a test set of 131 images:

| Metric      | Normal | Fever |
|-------------|--------|-------|
| Precision   | 1.00   | 0.615 |
| Recall      | 0.959  | 1.00  |
| F1-score    | 0.979  | 0.762 |

- Accuracy: 96.2%
- Confusion Matrix:
    - True Positive (Fever): 8
    - False Positive: 5
    - False Negative: 0


 Grad-CAM Visualization

To improve model interpretability, Grad-CAM is used to highlight which areas of the thermal image were most influential in the model’s decision (e.g., head and udder regions showing elevated temperature).

Streamlit App (Coming Next)

 preparing a simple  Streamlit interface  that will:
- Allow users to upload thermal cow images
- Predict if the animal has a fever
- Display confidence scores
- Visualize Grad-CAM result
- Provide model explanation
