# Hybrid Quantum-Classical Neural Network for MNIST Digit Classification

## Overview

This project implements a hybrid quantum-classical neural network to classify a subset of the MNIST dataset, specifically focusing on the digits 0 through 3. The approach combines classical convolutional neural networks (CNNs) for feature extraction with quantum neural networks (QNNs) for classification, leveraging the strengths of both paradigms.

## Motivation

The integration of quantum computing with classical machine learning techniques offers promising avenues for enhancing computational capabilities and achieving superior performance in various tasks. This project draws inspiration from recent advancements in quantum machine learning, particularly the work presented in the paper "[A multi-classification classifier based on variational quantum computation](https://link.springer.com/article/10.1007/s11128-023-04151-6)". By applying these concepts to the MNIST digit classification problem, we aim to explore the practical benefits and challenges of hybrid quantum-classical models.

## Project Structure

1. **Data Preparation**:
   - **Dataset**: Utilizes the MNIST dataset, filtered to include only the digits 0, 1, 2, and 3.
   - **Transformations**: Applies normalization and tensor conversion to the images.
   - **Splitting**: Divides the data into training and testing sets using an 80-20 split.

2. **Feature Extraction**:
   - **LeNet-Based CNN**: Implements a classical LeNet architecture to extract features from the input images, reducing dimensionality and capturing essential patterns.

3. **Quantum Neural Network**:
   - **Quantum Circuit**: Defines a variational quantum circuit with parameterized gates to process the extracted features.
   - **Hybrid Model**: Combines the classical feature extractor with the quantum circuit, forming a hybrid neural network capable of end-to-end training.

4. **Training and Evaluation**:
   - **Loss Function**: Employs cross-entropy loss suitable for multi-class classification.
   - **Optimizer**: Uses the Adam optimizer for efficient training.
   - **Metrics**: Evaluates the model based on training and testing accuracy over multiple epochs.

5. **Visualization**:
   - **Data Visualization**: Provides functions to visualize original images and transformed features.
   - **Quantum Circuit Diagram**: Displays the structure of the quantum circuit used in the model.

## Comparison of CNN and QNN Performance

The performance of the classical CNN and the hybrid QNN was analyzed using the training loss, training accuracy, and test accuracy.

### Training Loss

The training loss comparison shows that the QNN converges significantly faster and reaches a much lower loss compared to the CNN. This indicates that the QNN is learning the classification task more effectively with fewer epochs.

- **CNN Loss**: Starts high and gradually decreases over time but remains relatively large even after multiple epochs.
- **QNN Loss**: Decreases sharply within the first few epochs and stabilizes at a very low value, suggesting efficient training.

### Training and Test Accuracy

The accuracy comparison reveals that the QNN achieves near-perfect accuracy in both training and test sets, whereas the CNN struggles to exceed 85% accuracy.

- **CNN Accuracy**:
  - Training Accuracy: ~83%
  - Test Accuracy: ~83%
- **QNN Accuracy**:
  - Training Accuracy: ~99.9%
  - Test Accuracy: ~99.9%

This stark difference suggests that the QNN model is significantly more powerful in classifying the dataset, likely due to its quantum-enhanced feature representations.

## Dependencies

- Python 3.x
- PennyLane
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- numpy

## Installation

To set up the environment, install the required packages using pip:

```bash
pip install pennylane torch torchvision scikit-learn matplotlib numpy
```

## Usage

1. **Data Preparation**: Run the data loading and preprocessing scripts to obtain the training and testing datasets.
2. **Model Initialization**: Initialize the hybrid model by combining the classical feature extractor and the quantum neural network.
3. **Training**: Train the model using the provided training loop, adjusting hyperparameters as necessary.
4. **Evaluation**: Assess the model's performance on the test dataset and visualize the results.
5. **Visualization**: Utilize the visualization functions to inspect the data and the quantum circuit.

## References

- Zhou, J., Li, D., Tan, Y., Yang, X., Zheng, Y., & Liu, X. (2023). A multi-classification classifier based on variational quantum computation. *Quantum Information Processing, 22*(412). [https://link.springer.com/article/10.1007/s11128-023-04151-6](https://link.springer.com/article/10.1007/s11128-023-04151-6)

## Acknowledgments

This project is inspired by the methodologies presented in the aforementioned paper, adapting their approach to the MNIST digit classification task. Special thanks to the authors for their contributions to the field of quantum machine learning.

