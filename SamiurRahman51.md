# Fashion-MNIST Classifier: Neural Network from Scratch in NumPy

A neural network that classifies Fashion-MNIST clothing images, built from scratch with NumPy. It does not use TensorFlow, PyTorch or scikit-learn. I wrote this while working through the book *Neural Networks from Scratch in Python* by Harrison Kinsley and Daniel Kukieła. The code and the comments are my own notes from studying each chapter.

This project builds on my earlier repository, [neural-network-from-scratch-numpy](https://github.com/YOUR-USERNAME/neural-network-from-scratch-numpy). There, I wrote the layers, activations, losses and optimizers. Here, I put them together into a reusable `Model` class and trained it on a real image dataset.

## What is in the notebook

- **Layers and activations:** dense layer, dropout, ReLU, Softmax, Sigmoid and Linear
- **Losses:** categorical and binary cross-entropy, mean squared error, mean absolute error
- **Optimizers:** SGD (with decay and momentum), Adagrad, RMSprop, Adam
- **Regularization:** L1 and L2 penalties on weights and biases, and dropout
- **`Model` class:** add layers, finalize, train with mini-batches, evaluate on validation data, predict, and save or load parameters or the whole model
- **Accuracy classes:** for classification and regression

## Dataset

[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) has 60,000 training images and 10,000 test images of clothing items, each 28 x 28 pixels, in 10 classes. The pixel values are scaled to the range from -1 to 1 before training.

The CSV files are large, so they are not in this repository. Download `fashion-mnist_train.csv` and `fashion-mnist_test.csv` from the link above and put them in a `data/` folder.

## Model and result

| Item | Setting |
|---|---|
| Architecture | 784 inputs, 128 ReLU, 128 ReLU, 10 Softmax |
| Loss | Categorical cross-entropy |
| Optimizer | Adam, learning rate decay 1e-4 |
| Regularization | L1 and L2 on the weights |
| Training | 10 epochs, batch size 128 |

| Dataset | Accuracy | Loss |
|---|---|---|
| Training (epoch 10) | 86.0% | 0.500 (including 0.115 regularization loss) |
| Validation (test set) | 85.7% | 0.391 |

The training and validation accuracy stay close, so the model is not overfitting. I also tested saving and loading: after loading the saved model, I got the same 85.7% validation accuracy. In a quick test on 10 test images, all 10 predictions were correct.

## How to run

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook fashion_mnist_neural_network_from_scratch.ipynb
```

Put the two CSV files in `data/` first.

## Notes

- This is a learning project. A convolutional network would score much higher on this dataset. The goal here was to understand every part of a neural network by writing it myself.
- Next step: apply the same ideas to real geotechnical data, such as jute geotextile test results, and compare against simpler models like linear regression.

## About

Md. Samiur Rahman Surid, Civil Engineering graduate (BUET), Geotechnical Engineering. I am teaching myself machine learning and neural networks to use them in geotechnical research.
