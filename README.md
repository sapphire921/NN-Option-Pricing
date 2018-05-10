# NeuralNetworksForOptionsPricing
Three classes of neural networks are built to price European options. Results are compared with Variance Gamma model and Heston model.

1. Built Modular Neural Networks and Gated Neural Networks for options pricing using Keras and TensorFlow in Python.
2. Implemented Variance Gamma and Heston stochastic volatility models for options pricing; calibrated Heston model; developed a hybrid MNN by using difference between market prices and model prices as target values.
3. Downloaded S&P500 Index European Options data from WRDS; cleaned and parsed data using Pandas package in Python.
4. Trained and tested MNN, hybrid MNN and GNN using options from 2000 to 2016; used moneyness and time-to-maturity as features; tuned model hyperparameters; compared MSPE of the three classes of NNs with BS, VG and Heston models.
