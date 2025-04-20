# EDXGB-for-APT
EDXGB is an ensemble deep learning tree model for APT exfiltration detection. In EDXGB, a deep learning neuron model is used as a feature extraction module, and XGBoost is the prediction drive force instead of the Softmax function.




## Hardware Conditions

Our evaluation experiments conducted in this section are deployed on a host computer with the following conditions:

+ 12th Gen Intel(R) Core(TM) i9-12900 2.40 GHz CPU
+ 64GB RAM
+ NVIDIA RTX A4000 GPU
+ Windows 11 Pro OS
+ Python 3.9.16 version

## Experiment Implementation
+ Dataset:
```python
SCVIC-APT-2021: https://ieee-dataport.org/documents/scvic-apt-2021
Unraveled-2023: https://gitlab.com/asu22/unraveled
```
+ Feature Extractor

Four DNN models and four CNN models are prepared as the base deep learning feature extractor of EDXGB.

```python
- Optimizer: Adam
- Batch size: 64
- Epoch: 32
- Dimension: 
DNN_1: Input-128-64-Prediction Drive Force
DNN_2: In.-128-32-Pre.
DNN_3: In.-128-32-64-Pre.
DNN_4: In.-128-64-32-Pre.

CNN_1: In.-128-64-Pre.
CNN_2: In.-128-32-Pre.
CNN_3: In.-128-64-32-Pre.
CNN_4: In.-256-64-32-Pre.
```

+ Prediction Drive Force

An XGBoost is hired as the prediction drive force of EDXGB. We set the learning rate of EDXGBs to 0.01, the maximum depth of each tree to 10, and the number of generated trees to 30.
Optimization experiments are executed under parameters as follows:

```python
- Learing rate: [0.1, 0.01, 0.001]
- Maximum depth of each tree: [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
- Number of generated trees: [20, 30, 40, 50, 100, 150, 200]
```

For XGBoost, download and install:

```python
pip install xgboost
```

## Usage

```python
#
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
