In folder ```MutualInformationMeasure``` one can find our implemetation of Mutual information measure.

In folder ```ChiSquareMeasure``` one can find our implemetation of Chi square measure

In folder ```PearsonCorrelationMeasure``` one can find our implemetation of Pearson correlation measure

In the file ```example.py```, you will find a small working example to evaluate the information content between input and output and after training a model (MLP or Non-stationary transformer), it is evaluated how much information the residuals share with the input.


# Towards measuring predictability
Pytorch code for ICML 2024 submission [Towards measuring predictability]

# Requirements

All the required packages can be installed using the following command:

```
pip install --upgrade pip
conda create --name predictability python=3.10
conda activate predictability
pip3 install -r requirements.txt
```

# Requirements

All the required packages can be installed using the following command:

```
conda create --name predictability
conda activate predictability
pip3 install -r requirements.txt
```


How to Train
-------------

With ```predictability_measure``` as the working directory execute the python script
```python example.py```


Datasets
------------
We have provided a sample csv file including a sinusoid plus noise in the data folder.


