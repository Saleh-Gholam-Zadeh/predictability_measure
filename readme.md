# The dependence measures are in the folder ending with "measure".
# In the folder "Example", you will find a small working example to evaluate the information content between input and output and after training a model (Non-stationary transformer), it is evaluated how much information the residuals share with the input.

# Towards measuring predictability
Pytorch code for ICML 2024 paper [Towards measuring predictability]

Dependencies
--------------
* torch==1.3.1
* python 3.7
* PyYAML==5.3


How to Train
-------------

With ```predictability_measure``` as the working directory execute the python script
```python main.py```


Datasets
------------
We have provided a sample csv file including a sinusoid plus noise


