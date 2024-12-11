In folder ```MutualInformationMeasure``` one can find our implemetation of Mutual information measure.

In folder ```ChiSquareMeasure``` one can find our implemetation of Chi square measure

In folder ```PearsonCorrelationMeasure``` one can find our implemetation of Pearson correlation measure

In the file ```example.py```, you will find a small working example to evaluate the information content between input and output and after training a model (MLP or Non-stationary transformer), it is evaluated how much information the residuals share with the input.


# Towards measuring predictability
For detailed explanations, please see [**Towards Measuring Predictability**](https://openreview.net/forum?id=jZBAVFGUUo&noteId=LEMTDMLbq7)
, in course of which this repository was developed.

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

Citation
------------
If you use this repository in your research, please consider citing our work:

```bibtex
@article{saleh2024towards,
  title = {Towards Measuring Predictability: To which extent datadriven approaches can extract deterministic relations from data exemplified with time series prediction and classification},
  author = {Saleh, Gholam Zadeh and Vaisakh Shaj and Gerhard Neumann and Tim Breitenbach},
  journal = {Transactions on Machine Learning Research},
  year = {2024}
}


