# Hidden Parameter Recurrent State Space Models (HiP-RSSM)
Pytorch code for ICLR 2022 paper [Hidden Parameter Recurrent State Space Models For Changing Dynamics Scenarios](https://openreview.net/forum?id=ds8yZOUsea)

Dependencies
--------------
* torch==1.3.1
* python 3.*
* omegaconf==2.1.1
* hydra-core==1.1.1
* PyYAML==5.3
* wandb==0.10.25

How to Train
-------------

With ```HiP-RSSM``` as the working directory execute the python script
```python experiments/mobileRobot/mobile_robot_ffnn_acrkn.py model=default```


Datasets
------------
The dataset used here is that of a mobile robot traversing terrain of different slopes.


