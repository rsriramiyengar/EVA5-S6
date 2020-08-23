# **EVA5-Assignment-5**

## Group Members:
- Gaurav Makkar
- Sriram Iyengar
- Mohit Bhandari
- Suman Debnath


### Version 0 (Base Code)
-------------------------

**Target** 

Base Code with Squeeze and Excitation model, to make sure we are able to set the transforms, data loader, and basic workflow

**Results** 

- Parameters: 13,832
- Best Train Accuracy: 99.34%
- Best Test Accuracy: 98.71%

**Analysis** 

- Model is quite large but working
- Model is over-fitting

[Version 0 - Code](https://github.com/EVA5-Stars/S5/blob/master/00_EVA5_Session5_Base_Code_Step_0.ipynb)

### Version 1 (with GAP)
-------------------------

**Target** 

Add GAP and remove the last BIG kernel in the Output Block of the CNN.

**Results** 

- Parameters: 7,432
- Best Train Accuracy: 97.94%
- Best Test Accuracy: 97.92%

**Analysis** 

- As we replaced the last layer with GAP, the total no. of parameters reduced to bellow 8000
- Model was able to generalise it better
- Overall model's accuracy got reduced. , but test accuracy was observed to be little more w.r.t the training accuracy, most of the time, which was a good thing.

[Version 1 - Code](https://github.com/EVA5-Stars/S5/blob/master/01_EVA5_Session5_GAP_Step_1.ipynb)

### Version 2 (with Batch-Norm)
-------------------------------

**Target** 

Add Batch-norm to increase model efficiency.

**Results** 

- Parameters: 7,612
- Best Train Accuracy: 99.18%
- Best Test Accuracy: 99.13%

**Analysis** 

- No. of model parameters got increased a bit, which was expected beacuse of Batch Normalization
- Overall model performance got increased a bit, but we again started to see overfit and its not able to reach 99.4% accuracy

[Version 2 - Code](https://github.com/EVA5-Stars/S5/blob/master/02_EVA5_Session5_BN_Step_2.ipynb)

### Version 3 (with LR Scheduler)
---------------------------------

**Target** 

Add LR Scheduler

**Results** 

- Parameters: 7,612
- Best Train Accuracy: 99.62%
- Best Test Accuracy: 99.28%

**Analysis** 

- Got better performance a bit, but still not 99.4%
- We played with different combinations of step_size and gamma value for the StepLR,
    - step_size = [5,6,7,8,9,10,11,12,13]
    - gamma = [.1, .2, .3]
- Finally we frozen to (step_size=9, gamma=0.2)

[Version 3 - Code](https://github.com/EVA5-Stars/S5/blob/master/03_EVA5_Session5_StepLR_Step_3.ipynb)

### Version 4 (with ImageAugmentation)
--------------------------------------

**Target** 

Add rotation, try with 5-10 degrees

**Results** 

- Parameters: 7,612
- Best Train Accuracy: 99.23%
- Best Test Accuracy: 99.45%

**Analysis** 

- Tried with multiple degrees for image augmentation, but finally decided to go with (-10, +10)
- Model was not able to generalize it better.
- Reached the accuracy of 99.4 % for last 4 epocs with total no. of parameters less than 8000

[Version 4 - Code](https://github.com/EVA5-Stars/S5/blob/master/04_EVA5_Session5_ImageAugmentation_Step_4.ipynb)


