#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:30:49 2024

@author: danieldabbah
"""
"""
I made the follwing changes to the API:

1. In the train_epoch function, I decided to return nothing, and instead
I decided to use the evaluate function to check the loss and accuracy of the training data after each epoch.
We already used this function to evaulate the validation data, and therefore I preferd to avoid the duplicate code.

 
2. In train_log_linear_with_one_hot function, I decided to return the model.

3. In DataManager class, I added a getter function named get_sents in order to return the sents objects of the desired dataset.
"""
