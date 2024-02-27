#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:30:49 2024

@author: danieldabbah
"""
"""
I made the follwing changes to the API:

In the train_epoch function, i decided to return nothing, and instead
I decided to use the evaluate function to check the loss and accuracy of the training data after each epoch.
We already used this function to evaulate the validation data, and therefore I preferd to avoid the duplicate code.
"""
