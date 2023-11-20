#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   init.py
@Time    :   2023/03/05 16:14:14
@Author  :   Felix
'''


import wandb

if __name__ == "__main__":
    wandb.init(
            # set the wandb project where this run will be logged
            project="test-run-cifar",

            # track hyperparameters and run metadata
            config={
                "dataset": "CIFAR"
            }
    )