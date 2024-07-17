#!/bin/bash

# First set of 4 runs
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=true CROSS_VALIDATION=true --multirun
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=false CROSS_VALIDATION=true --multirun
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=true CROSS_VALIDATION=true --multirun
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=false CROSS_VALIDATION=true --multirun

# Second set of 4 runs
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=true CROSS_VALIDATION=true --multirun
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=false CROSS_VALIDATION=true --multirun
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=true CROSS_VALIDATION=true --multirun
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=false CROSS_VALIDATION=true --multirun

