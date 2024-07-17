#!/bin/bash

# Run 1
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=true CROSS_VALIDATION=false 

# Run 2
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=false CROSS_VALIDATION=false 

# Run 3
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=true CROSS_VALIDATION=false 

# Run 4
python train_and_test.py WEIGHT_STRATEGY=uniform SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=false CROSS_VALIDATION=false 

# Run 5
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=true CROSS_VALIDATION=false 

# Run 6
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=true WEIGHT_LOSS=false CROSS_VALIDATION=false 

# Run 7
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=true CROSS_VALIDATION=false 

# Run 8
python train_and_test.py WEIGHT_STRATEGY=inverse SPLIT_WEATHER_ATTRIBUTE=false WEIGHT_LOSS=false CROSS_VALIDATION=false 

