#!/usr/bin/env python
# coding: utf-8


from nnet import ReadDataset
import os


# If scaler existe remove it
if os.path.isfile("output/scaler.save"):
    os.remove("output/scaler.save")
    print("Found a scaler and removed it")

file = "data/full.csv"
trainset = ReadDataset(file, for_test=True)
print("done")
a = [
    "ass",
    "dddd",
    "dddddddddd",
    "aaaaasdñfasdjfladsjflkdsjfkasdjfkads",
    "sdfsdfsdfsdfsd",
]
