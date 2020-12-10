# Warmstarting Parameterizable CFD Simulations With Deep Learningls

Generating example CFD simulations for deep learning can be extremely costly.
Thus we outline a strategy for warm-starting parameterizable 3D CFD domains. We re-parameterize our mesh using
transfinite interpolation to allow for a variety of domains to be processable before
processing the data using 3D convolutional neural networks.

## Requirements

1. [pymethods](https://github.com/notifications?query=repo%3AChr1sC0de%2Fpymethods)

2. [CGALMethods](https://github.com/Chr1sC0de/CGALUnwrapper)

## Contents

1. [Creating Structured Internal Fields From Openfoam VTK](./01_creating_structured_internal_fields_from_vtk.ipynb)
2. [Creating Useful Features For Deep Learning](./02_generating_useful_features.ipynb)
3. [Training and Testing a Deep Learning Model For Predicting the Internal Fields]
