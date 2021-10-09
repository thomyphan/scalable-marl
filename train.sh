#!/bin/sh

for d in `seq 1 30`;
do
	python train.py VAST-QTRAN Warehouse-4 0.25
	python train.py VAST-QTRAN Warehouse-4 0.5
	python train.py QMIX Warehouse-4
	python train.py QTRAN Warehouse-4
	python train.py IL Warehouse-4
	
	python train.py VAST-QTRAN Battle-20 0.25
	python train.py VAST-QTRAN Battle-20 0.5
	python train.py QMIX Battle-20
	python train.py QTRAN Battle-20
	python train.py IL Battle-20
	
	python train.py VAST-QTRAN GaussianSqueeze-200 0.25
	python train.py VAST-QTRAN GaussianSqueeze-200 0.5
	python train.py QMIX GaussianSqueeze-200
	python train.py QTRAN GaussianSqueeze-200
	python train.py IL GaussianSqueeze-200
done