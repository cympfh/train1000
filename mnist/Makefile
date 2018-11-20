DATE := $(shell date "+%Y-%m%d-%H%M%S-%s")

train:
	mkdir -p logs images
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python ./main.py train | tee logs/$(DATE)
	gnuplot -e 'date="$(DATE)"' script/plot.gp

dataset:
	git clone --depth 1 https://github.com/mastnk/train1000 dataset
	sed -i 's/import data/import dataset.data as data/g' dataset/train1000.py
	cp _dataset.py dataset/__init__.py
