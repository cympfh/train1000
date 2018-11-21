DATE := $(shell date "+%Y-%m%d-%H%M%S-%s")

train:
	mkdir -p logs images
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python ./main.py train | tee logs/$(DATE)

plots:
	for f in $(shell ls logs); do make images/$$f.png; done

images/%.png: logs/%
	gnuplot -e 'src="$^"' -e 'dest="$@"' -e 'exname="$(shell head -1 $^ | sed 's/_/-/g')"' script/plot.gp

dataset:
	git clone --depth 1 https://github.com/mastnk/train1000 dataset
	sed -i 's/import data/import dataset.data as data/g' dataset/train1000.py
	cp _dataset.py dataset/__init__.py
