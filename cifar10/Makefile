DATE := $(shell date "+%Y-%m%d-%H%M%S-%s")

train:
	mkdir -p logs images
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python ./main.py train | tee logs/$(DATE)
	make images/$(DATE).png

images/%.png: logs/%
	gnuplot -e 'src="$^"' -e 'dest="$@"' -e 'exname="$(shell head -1 $^ | sed 's/_/-/g')"' script/plot.gp

dataset:
	git clone --depth 1 https://github.com/mastnk/train1000 dataset
	sed -i 's/import data/import dataset.data as data/g' dataset/train1000.py
	ln -s $(PWD)/_dataset.py $(PWD)/dataset/__init__.py

serve:
	-echo $(shell hostname):8080
	python -m http.server 8080
