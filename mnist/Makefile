dataset:
	git clone --depth 1 https://github.com/mastnk/train1000 dataset
	sed -i 's/import data/import dataset.data as data/g' dataset/train1000.py
	cp _dataset.py dataset/__init__.py
