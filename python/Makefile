untrained: compile
	python3 main.py --dataset cifar10 --kahan True

trained: compile
	python3 trained_probe.py

compile:
	black .
	flake8 --ignore=E,W .

pdf:
	a2ps *.py -o *.ps --pro=color