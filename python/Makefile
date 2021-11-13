untrained: compile
	python3 reg_vit_test.py --dataset cifar10 --kahan True

trained: compile
	python3 trained_probe.py

compile:
	black .
	flake8 --ignore=E,W .

pdf:
	for FILE in *; do \
		if [ -d $${FILE} ]; then \
			continue; fi; \
		a2ps $${FILE} -o "$${FILE%%.*}.ps" --pro=color; \
		ps2pdf "$${FILE%%.*}.ps"; done; \
	rm *.ps; \
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=vit_probes.pdf *.pdf;
	
clean: 
	rm *.pdf