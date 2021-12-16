untrained: 
	python3 main_driver.py --dataset cifar10 --kahan True

trained: 
	python3 trained_probe.py

automate_train_probes: 
	python3 main_driver.py --dataset cifar10 --kahan True --slurm "probes pconv_vs_vit"
 		

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
 