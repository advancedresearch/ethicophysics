TEXS=$(shell find *.tex)
PDFS=$(TEXS:%.tex=%.pdf)

all: $(PDFS)
	echo $(TEXS)
	echo $<

%.pdf: %.tex
	pdflatex $<
