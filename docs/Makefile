# Build the project report.
#
# Usage:
#   make          # build report.pdf
#   make clean    # remove LaTeX build artefacts (keeps .pdf)
#   make purge    # remove everything, including report.pdf

TEX      := report.tex
PDF      := $(TEX:.tex=.pdf)
ENGINE   := pdflatex
FLAGS    := -interaction=nonstopmode -halt-on-error -shell-escape
AUXFILES := *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean purge view

all: $(PDF)

$(PDF): $(TEX)
	$(ENGINE) $(FLAGS) $(TEX)
	$(ENGINE) $(FLAGS) $(TEX)

clean:
	rm -f $(AUXFILES)

purge: clean
	rm -f $(PDF)

view: $(PDF)
	@which xdg-open >/dev/null 2>&1 && xdg-open $(PDF) || echo "open $(PDF) manually"
