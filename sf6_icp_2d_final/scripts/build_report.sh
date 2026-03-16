#!/bin/bash
# Compile the LaTeX report.
# Run from repository root: bash scripts/build_report.sh

cd docs
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
echo "Report compiled: docs/final_report.pdf"
