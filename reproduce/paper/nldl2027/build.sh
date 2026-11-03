#!/bin/sh
# Clean, from-scratch build. Always verify page count with this, never with an
# incremental run: stale .aux files converge to different float placement and
# can silently report one page fewer than a fresh build.
set -e
cd "$(dirname "$0")"
rm -f main.aux main.log main.out main.bcf main.run.xml main.blg main.pdf
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
if command -v biber >/dev/null 2>&1; then biber main >/dev/null 2>&1; fi
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
printf 'pages:     %s (whole PDF, appendix included)\n' "$(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
# NLDL camera ready: the main text, up to the end of the conclusions, must end
# on page 9 or earlier. \label{main:end} sits after the Reproducibility paragraph.
printf 'main ends: page %s (limit 9)\n' "$(sed -n 's/.*newlabel{main:end}{{[^}]*}{\([0-9]*\)}.*/\1/p' main.aux)"
printf 'errors:    %s\n' "$(grep -cE '^!' main.log || true)"
printf 'overfull:  %s\n' "$(grep -c Overfull main.log || true)"
printf 'undefined: %s\n' "$(grep -ci undefined main.log || true)"
printf 'em dashes: %s\n' "$(grep -c -- '---' main.tex || true)"
printf 'TODOs:     %s (must be 0 before submitting)\n' "$(grep -o '\\TODO{' main.tex | wc -l | tr -d ' ')"
