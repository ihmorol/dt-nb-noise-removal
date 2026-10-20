# ICEEICT 2027 conference draft

main.pdf is the anonymous paper: six A4 pages, at the six-page limit (five tables, five vector figures). Edit main.tex and references.bib; manuscript.txt is extracted reading text for language revision. The original manuscripts and experiment outputs remain unchanged.

## Compile

Run in this folder with a LaTeX installation:

    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex

IEEEtran.cls and IEEEtran.bst are included with their original notices. Required packages are cite, amsmath, amssymb, graphicx, booktabs, algorithm, algpseudocode and hyperref. Generated tables and vector figures are included, so compilation needs neither experiment data nor Python.

## Regenerate evidence

Inside the original repository:

    python paper/iceeict2027-submission/scripts/generate_evidence.py

The generator reads results/EXP-E9_farid, the 2014 table values in results/EXP-R1_replication/replication.csv and the existing dataset loader, checks saved fold means, retains the eight exploratory accuracy tests, and exports tables/figures. It fits no models. Regeneration requires the original repository, its existing Python dependencies and installed Times New Roman fonts. The source ZIP can compile without regeneration.

## Language revision

Follow the final Lorenz paper's plain first-person plural style while retaining DT-NB terminology. Preserve numerical observations, exploratory-test disclosure, descriptive parallel comparisons and scientific limits. Tables are generated, so regenerate rather than typing new values. Recompile and check page count, figures, anonymity and citations after editing.

## Outstanding checks

Author approval/rewrite, fresh similarity screening, CMT accounts/metadata and upload remain. The draft truthfully discloses AI assistance with drafting, language, typesetting and figures/tables. The conference requires sole non-intellectual LLM use; compliance remains unresolved. Rewriting alone does not establish eligibility.

Official rules: https://iceeict.mist.ac.bd/authors/paper-submission/

See review/pdf-checks.json, review/requirements-check.md, review/citation-ledger.md and review/changes.md. No acceptance or indexing guarantee is made.
