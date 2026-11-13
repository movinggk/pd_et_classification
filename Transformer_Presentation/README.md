# The Transformer: architecture, mathematics, and computation

This is a general English lecture on the neural Transformer architecture.
The deck has 50 slides: 38 main slides, 10 mathematical appendices, and
2 reference slides. Allow approximately 50-60 minutes for the main lecture,
with additional time for selected appendices and discussion.

The main construction follows the original encoder-decoder Transformer
with post-layer normalization. Later choices, including Pre-LN, RoPE,
and FlashAttention, are identified separately. The audience should know
matrix multiplication, elementary probability, and derivatives.

## Files

- `main.tex`: the complete editable Beamer source, including speaker notes
  and the bibliography.
- `images/transformer_original.png`: the attributed architecture figure.

The source uses 16:9 slides. It requires no dataset, model checkpoint,
training run, network connection at compilation time, BibTeX, or Biber.
All equations, tables, and function plots are editable in the LaTeX source.

## Build the slides

Use a TeX distribution with Beamer, Latin Modern, PGFPlots, and the standard
packages listed in the preamble. TeX Live is suitable.

Run these commands from the directory containing `main.tex`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, compile twice with pdfLaTeX:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

For Overleaf, upload the source ZIP, keep `main.tex` as the main document,
and select pdfLaTeX. Retain the `images` directory.

## Optional presenter copy

Every slide contains a `\note{...}` block in the source. The default PDF
hides notes. To place speaker notes to the right of each slide, run twice:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -jobname=presenter_notes '\def\PresenterNotes{1}\input{main.tex}'
pdflatex -interaction=nonstopmode -halt-on-error -jobname=presenter_notes '\def\PresenterNotes{1}\input{main.tex}'
```

The resulting `presenter_notes.pdf` has a double-width page for each slide.

## Mathematical conventions

- Token states are row vectors. Appendix I explicitly switches to columns
  for the RoPE rotation identity.
- Source and target indices start at 1. Position coordinates start at 0.
- Target length includes EOS. BOS is the preceding decoder input token.
- Attention softmax normalizes over keys. The output softmax normalizes
  over vocabulary tokens.
- A mathematical masked softmax requires at least one allowed key per row.
- The displayed core attention equations omit projection biases and
  attention-weight dropout. Residual dropout is explicit in the block
  equations. Embedding dropout is discussed in the notes.
- The loss slide defines token-averaged hard-label negative log-likelihood.
  Appendix F gives a separate, explicit label-smoothing convention.
- The scaling argument assumes mutually independent zero-mean, unit-variance
  components. The assumptions are a motivation, not a claim about all
  learned query/key distributions.

## Figure attribution and sources

The architecture figure is Figure 1 from:

Ashish Vaswani and colleagues, *Attention Is All You Need*, NeurIPS 2017.

- Paper: https://arxiv.org/abs/1706.03762
- Figure: https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png

The paper expressly permits reproducing its tables and figures in scholarly
or journalistic works with attribution. The figure retains the original
content and is attributed on the slide. This permission does not imply
that the figure has been placed in the public domain.

The final two slides provide linked primary references for layer
normalization, BERT, T5, Pre-LN analysis, FlashAttention, RoFormer,
Vision Transformer, and attention interpretation. Worked numerical
examples and algebraic derivations are identified as such in the deck.
