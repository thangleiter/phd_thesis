# Repository layout
- The most recent version of the compiled document is located at `build/default/default.pdf`
- Reviews should be placed in `reviews/` :)
- The TeX style is included as a git submodule at `lib/kaobook`. 
- There are two concurrent branches, `kaobook` and `kaobook_tectonic`, which use different project layouts for different TeX engines, `latexmk` and `tectonic`, respectively. 
  To automatically switch to the correct submodule branch when switching branches in the main repository, run 
  ```
  git config core.hooksPath .githooks
  ```

# To Do's
## Content
- [ ]
## Layout
- [ ] Tune the bibliography style
## `latexmk`
- [ ]
## `tectonic`
- [ ] The bibliography font is off

# XeTeX
 
## `luaotfload-tool`
- Apply patch https://github.com/latex3/luaotfload/commit/12521e87463d78e2cbf0bd94a09381bf97ee29be

# Tectonic
- Build from source using `cargo install --path .`
  - Repo at `git@git.rwth-aachen.de:tobias.hangleiter/tectonic`
- Biber and BibLaTeX versions need to be compatible. Download matching binary and replace TeXlive's: https://sourceforge.net/projects/biblatex-biber/files/biblatex-biber/2.17/binaries
- Warnings from `algorithm2e.sty` are due to non-UTF-8 formatting of that file while including UTF-8 characters. Ignore.

# Kaobook
- Clone from source at `git@git.rwth-aachen.de:tobias.hangleiter/kaobook`
- Branches `lualatex` or `tectonic`
## Kaobook diffs
- `kaorefs.sty`
   - `cleveref`
     Old:
	 ```latex
	 \RequirePackage{hyperref}
	 \RequirePackage{varioref}
	 %\RequirePackage{cleveref} % Don't use cleveref! It breaks everything
	 ```
	 New:
	 ```latex
	 \RequirePackage{varioref}
	 \RequirePackage{hyperref}
	 \RequirePackage{cleveref} % Use cleveref! It works perfectly fine
	 ```
	 See [here](https://tex.stackexchange.com/questions/83037/difference-between-ref-varioref-and-cleveref-decision-for-a-thesis).

   - Commented out the line
     ```latex
     \newcommand{\refeq}[1]{\hyperref[eq:#1]\eqname\xspace\ref{eq:#1}}
     ```
     because `\refeq` is already defined by `mathtools`.

## Fonts
Downloaded and followed instructions here: https://git.nsa.his.se/latex/fonts/-/tree/master.

---
**IMPORTANT**

One of those variables breaks `kpsewhich` finding files in `TEXMFHOME`!

---

## TeXLive Integration

---
**IMPORTANT**

Delete the `build/` directories in the `examples/` subdirectories. Otherwise latexmk breaks!

---

Normally, when you write a book with this template, you need that the
`kaobook.cls` and the `styles` directory be in the same directory as the
`main.tex`. After following these instructions, kindly provided by
@pwgallagher, you can start writing your `main.tex` anywhere in your
computer and still be able to use the kaobook class.

LaTeX looks at certain directories to find all the packages it can use.
Integrating the kaobook with the TeXLive installation amounts to
copying all the `*.cls` and `*.sty` files in one of the places that are
searched by LaTeX.

1. Find the appropriate directory by running `kpsewhich 
   -var-value=TEXMFHOME`. For instance, suppose it is
   `/home/john/texmf/`.

1. Create the following hierarchy of directories under the texmf home:
   `tex/latex/kaobook/`.

1. Copy all the `\*.cls` files and the `styles` directory from the
   repository into the directory you just created. If you are in a
   hurry, you can copy the whole repository into that directory.
   Alternatively, you can `git clone` the `kaobook` repository into that folder
   and periodically `git pull` to update your `kaobook` installation.
   In the end, the folder `/home/john/texmf/tex/latex/kaobook` should contain the
   following files
   ```
   kao.sty
   kaobiblio.sty
   kaobook.cls
   kaohandt.cls
   kaorefs.sty
   kaotheorems.sty
   ```

1. Run `kpsewhich kaobook.cls` to make sure that LaTeX can find the
   template.
