# Repository layout
- The most recent version of the compiled document is automatically placed at `review/main.pdf`.
- Reviews should be placed in `reviews/`, preferably by branching off at the commit at which the review starts :)
- The TeX style is included as a git submodule at `lib/kaobook`. 
- There are two concurrent branches, `kaobook` and `kaobook_tectonic`, which use different project layouts for different TeX engines, `latexmk` and `tectonic`, respectively. 
  To automatically switch to the correct submodule branch when switching branches in the main repository, run 
  ```
  git config core.hooksPath .githooks
  ```

# To Do's
## Content
### FF
- [ ] `\mathsf{}` for `\Hspace`, `\basis`, `\Lspace`?
- [ ] `\mathcal{}` for FF, regular for fidelity.
 
## Layout
- [ ] Tune the bibliography style
- [ ] Chapters instead of parts?
	- [ ] Else numbered parts

## `latexmk`
- [ ] ?

## `tectonic`
- [ ] `\underbrace{}`
- [ ] marginfigure placement

# IntelliJ IDEA
Set the "Use single dictionary for saving words" setting to "project level" to sync the dictionary using git.

# XeTeX
 
## `luaotfload-tool`
- Apply [this](https://github.com/latex3/luaotfload/commit/12521e87463d78e2cbf0bd94a09381bf97ee29be) patch (TexLive 2024)

# Tectonic
- Build from source using `cargo install --path .`
  - Repo at `git@git.rwth-aachen.de:tobias.hangleiter/tectonic @ main`
- Biber and BibLaTeX versions need to be compatible. Download [matching binary](https://sourceforge.net/projects/biblatex-biber/files/biblatex-biber/2.17/binaries) and replace TeXlive's.
- Warnings from `algorithm2e.sty` are due to non-UTF-8 formatting of that file while including UTF-8 characters. Ignore.
- The `autogobble` option of `minted` does not seem to work.
- The font size in `minted` also does not seem to adjust.

# Kaobook
- Linux:
  - Soft link `lib/kaobook` to `$HOME/texmf/tex/latex/kaobook`.
- Windows:
  - No extra setup is needed. Because latex does not follow soft links, the `post-checkout` hook copies the folder.

## Kaobook diffs
These diffs should already be applied in the submodule shipped with this repository.

- `kaorefs.sty`

   - `cleveref`
	 ```diff	 
	 - \RequirePackage{hyperref}
	 - \RequirePackage{varioref}
	 - \RequirePackage{cleveref} % Don't use cleveref! It breaks everything
	 + \RequirePackage{varioref}
	 + \RequirePackage{hyperref}
	 + \RequirePackage{cleveref} % Use cleveref! It works perfectly fine
	 ```
See [here](https://tex.stackexchange.com/questions/83037/difference-between-ref-varioref-and-cleveref-decision-for-a-thesis).

   - Commented out the line
     ```latex
     \newcommand{\refeq}[1]{\hyperref[eq:#1]\eqname\xspace\ref{eq:#1}}
     ```
     because `\refeq` is already defined by `mathtools`.

## Fonts
- Download Libertinus [here](https://github.com/alerque/libertinus).
- Download and follow instructions [here](https://git.nsa.his.se/latex/fonts/-/tree/master) for Liberation Mono.
- Download NewComputerModernMath [here](https://ctan.org/pkg/newcomputermodern?lang=en).


## TeXLive Integration
**IMPORTANT**

Delete the `build/` directories in the `examples/` subdirectories of `lib/kaobook`. Otherwise latexmk breaks!


