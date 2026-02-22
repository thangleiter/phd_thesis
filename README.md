# Compiling
- The thesis is known to compile with TexLive 2025.
- Dependencies are included as git submodules
    - The TeX style is at `lib/kaobook`.
    - Fonts not available from CTAN are at `lib/fonts`.
- To automatically set up the repository to compile with `latexmk`, run `shell/init.{sh,ps1}`.
  This creates symlinks (copies on Windows) to `$HOME/texmf/tex/latex/kaobook` and `$HOME/texmf/fonts/truetype/LiberationMono/`.
- `minted` needs to be installed and Python on `PATH`. If Python is installed as part of the miniforge3 distribution at either `$HOME/miniforge3` or `$LOCALAPPDATA/miniforge3`, this is automatically taken care of when compiling with `latexmk -r windows.latexmkrc`.
- Once you started the compilation, lay back and relax. Make some tea. Chat with a colleague. It will take a while, and might also not succeed on the first run. If it did not, clean auxiliary files using `latexmk -r [windows,linux].latexmkrc -C` and try again.

# Fonts
If fonts are not found, download and install them in your system:

- Libertinus [here](https://github.com/alerque/libertinus).
- ~~Liberation Mono [here](https://git.nsa.his.se/latex/fonts/-/tree/master)~~. This font is included as a submodule at `lib/fonts`. Initialize manually or run `shell/init.sh`.
- NewComputerModernMath [here](https://ctan.org/pkg/newcomputermodern?lang=en).

# TeXLive Integration
**IMPORTANT**

Delete the `build/` directories in the `examples/` subdirectories of `lib/kaobook`. Otherwise latexmk breaks!

- For a [bugfix concerning spacing](https://github.com/reutenauer/polyglossia/issues/686) in the list of figures, `polyglossia>=2.7` is required.
  A patched file is included at `lib/polyglossia.sty`, but not automatically copied to `$TEXMFHOME`.

# Kaobook
The LaTeX style is at `lib/kaobook`. You might need to initialize the submodule, or just run `shell/init.sh` to set up.

