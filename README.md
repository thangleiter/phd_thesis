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

# Zotero
## Better BibTeX
- Add the following javascript to the `postscript` field:
  <details><summary>postscript code</summary>

  ```js
  /* 
  Thanks chatty:
  https://genai.rwth-aachen.de/app/conversations/6800f383ef384a984672ee4a
  */
  /**
   * Process text to protect math mode and backslashed LaTeX commands.
   * - Math mode fragments ($...$) are wrapped in <script>{…}</script>
   * - All backslashed LaTeX commands (with an optional argument) are also wrapped,
   *   while avoiding nesting over already-protected parts.
   * - For \texttt commands, underscores in their argument are escaped.
   *
   * @param {string} text - The input text.
   * @returns {string} - The processed (protected) text.
   */
  function protectLatex(text) {
    // 1. Protect math mode fragments
    text = text.replace(/(\$.*?\$)/g, '<script>{$1}</script>');

    // 2. Protect LaTeX commands, but skip any parts that are already protected.
    text = protectLatexCommandsAvoidNesting(text);

    return text;
  }

  /**
   * Processes the text so that any region that is not already wrapped in
   * a <script>{…}</script> block gets its LaTeX commands protected.
   *
   * @param {string} input - The input text.
   * @returns {string} - The text with unprotected regions processed.
   */
  function protectLatexCommandsAvoidNesting(input) {
    let output = "";
    let pos = 0;
    const openTag = "<script>{";
    const closeTag = "</script>";

    while (pos < input.length) {
      // Find the next already protected block
      let nextIdx = input.indexOf(openTag, pos);
      if (nextIdx === -1) {
        // Process remainder
        output += protectLatexCommandsInSegment(input.substring(pos));
        break;
      }
      // Process the text segment that is not yet protected.
      output += protectLatexCommandsInSegment(input.substring(pos, nextIdx));
      // Then, copy the already protected block unmodified.
      let closeIdx = input.indexOf(closeTag, nextIdx);
      if (closeIdx === -1) {
        // If there's no closing tag (should not happen), append the rest.
        output += input.substring(nextIdx);
        break;
      }
      output += input.substring(nextIdx, closeIdx + closeTag.length);
      pos = closeIdx + closeTag.length;
    }
    return output;
  }

  /**
   * Processes a text segment to wrap LaTeX commands in <script>{…}</script>.
   * This function uses a regex that looks for a backslash command (one or more word characters)
   * optionally followed by a braced argument. For \texttt commands, underscores in the argument are escaped.
   *
   * Note: This simplified regex does not handle nested braces.
   *
   * @param {string} segment - The input text segment.
   * @returns {string} - The processed segment.
   */
  function protectLatexCommandsInSegment(segment) {
    return segment.replace(/(\\\w+)(\{[^{}]*\})?/g, function(match, command, arg) {
      let fullCommand = command;
      if(arg) {
        // For \texttt commands, escape underscores in the argument.
        if (command === '\\texttt') {
          arg = arg.replace(/_/g, '\\_');
        }
        fullCommand += arg;
      }
      return '<script>{' + fullCommand + '}</script>';
    });
  }

  // Example usage in your hook:
  if (Translator.BetterTeX && tex.has.title) {
    let title = zotero.title;
    title = protectLatex(title);
    tex.add({ name: 'title', value: title });
  }
  ```
  
  </details>
   
  This does a couple of things. 
  - First, it protects `$`-delimited math mode (see [here](https://retorque.re/zotero-better-bibtex/exporting/scripting/#detect-and-protect-latex-math-formulas)) in `title` fields.
  - Next, it protects LaTeX macros (`\texttt{}` for example) in `title` fields, but takes care to not nest `<script>` blocks from previous math mode protections.
  - Finally, it replaces `_` underscores.
- Alternatively, import BBT settings from `lib/bbt_settings.json`.

# XeTeX
 
## `luaotfload-tool`
- Apply [this](https://github.com/latex3/luaotfload/commit/12521e87463d78e2cbf0bd94a09381bf97ee29be) patch (TexLive 2024)

# Tectonic
- Build from source using `cargo install --path .`
  - Repo at `git@git.rwth-aachen.de:tobias.hangleiter/tectonic @ main`
- Biber and BibLaTeX versions need to be compatible. Download [matching binary](https://sourceforge.net/projects/biblatex-biber/files/biblatex-biber/2.17/binaries) and replace TeXlive's.
- Warnings from `algorithm2e.sty` are due to non-UTF-8 formatting of that file while including UTF-8 characters. Ignore.
- The `autogobble` option of `minted` does not seem to work.
- Adding options to a custom `minted` environment does not work, e.g.,
  ```latex
  \newminted[py]{python}
  \begin{py}[fontsize=\footnotesize]
  \end{py}
  ```
  Likely a TL 2022 issue.

# Kaobook
- Linux:
  - Soft link `lib/kaobook` to `$HOME/texmf/src.tex/latex/kaobook`.
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


