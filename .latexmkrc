$out_dir = "./out";
$aux_dir = "./out";  # needs to be set; setting to auxil errors
$bibtex = 'bibtex %O %B';
$biber = 'biber %O %B';
$preview_continuous_mode = 1;
$pdf_mode = 4;  # 5: xetex, 4: luatex
$postscript_mode = $dvi_mode = 0;
# $pdf_previewer = "okular --noraise";
set_tex_cmds('--synctex=1 --interaction=nonstopmode --file-line-error %O %S');
$clean_ext = "bak dvi aux log toc fls bcf run.xml out .fdb_latexmk blg bbl nav snm";
@default_files = ('main.tex',);

# $compiling_cmd = "xdotool search --name \"%D\" set_window --name \"%D compiling\"";
# $success_cmd = "xdotool search --name \"%D\" set_window --name \"%D OK\"";
# $warning_cmd = "xdotool search --name \"%D\" ". "set_window --name \"%D CITE/REF ISSUE\"";
# $failure_cmd = "xdotool search --name \"%D\" set_window --name \"%D FAILURE\"";
