# Add python to path
$ENV{'PATH'} = join(';', $ENV{'LOCALAPPDATA'} . "/miniforge3", $ENV{'PATH'});
# print "Updated PATH: $ENV{'PATH'}\n";

$out_dir = "./build";
$aux_dir = "./build"; # needs to be set; setting to auxil errors (win)
$allow_subdir_creation = 1; # needs to be set for \include to work with subdirs!
$preview_continuous_mode = 1;
$pdf_mode = 4;  # 5: xetex, 4: luatex, 1: pdflatex
$pdf_previewer = join(' ', %ENV{'LOCALAPPDATA'} . '/SumatraPDF/SumatraPDF.exe', '-reuse-instance');
$clean_ext = " acr acn alg glo gls glg bak dvi aux log toc fls bcf run.xml out .fdb_latexmk blg bbl nav snm xdy synctex synctex(busy)";

# By default compile only the file called 'main.tex'
@default_files = ('main.tex');

# IMPORTANT:
# -file-line-error kills the $allow_subdir_creation functionality!
set_tex_cmds('-synctex=1 -interaction=nonstopmode %O %S');

# Compile the glossary and acronyms list (package 'glossaries')
add_cus_dep( 'acn', 'acr', 0, 'makeglossaries' );
add_cus_dep( 'glo', 'gls', 0, 'makeglossaries' );
sub makeglossaries {
   my ($base_name, $path) = fileparse( $_[0] );
   pushd $path;
   my $return = system "makeglossaries", $base_name;
   popd;
   return $return;
}

# # Compile the nomenclature (package 'nomencl')
# add_cus_dep( 'nlo', 'nls', 0, 'makenlo2nls' );
# sub makenlo2nls {
#     system( "makeindex -s nomencl.ist -o \"$_[0].nls\" \"$_[0].nlo\"" );
# }
