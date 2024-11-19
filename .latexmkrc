$out_dir = "./out";
$aux_dir = "./out"; # needs to be set; setting to auxil errors (win)

# Output a pdf
set_tex_cmds('-synctex=1 -interaction=nonstopmode -file-line-error -shell-escape %O %S');
$pdf_mode = 4;  # 5: xetex, 4: luatex

# By default compile only the file called 'main.tex'
@default_files = ('main.tex');

# Compile the glossary and acronyms list (package 'glossaries')
add_cus_dep( 'acn', 'acr', 0, 'makeglossaries' );
add_cus_dep( 'glo', 'gls', 0, 'makeglossaries' );
$clean_ext .= " acr acn alg glo gls glg bak dvi aux log toc fls bcf run.xml out .fdb_latexmk blg bbl nav snm";
sub makeglossaries {
   my ($base_name, $path) = fileparse( $_[0] );
   pushd $path;
   my $return = system "makeglossaries", $base_name;
   popd;
   return $return;
}

# Compile the nomenclature (package 'nomencl')
# add_cus_dep( 'nlo', 'nls', 0, 'makenlo2nls' );
# sub makenlo2nls {
#     system( "makeindex -s nomencl.ist -o \"$_[0].nls\" \"$_[0].nlo\"" );
# }
