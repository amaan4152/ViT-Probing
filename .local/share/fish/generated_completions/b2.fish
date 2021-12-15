# b2
# Autogenerated from man page /usr/share/man/man1/b2.1.gz
complete -c b2 -l prefix --description 'Install architecture independent files here.'
complete -c b2 -l exec-prefix --description 'Default: <PREFIX>.'
complete -c b2 -l libdir --description 'Install library files here.  Default: <EPREFIX>/lib.'
complete -c b2 -l includedir --description 'Install header files here.  Default: <PREFIX>/include.'
complete -c b2 -l cmakedir --description 'Install CMake configuration files here.  Default: <LIBDIR>/cmake.'
complete -c b2 -l no-cmake-config --description 'Do not install CMake configuration files.'
complete -c b2 -l stagedir --description 'Install library files here Default: . /stage . PP Other Options:.'
complete -c b2 -l build-type --description 'Build the specified pre-defined set of variations of the libraries.'
complete -c b2 -l build-dir --description 'Build in this location instead of building within the distribution tree.'
complete -c b2 -l show-libraries --description 'Display the list of Boost libraries that require build and installation steps…'
complete -c b2 -l layout --description 'Determine whether to choose library names and header locations such that mult…'
complete -c b2 -l buildid --description 'Add the specified ID to the name of built libraries.'
complete -c b2 -l python-buildid --description 'Add the specified ID to the name of built libraries that depend on Python.'
complete -c b2 -l help --description 'This message.'
complete -c b2 -l 'with-<library>' --description 'Build and install the specified <library>.'
complete -c b2 -l 'without-<library>' --description 'Do not build, stage, or install the specified <library>.'

