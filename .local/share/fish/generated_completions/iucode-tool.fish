# iucode-tool
# Autogenerated from man page /usr/share/man/man8/iucode-tool.8.gz
complete -c iucode-tool -s q -l quiet --description 'Inhibit usual output.'
complete -c iucode-tool -s v -l verbose --description 'Print more information.   Use more than once for added verbosity.'
complete -c iucode-tool -s h -s '?' -l help --description 'List all available options and their meanings.'
complete -c iucode-tool -l usage --description 'Show summary of options.'
complete -c iucode-tool -s V -l version --description 'Show version of program.'
complete -c iucode-tool -s t --description 'RI "Sets the file type of the following files.  " type " can be:" . RS.'
complete -c iucode-tool -l downgrade --description 'When multiple versions of the microcode for a specific processor are availabl…'
complete -c iucode-tool -l no-downgrade --description 'When multiple versions of the microcode for a specific processor are availabl…'
complete -c iucode-tool -l strict-checks --description 'Perform strict checks on the microcode data.'
complete -c iucode-tool -l no-strict-checks --description 'Perform less strict checks on the microcode data.'
complete -c iucode-tool -l ignore-broken --description 'Skip broken microcode entries when loading a microcode data file, instead of …'
complete -c iucode-tool -l no-ignore-broken --description 'Abort program execution if a broken microcode is found while loading a microc…'
complete -c iucode-tool -s s --description 'Select microcodes by the specified signature, processor flags mask (pf_mask),…'
complete -c iucode-tool -s S -l scan-system --description 'Select microcodes by scanning online processors on this system for their sign…'
complete -c iucode-tool -l date-before -l date-after --description 'Limit the selected microcodes by a date range.'
complete -c iucode-tool -l loose-date-filtering --description 'When a date range is specified, all revisions of the microcode will be consid…'
complete -c iucode-tool -l strict-date-filtering --description 'When a date range is specified, select only microcodes which are within the d…'
complete -c iucode-tool -s l -l list --description 'List selected microcode signatures to standard output (stdout).'
complete -c iucode-tool -s L -l list-all --description 'List all microcode signatures while they\'re being processed to standard outpu…'
complete -c iucode-tool -s k -l kernel --description 'Upload selected microcodes to the kernel.'
complete -c iucode-tool -s K -l write-firmware --description 'Write selected microcodes with the file names expected by the Linux kernel fi…'
complete -c iucode-tool -o wfile -l write-to --description 'Write selected microcodes to a file in binary format.'
complete -c iucode-tool -l write-earlyfw --description 'Write selected microcodes to an early initramfs archive, which should be prep…'
complete -c iucode-tool -o Wdirectory -l write-named-to --description 'Write selected microcodes to the specified directory, one microcode per file,…'
complete -c iucode-tool -l write-all-named-to --description 'Write every microcode to the specified directory, one microcode per file, in …'
complete -c iucode-tool -l overwrite --description 'Remove the destination file before writing, if it exists and is not a directo…'
complete -c iucode-tool -l no-overwrite --description 'Abort if the destination file already exists.'
complete -c iucode-tool -l mini-earlyfw --description 'Optimize the early initramfs cpio container for minimal size.'
complete -c iucode-tool -s w --description '.'
complete -c iucode-tool -s W --description '.'
complete -c iucode-tool -l normal-earlyfw --description 'Optimize the early initramfs size for tool compatibility.'

