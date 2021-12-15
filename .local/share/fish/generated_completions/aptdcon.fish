# aptdcon
# Autogenerated from man page /usr/share/man/man1/aptdcon.1.gz
complete -c aptdcon -s v -l version --description 'Show the version number of the aptdcon.'
complete -c aptdcon -s h -l help --description 'Show information about the usage of the command.'
complete -c aptdcon -s d -l debug --description 'Show additional information on the command line.'
complete -c aptdcon -s i -l install --description 'Install the list of PACKAGES.'
complete -c aptdcon -l reinstall --description 'Reinstall the list of PACKAGES.'
complete -c aptdcon -s r -l remove --description 'Remove the list of PACKAGES.'
complete -c aptdcon -s p -l purge --description 'Purge the list of PACKAGES.'
complete -c aptdcon -s u -l upgrade --description 'Upgrade the list of PACKAGES.'
complete -c aptdcon -l upgrade-system --description 'Upgrade the whole system.'
complete -c aptdcon -l fix-install --description 'Try to complete a previously cancelled installation by calling "dpkg --config…'
complete -c aptdcon -l fix-depends --description 'Try to resolve unsatisified dependencies.'
complete -c aptdcon -l add-vendor-key --description 'Install the PUBLIC_KEY_FILE to authenticate and trust packages singed by the …'
complete -c aptdcon -l add-vendor-key-from-keyserver --description 'Download and install the PUBLIC_KEY_ID to authenticate and trust packages sin…'
complete -c aptdcon -l key-server --description 'Download vendor keys from the given KEYSERVER.'
complete -c aptdcon -l remove-vendor-key --description 'Remove the vendor key of the given FINGERPRINT to no longer trust packages fr…'
complete -c aptdcon -l add-repository --description 'Allow to install software from the repository specified by the given  DEB_LIN…'
complete -c aptdcon -l sources-file --description 'Specify an alternative sources file to which the new repository should be wri…'
complete -c aptdcon -l list-trusted-vendors --description 'Show all trusted software vendors and theirs keys.'
complete -c aptdcon -l hide-terminal --description 'Do not attach to the interactive terminal of the underlying dpkg call.'
complete -c aptdcon -l allow-unauthenticated --description 'Allow to install packages which are not from a trusted vendor.'

