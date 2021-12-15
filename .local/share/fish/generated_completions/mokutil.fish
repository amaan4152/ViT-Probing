# mokutil
# Autogenerated from man page /usr/share/man/man1/mokutil.1.gz
complete -c mokutil -s l -l list-enrolled --description 'List the keys the already stored in the database.'
complete -c mokutil -s N -l list-new --description 'List the keys to be enrolled.'
complete -c mokutil -s D -l list-delete --description 'List the keys to be deleted.'
complete -c mokutil -s i -l import --description 'Collect the followed files and form a enrolling request to shim.'
complete -c mokutil -s d -l delete --description 'Collect the followed files and form a deleting request to shim.'
complete -c mokutil -l revoke-import --description 'Revoke the current import request (MokNew).'
complete -c mokutil -l revoke-delete --description 'Revoke the current delete request (MokDel).'
complete -c mokutil -s x -l export --description 'Export the keys stored in MokListRT.'
complete -c mokutil -s p -l password --description 'Setup the password for MokManager (MokPW).'
complete -c mokutil -s c -l clear-password --description 'Clear the password for MokManager (MokPW).'
complete -c mokutil -l disable-validation --description 'Disable the validation process in shim.'
complete -c mokutil -l enrolled-validation --description 'Enable the validation process in shim.'
complete -c mokutil -l sb-state --description 'Show SecureBoot State.'
complete -c mokutil -s t -l test-key --description 'Test if the key is enrolled or not.'
complete -c mokutil -l reset --description 'Reset MOK list.'
complete -c mokutil -l generate-hash --description 'Generate the password hash.'
complete -c mokutil -l hash-file --description 'Use the password hash from a specific file.'
complete -c mokutil -s P -l root-pw --description 'Use the root password hash from /etc/shadow.'
complete -c mokutil -s s -l simple-hash --description 'Use the old SHA256 password hash method to hash the password .'
complete -c mokutil -l ignore-db --description 'Tell shim to not use the keys in db to verify EFI images.'
complete -c mokutil -l use-db --description 'Tell shim to use the keys in db to verify EFI images (default).'
complete -c mokutil -s X -l mokx --description 'Manipulate the MOK blacklist (MOKX) instead of the MOK list.'
complete -c mokutil -l import-hash --description 'Create an enrolling request for the hash of a key in DER format.'
complete -c mokutil -l delete-hash --description 'Create an deleting request for the hash of a key in DER format.'
complete -c mokutil -l set-verbosity --description 'Set the SHIM_VERBOSE to make shim more or less verbose.'
complete -c mokutil -l pk --description 'List the keys in the public Platform Key (PK).'
complete -c mokutil -l kek --description 'List the keys in the Key Exchange Key Signature database (KEK).'
complete -c mokutil -l db --description 'List the keys in the secure boot signature store (db).'
complete -c mokutil -l dbx --description 'List the keys in the secure boot blacklist signature store (dbx).'
