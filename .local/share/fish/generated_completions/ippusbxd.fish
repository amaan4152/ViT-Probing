# ippusbxd
# Autogenerated from man page /usr/share/man/man8/ippusbxd.8.gz
complete -c ippusbxd -s h -l help --description 'Show help message.'
complete -c ippusbxd -s v -l vid --description 'USB vendor ID of desired printer.'
complete -c ippusbxd -s m -l pid --description 'USB product ID of desired printer.'
complete -c ippusbxd -s s -l serial --description 'Serial number of desired printer.'
complete -c ippusbxd -l bus -l device -l bus-device --description 'USB bus and device numbers where the device is currently connected (see outpu…'
complete -c ippusbxd -s p -l only-port --description 'Port number ippusbxd will expose the printer over.'
complete -c ippusbxd -s P -l from-port --description 'Port number ippusbxd will expose the printer over.'
complete -c ippusbxd -s i -l interface --description 'Network interface to use.  Default is the loopback interface (lo, localhost).'
complete -c ippusbxd -s l -l logging --description 'Send all logging to syslog.'
complete -c ippusbxd -s q -l verbose --description 'Enable verbose logging.'
complete -c ippusbxd -s d -l debug --description 'Enables debug mode.  Implies -q and -n.'
complete -c ippusbxd -s n -l no-fork --description 'Enables no fork mode.  Disables deamonization.'
complete -c ippusbxd -s B -l no-broadcast --description 'No-broadcast mode, do not DNS-SD-broadcast.'
complete -c ippusbxd -s N -l no-printer --description 'No-printer mode, debug/developer mode which makes ippusbxd run without IPP-ov…'
