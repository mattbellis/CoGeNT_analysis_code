@ i = 0
while ( $i < 8 )
    @ start = `echo $i | awk '{print 125*$1}'`

    #echo csh run_many_fits.csh $start log_FIT0001B_"$i".log 
    #csh run_many_fits.csh $start >& log_FIT0001B_"$i".log &
    echo csh run_many_mass_sec_scans.csh $start log_FIT0002B_"$i".log 
    csh run_many_mass_sec_scans.csh $start >& log_FIT0002B_"$i".log &

    @ i += 1
end
