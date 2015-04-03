@ i = 0
while ( $i < 8 )
    @ start = `echo $i | awk '{print 125*$1}'`
    echo csh run_many_fits.csh $start log_FIT0001_"$i".log 
    csh run_many_fits.csh $start >& log_FIT0001_"$i".log &
    @ i += 1
end
