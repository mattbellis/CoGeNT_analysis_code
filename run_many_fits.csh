@ start = `echo $1 | awk '{print $1}'`
echo here
@ stop = $start + 125
@ i = $start

#foreach file ($*)
while ( $i < $stop)
    #echo $file
    echo 
    set search_string = `printf "samples_%03d.dat" $i`
    #echo $search_string
    set file = `ls MC_files/sample_FIT0001_*100k* | grep $search_string | tail -1`
    set logfile_name = "log_files/log_FIT0001B_"`basename $file dat`"log"
    
    echo $file
    echo $logfile_name

    echo python fit_cogent_data.py $file --batch >& $logfile_name
    python fit_cogent_data.py $file --batch >& $logfile_name

    @ i += 1
end
