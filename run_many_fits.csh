foreach file ($*)
    echo $file
    set logfile_name = "log_files/log_"`basename $file dat`"log"
    echo $logfile_name
    #python fit_cogent_data.py $file --batch >& $logfile_name
end
