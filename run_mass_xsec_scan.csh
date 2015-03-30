foreach file ($1)
    # MC
    #set logfile_name = "log_files/log_"`basename $file dat`"log"
    # Data
    set logfile_name = "log_files/log_data.log"
    touch $logfile_name
    foreach mass(5 10 15)
        foreach xsec(4e-42 7e-42 10e-42)
            #echo $file
            #echo $logfile_name
            # MC
            #echo python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec $file --batch 
            #python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec $file --batch >>& $logfile_name

            # Data
            echo python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec --batch 
            python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec --batch >>& $logfile_name
        end
    end
end
