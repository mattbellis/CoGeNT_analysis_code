foreach file ($1)
    # MC
    set logfile_name = "log_files/log_MASS_SCAN_FIT0001D_"`basename $file dat`"log"
    # Data
    #set logfile_name = "log_files/log_data.log"
    rm -f $logfile_name
    python fit_cogent_data.py $file --batch >& $logfile_name
    #foreach mass(7 10 12)
    foreach mass(7)
        #foreach xsec(2e-42 5e-42 7e-42 10e-42)
        foreach xsec(7e-42)

            echo $file
            echo $logfile_name
            # MC
            echo python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec $file --batch 
            python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec $file --batch >>& $logfile_name

            # Data
            #echo python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec --batch 
            #python fit_cogent_data.py --fit 2 --mDM $mass --sigma_n $xsec --batch >>& $logfile_name

        end
    end
end
