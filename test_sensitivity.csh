foreach file($*)

    set no_wimp = `grep lh $file | grep mDM | head -1 | awk '{print $3}'`
    set with_wimp = `grep lh $file | grep mDM | tail -1 | awk '{print $3}'`

    python calc_significance.py $no_wimp $with_wimp 2 

end
