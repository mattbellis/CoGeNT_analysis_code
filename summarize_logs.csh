#printf "ls_ncalc2\tuncer\tls_ncalc3\tuncer\tnum_neutrons\tuncer\tt_surf\tuncer\tk1_surf\tuncer\tt_exp_flat\tuncer\tnum_surf\tuncer\tnum_comp\tuncer\te_exp_flat\n"
foreach file($*)
    set surface0 = `grep org_surface $file | awk '{print $2}'`
    set neutron0 = `grep org_neutron $file | awk '{print $2}'`
    set compton0 = `grep org_compton $file | awk '{print $2}'`
    set lshell0 = `grep org_lshell $file | awk '{print $2}'`
    #set surface0 = `echo $file | awk -F"_" '{print $7}'`
    #set neutron0 = `echo $file | awk -F"_" '{print $11}'`
    #set compton0 = `echo $file | awk -F"_" '{print $15}'`
    #set lshell0 = `echo $file | awk -F"_" '{print $19}'`
  set A = `grep ls_ncalc2 $file | tail -1| awk '{print$2}'`
  set B = `grep ls_ncalc2 $file | tail -1| awk '{print$4}'`
  set C = `grep ls_ncalc3 $file | tail -1| awk '{print$2}'`
  set D = `grep ls_ncalc3 $file | tail -1| awk '{print$4}'`
  set E = `grep num_neutrons $file | tail -1| awk '{print$2}'`
  set F = `grep num_neutrons $file | tail -1| awk '{print$4}'`
  #set G = `grep t_surf $file | tail -1| awk '{print$2}'`
  #set H = `grep t_surf $file | tail -1| awk '{print$4}'`
  set G = `grep l-shell $file | tail -1| awk '{print$3}'`
  set H = `grep l-shell $file | tail -1| awk '{print$5}'`
  set I = `grep k1_surf $file | tail -1| awk '{print$2}'`
  set J = `grep k1_surf $file | tail -1| awk '{print$4}'`
  #set K = `grep t_exp_flat $file | tail -1| awk '{print$2}'`
  #set L = `grep t_exp_flat $file | tail -1| awk '{print$4}'`
  set K = `grep k2_surf $file | tail -1| awk '{print$2}'`
  set L = `grep k2_surf $file | tail -1| awk '{print$4}'`
  set M = `grep num_surf $file | tail -1| awk '{print$2}'`
  set N = `grep num_surf $file | tail -1| awk '{print$4}'`
  set O = `grep num_comp $file | tail -1| awk '{print$2}'`
  set P = `grep num_comp $file | tail -1| awk '{print$4}'`
  #set Q = `grep e_exp_flat $file | tail -1| awk '{print$2}'`
  #set R = `grep e_exp_flat $file | tail -1| awk '{print$4}'`
  #printf "$surface0\t$neutron0\t$compton0\t$lshell0\t$A\t$B\t$C\t$D\t$E\t$F\t$G\t$H\t$I\t$J\t$K\t$L\t$M\t$N\t$O\t$P\t$Q\t$R\n"
  printf "$surface0\t$neutron0\t$compton0\t$lshell0\t$A\t$B\t$C\t$D\t$E\t$F\t$G\t$H\t$I\t$J\t$K\t$L\t$M\t$N\t$O\t$P\n"
end
