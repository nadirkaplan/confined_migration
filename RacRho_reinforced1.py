import math as m
import numpy as np
import User_functions_R as usf
import Traction_R as T
import Check_traction as C
from random import random 
from datetime import date
import pdb

#One can modify 'conf_R = 200.0' to simulate different curvatures and 'conf_width = 8.0' to simulate different confinements.


def migration_simulator(uniform, duro, K_s, K_a, K_l, conf_width, gama_sub, opt, nnum):

    fvr_out = 1; #output vs, vr, vp, fsub ... for postprocessing
    if opt == 0:
        dt_min = 200/120/K_s/15; 
    else:
        dt_min = 200/120/K_l/15;  
    dt = min(0.0015, dt_min);  #  step size 
    Nsteps = int(round(4000/dt/1e4)*1e4) ;
    save_memb_inv = int(10000); # save the membrane coordinates every # steps
    save_nuc_inv = int(100); # save the nucleus coordinates and forces every # steps
    memb_nuc_times = save_memb_inv/save_nuc_inv;
    Step_inv = int(10000) ;  #
    zeta_step = int(60/dt) ;
    conf_R = 200.0 ; 

    gama_id = np.amax(gama_sub*nnum/16*100)/100 ; 
    ks_id = np.around(K_s*nnum/16*100)/100 ; 
    ka_id = np.around(K_a*nnum/16*100)/100 ; 
    kl_id = np.around(K_l*nnum/16*100)/100 ; 
    fname = 'CBiphasic1'+'_W'+str(conf_width)+'_R'+str(conf_R)+'_Uni'+str(uniform)+'_Ka'+str(round(kl_id*100)/100)+'_eta'+str(round(gama_id*100)/100)
    savegifname = fname+'.gif' ;

    today = date.today()
    dd = today.strftime("%y_%m_%d")

    filetarray=fname+'_tarray_'+dd+'.dat'
    filemembX=fname+'_Xmemb_'+dd+'.dat'
    filemembY=fname+'_Ymemb_'+dd+'.dat'
    filenucX=fname+'_Xnuc_'+dd+'.dat'
    filenucY=fname+'_Ynuc_'+dd+'.dat'
    fileFsubdiff=fname+'_Fsub_diff_'+dd+'.dat'

    shift_dist1 = 0.0  #conf_R  #*np.sin(0.5*np.pi);
    shift_dist2 = -conf_R  #-conf_R*np.cos(0.5*np.pi);
    rac_tol = 0.0002*8/nnum; 
    Rac_cyto_tol = 1e-9 ; 
    Rcell = float(3.0) ; 
    R_nuc0 = float(2.0) ; 
    Area_nuc0 = np.pi*R_nuc0**2.0 ; 


    # define parameters of chemical modeling
    gama_r = float(0.3) # antagonistic effect of Rho to Rac
    gama_rho = float(0.3) # antagonistic effect of Rac to Rho 
    kb_plus = float(0.3)  # 2.4E-1 # Kp+  baseline Rac1 activation coefficient
    kb_minus = float(0.8); # 6.0E-1  # Kb-    baseline Rac1 inactivation coefficient
    kpolar_minus = float(0.0); #  # Kp-
    kapbb_plus = float(0.3);  #  # kap_p+  2.8E-1 # baseline Rho activation rate 
    kapbb_minus = float(0.8); #  # kap_b-  6.0E-1  # baseline Rho inactivation rate 
    kap_polar_minus = float(0.0); #   # kap_p- 
    beta_r = float(0.3); # 
    beta_rho = float(0.3); #  
    M_plus =  float(0.02); # float(0.1)  # Rac1 membrane association dissociation rates  
    M_minus = float(0.02); # float(0.05/nnum )  
    gg_val = float(0.0);  # float(0.08) 
    mu_plus = float(0.02); # float(0.1)  # RhoA membrane association dissociation rates  
    mu_minus = float(0.02);  # float(0.05/nnum ) 
    hh_val = float(0.0);  # float(0.08);  
    D = 0.010 ; 
    lamb = 36*1e-6 ; 
    K_repulsive = 1000.
    thresh_angle = np.pi/30;
    fr_coef = 2.0
    mu_coef = 2.0


    AA = 1.32;   BB = 0.26;   
    Nuc_max = 3.0;   Amax = 1900;    Area0 = 78;   A_soft = 400 ; 
    eta_nuc0 = float(860)*1e-6;  # 0.01 pn*s/um^2 = 0.01*1e-6 pn*s/nm^2
    ksub0 = 0.01;   ksub_max = 100;
    if opt == 0: ksub = K_s;
    else: ksub = K_l;
    [eta_nuc, Area_nuc, R_nuc] = usf.nuc_eta_area(ksub, Area0, A_soft, Area_nuc0, lamb, eta_nuc0, conf_width)
    
    # define parameters of mechanics modeling
    alpha = 3.0 
    kc = 120.0 
    ks = 0.25 
    N_M = 100.0*16/nnum   
    fs = 1.0
    fm = 2.0  #/fs
    epsilon = 0.0  #/fs
    zeta = 0.5
    fc = 0.50   #*fs
    fcr = 2.5
    v0 = 120.0 
    V_p = 120.0  #average polymerization rate
    N_C0 = 100.0*16/nnum 
    K_clutch = 2.0 
    K_memb = 10.0*nnum/16 #membrane stiffness  pn/um  
    eta_memb = 0.05 #membrane viscoelastic coefficient    pn*s/um
    K_ck0 = 1.25*16/nnum #Microtubules stiffness pn/um   *Rcell
    K_ck1 = 2.8*16/nnum #Microtubules stiffness pn/um   *Rcell
    f_lim = 100.0  # 
        

    Ini_Raca = float(0.3)
    Ini_Raci = float(0.2)
    Ini_Rhoa = float(0.3)
    Ini_Rhoi = float(0.2)
    rac_0  = Ini_Raca/nnum
    rho_0  = Ini_Rhoa/nnum
    polar_num = int(nnum/4)
    #conv_iter = np.zeros([Nsteps, 2]);
    #trigger_ary = np.zeros([Nsteps, nnum])
    #repulsive_fxary = np.zeros([nnum, Nsteps])
    #repulsive_fyary = np.zeros([nnum, Nsteps])
    #angles_fibers = np.zeros([nnum, Nsteps])
    coords_scale = np.linspace(0.0, 1.0, nnum+1)

    xcell = Rcell*np.cos(2*m.pi*coords_scale[0:nnum])
    ycell = Rcell*np.sin(2*m.pi*coords_scale[0:nnum])
    edge_L0 = np.sqrt(2.*(Rcell**2.)*(1 - np.cos(2*m.pi*coords_scale[1])))
    eglen_max = edge_L0;
    
    ''' 
    Pre-define values of chemo-singals 
    ''' 
    Rac_cyto = np.zeros([Nsteps]) 
    #Rho_cyto = np.zeros([Nsteps]) 
    F_pro = np.zeros([nnum]);
    #R_len_ary = np.zeros([nnum, Nsteps])
    R_len_new = np.zeros([nnum]);
    R_len_old = np.zeros([nnum]);
    
    Force_ck_new = np.zeros([nnum])
    Force_ck_old = np.zeros([nnum])
    F_wall = np.zeros([nnum])

    #Fsub_diff = np.zeros([Nsteps,2]);
    Fsub_diff = np.zeros([int(Nsteps/save_nuc_inv+1),2]);

    cycid=1;
    Ini_RacRho = usf.Initialize_RacRho(Ini_Raca, Ini_Raci, Ini_Rhoa, Ini_Rhoi, polar_num, nnum, uniform, cycid);
    rac_an=Ini_RacRho[0];  rac_in=Ini_RacRho[1];
    rho_an=Ini_RacRho[2];  rho_in=Ini_RacRho[3];

    N_C = N_C0 * np.ones(nnum) 
    nc = N_C ; 
    nm = N_M * np.ones(nnum) 
    Rcn =   1- (np.sum(rac_an) + np.sum(rac_in))   #assign value 
    Rhocn = 1- (np.sum(rho_an) + np.sum(rho_in))   #assign value 
    Rac_cyto[0] = Rcn 
    #Rho_cyto[0] = Rhocn 
    Avg_Rac1_mb = np.mean(rac_an) 
    Avg_Rhoa_mb = np.mean(rho_an) 

    Fsub_val = np.zeros([nnum]);

    #Pre-define mechanical responses
    xnuc_pst = 0.0 + shift_dist1;
    ynuc_pst = 0.0 + shift_dist2;
    xnuc_pst0 = usf.assign_scalar(xnuc_pst);
    ynuc_pst0 = usf.assign_scalar(ynuc_pst);
    xnuc_all = np.zeros([int(Nsteps/save_nuc_inv+1)]);  #Nucleus velocity and position vectors
    xnuc_all[0] = xnuc_pst;
    ynuc_all = np.zeros([int(Nsteps/save_nuc_inv+1)]);
    xnuc_all[0] = xnuc_pst;
    ynuc_all[0] = ynuc_pst;

    #xcen_all = np.zeros([Nsteps]);  #Nucleus velocity and position vectors
    #ycen_all = np.zeros([Nsteps]);
    Area = np.zeros([int(Nsteps/save_nuc_inv+1)]);  #Cell areas

    #Ro_all = np.zeros([nnum, Nsteps])  #bonded clutch density
    #f_all = np.zeros([nnum, Nsteps]) #molecular clutch forces
    Ro_all0 = np.zeros([nnum])  #bonded clutch density
    Ro_all = np.zeros([nnum])  #bonded clutch density
    f_all = np.zeros([nnum]) #molecular clutch forces
    xc_all = np.zeros([nnum]) #displacements of actin
    xs_all = np.zeros([nnum]) #displacements of substrate

       
    Vs = np.zeros([nnum]) ;                Vm0 = np.zeros([nnum]);
    xcoord = np.zeros([nnum+1]);  xcoord[:nnum] = xcell + shift_dist1;   xcoord[nnum] = xcoord[0];
    ycoord = np.zeros([nnum+1]);  ycoord[:nnum] = ycell + shift_dist2;   ycoord[nnum] = ycoord[0];
    xcoord0 = np.zeros([nnum+1]);  xcoord0 = usf.assign_values(xcoord);
    ycoord0 = np.zeros([nnum+1]);  ycoord0 = usf.assign_values(ycoord);
    xcoord_tem = np.zeros([nnum+1]);  xcoord_tem = usf.assign_values(xcoord);
    ycoord_tem = np.zeros([nnum+1]);  ycoord_tem = usf.assign_values(ycoord);
    xcoord_all = np.zeros([nnum+1, int(Nsteps/save_memb_inv+1)]);  
    ycoord_all = np.zeros([nnum+1, int(Nsteps/save_memb_inv+1)]);
    Rnuc_all = np.zeros([int(Nsteps/save_memb_inv+1)]);
    xcoord_all[:,0] = xcoord;            ycoord_all[:,0] = ycoord;
    memb_step = 0;
    nuc_step = 0;
    nuc_mem_vx = xcoord_all[:nnum,0] - xnuc_all[0];  # direction vectors of every membrane pts 
    nuc_mem_vy = ycoord_all[:nnum,0] - ynuc_all[0];  # direction vectors from nucleus to membrane
    [nm_pts_vx, nm_pts_vy, R_len, Aa] = usf.Normalize_vector(nuc_mem_vx, nuc_mem_vy) #normalized position vectors
    #R_len_ary[:,0] = R_len;
    R_len_new = usf.assign_values(R_len)
    R_len_old = usf.assign_values(R_len)
    Area[0] = Aa;     Aa0 = Area[0];     
    #edge_ang_all = np.zeros([nnum, Nsteps]);
    edge_ang_all = np.zeros([nnum, Step_inv]);   # all edge angles of very time step

    ang_diff_all = np.zeros([nnum]);
    edge_ang_avg = np.zeros([nnum, int(Nsteps/Step_inv)+1]);
    avgang_diff_all = np.zeros([nnum, int(Nsteps/Step_inv)+1]);
    cr_all = np.zeros([nnum]);
    dr_all = np.zeros([nnum]);
    Fst= np.zeros([nnum]);
    #strain_energy= np.zeros([3, Nsteps]);
    Fstall=np.zeros([nnum]);
    #step_ary = np.zeros([nnum, int(Nsteps/100)]);
    fa_bond = np.zeros([nnum]);
    #fa_bond_ary = np.zeros([nnum, Nsteps]);

    '''
    Calculate and assemble global derivative matrix
    '''
    vec_edges=usf.edge_vectors(xcell, ycell);  #get edge length and direction vectors
    edge_ang=usf.edge_angles(vec_edges, nnum);
    edge_ang_all[:,0] = edge_ang;
    Angle0 = edge_ang_all[0,0];
    edge_len = usf.edge_length(xcell, ycell) 
    tm = float(0.0)
    time = np.zeros([int(Nsteps/save_nuc_inv+1)] )
    #RacRho_last= np.zeros([4*nnum + 2] )
    RacRho_current= np.zeros([4*nnum + 2])
    RacRho_inputs= np.zeros([4*nnum + 2] )
    Step_inv_no = 0;      steady_step_num = np.zeros([20000]);  
    #xcoord=xcoord_all[:, 0];       ycoord=ycoord_all[:, 0];
    for i in range(Nsteps-1):
        #if i%20000<1e-3: print('Nsteps=', i);    print('Area', Aa);    print('Fwall', F_wall)
        if i%200000<1e-3 and i>0:
            np.savetxt(fileFsubdiff, Fsub_diff.transpose(),  fmt='%.6e', delimiter='	');
            #print('Nsteps=', i)
            np.savetxt(filetarray, time.transpose(),  fmt='%.6e', delimiter='\t');
            np.savetxt(filenucX, xnuc_all.transpose(),  fmt='%.6e', delimiter='\t');
            np.savetxt(filenucY, ynuc_all.transpose(),  fmt='%.6e', delimiter='\t');
            np.savetxt(filemembX, xcoord_all.transpose(),  fmt='%.6e', delimiter='\t');
            np.savetxt(filemembY, ycoord_all.transpose(),  fmt='%.6e', delimiter='\t');
            
        if i%Step_inv<1e-3 and i>0:
            step_bn=Step_inv_no*Step_inv;   Step_inv_no += 1;   step_en=Step_inv_no*Step_inv-1;
            for nn in range(nnum):
                #edge_ang_avg[nn, Step_inv_no] = np.average(edge_ang_all[nn,step_bn:step_en]);
                edge_ang_avg[nn, Step_inv_no] = np.average(edge_ang_all[nn,:]);
            avgang_diff_all[:,Step_inv_no] = (edge_ang_avg[:,Step_inv_no]-edge_ang_avg[:,Step_inv_no-1]);
        
        if i%Step_inv<1e-3 and Step_inv_no>1:
            avgang_diff0 = np.zeros(nnum);
            for ij in range(nnum): 
                avgang_diff0[ij] = np.amax(np.absolute(avgang_diff_all[ij,0:Step_inv_no]));
            avgang_diff = (edge_ang_avg[:,Step_inv_no]-edge_ang_avg[:,Step_inv_no-1]);
            polar_or_contract = (edge_ang_avg[:,Step_inv_no] - Angle0);  # + in contraction; - in protrusion

        for j in range(nnum):
            ang_diffj = ang_diff_all[j];
            if Step_inv_no<=1:
                cr = 1.0; dr = 1.0;
                cr_all[j]=cr;   dr_all[j]=dr; 
            elif i%Step_inv<1e-3 and Step_inv_no>1:
                avgang_diff_tol = np.absolute(avgang_diff);
                avgang_diff0_tol = np.absolute(avgang_diff0);   
                if avgang_diff_tol[j]<0.3*avgang_diff0_tol[j] and polar_or_contract[j] > 0.3*Angle0:
                    coeff_cr=np.absolute(2*ang_diffj)/Angle0;   cr = m.exp(coeff_cr);
                elif avgang_diff_tol[j]<0.3*avgang_diff0_tol[j] and polar_or_contract[j] < -0.3*Angle0:
                    coeff_dr=np.absolute(2*ang_diffj)/Angle0;   dr = m.exp(coeff_dr);
                else:
                    cr = 1.0; dr = 1.0;
                cr_all[j]=cr;   dr_all[j]=dr; 

        tm += dt
        #time[i+1]= tm
        if i<1: Rac_cyto_diff = 1;
        else:   Rac_cyto_diff = abs(Rac_cyto[i] - Rac_cyto[i-2]); 

        if Rac_cyto_diff < Rac_cyto_tol:       
            Ini_RacRho = usf.Initialize_RacRho(Ini_Raca, Ini_Raci, Ini_Rhoa, Ini_Rhoi, polar_num, nnum, uniform, cycid);

            rac_an = Ini_RacRho[0];   rac_in = Ini_RacRho[1];
            rho_an = Ini_RacRho[2];   rho_in = Ini_RacRho[3];
            Rac_cyto[i] = 1- (np.sum(rac_an) + np.sum(rac_in))   #assign value 
            #Rho_cyto[i] = 1- (np.sum(rho_an) + np.sum(rho_in))   #assign value
            Rcn =   1- (np.sum(rac_an) + np.sum(rac_in))   #assign value 
            Rhocn = 1- (np.sum(rho_an) + np.sum(rho_in))   #assign value
            steady_step_num[cycid] = i; cycid += 1;

        RacRho_current = usf.Assemble_RacRho(rac_an, rac_in, rho_an, rho_in, Rcn, Rhocn, nnum)
        rac_a = usf.assign_values(rac_an);   rac_i = usf.assign_values(rac_in);
        rho_a = usf.assign_values(rho_an);   rho_i = usf.assign_values(rho_in);
        Rc = usf.assign_scalar(Rcn);         Rhoc = usf.assign_scalar(Rhocn)
        
        '''
        # update Rac and Rho signaling  
        '''
        tol = 1E-6 
        maxiter = 20 
        eps = 1E3*tol 
        iter = int(0) 
        while eps > tol and iter < maxiter:
            iter += 1
            K_plus = np.zeros([nnum,3]);    K_minus = np.zeros([nnum]);
            kappa_p = np.zeros([nnum,3]);   kappa_m = np.zeros([nnum]);
            for j in range(nnum):
                rac_aj = rac_a[j];     rac_ij = rac_i[j];
                rho_aj = rho_a[j];     rho_ij = rho_i[j];
                Rj_len = R_len[j];     ncj = nc[j];    nmj = nm[j];
                vsj = Vs[j];           #angj = edge_ang_all[j,i];
                angj = edge_ang[j];
                K_plus[j,:]  = usf.Rac1_activation(kb_plus, gama_r, beta_r, rho_aj, rac_aj, \
                                                       rac_0, rho_0, cr_all[j], norder = 3.0)
                K_minus[j] = usf.Rac1_inactivation(kb_minus, kpolar_minus, angj, Angle0)
                kappa_p[j,:] = usf.RhoA_activation(kapbb_plus, gama_rho, beta_rho, rac_aj, rho_aj, \
                                                       rac_0, rho_0, dr_all[j], norder = 3.0)
                kappa_m[j] = usf.RhoA_inactivation(kapbb_minus, kap_polar_minus, angj, Angle0)
            matrix_res = usf.global_matrix_residue(rho_a, rac_a, rho_i, rac_i, rho_an, rac_an,  \
                     rho_in, rac_in, K_plus, K_minus, kappa_p, kappa_m, M_plus, M_minus, \
                     mu_plus, mu_minus, D, edge_len, Rc, Rhoc, Rcn, Rhocn, nnum, dt ) 
            Delta_RacRho = - np.linalg.solve(matrix_res[0], matrix_res[1])
            eps = np.linalg.norm(Delta_RacRho)
            RacRho_current += Delta_RacRho
            RacRho_inputs = usf.Disassemble_RacRho(RacRho_current, nnum)
            rac_a = RacRho_inputs[0];   rac_i = RacRho_inputs[1];
            rho_a = RacRho_inputs[2];   rho_i = RacRho_inputs[3];
            Rc = RacRho_inputs[4];      Rhoc = RacRho_inputs[5];
        
        rho_an = usf.assign_values(rho_a);  rac_an =  usf.assign_values(rac_a);
        rho_in = usf.assign_values(rho_i);  rac_in =  usf.assign_values(rac_i);
        Rcn = RacRho_inputs[4];             Rhocn = RacRho_inputs[5];
        Rac_cyto[i+1] = RacRho_inputs[4];   #Rho_cyto[i+1] = RacRho_inputs[5];
                
        if iter >= maxiter-1: print('Cannot converge at step =',i); break
        
        #update/calculate N_clutch and N_myosin based on chemical signaling
        Avg_Rac1_mb = np.mean(rac_a);  #np.amax(RacRho_pts[0])
        Avg_Rhoa_mb = np.mean(rho_a);  #np.amax(RacRho_pts[0])
        nc = rac_a/Avg_Rac1_mb*N_C;   # RacRho_pts[0]/RacRho_pts[2]*N_C;  #
        nm = rho_a/Avg_Rhoa_mb*N_M;   # RacRho_pts[2]/RacRho_pts[0]*N_M;  #
        nc = np.minimum(nc, 3*N_C);     nm = np.minimum(nm, 3*N_M);
        nc = np.maximum(nc, 0.3*N_C);   nm = np.maximum(nm, 0.3*N_M);
        Vp = np.minimum(rac_a/Avg_Rac1_mb, 3)*V_p;
        

        if Rac_cyto_diff < Rac_cyto_tol: 
            fa_bond = np.zeros([nnum]);
            if zeta > 0.0:
               for j in range(nnum):
                   F_p = F_pro[j];   ncj = nc[j];    nmj = nm[j];
                   fa = C.check_faval(alpha, epsilon, ks, kc, fc, ncj, nmj, fm, v0, opt, K_a, K_l, K_s, K_clutch, gama_sub, F_p) ;
                   fa_bond[j] = fa;
               #fa_bond_ary[:, i] = fa_bond;            

        disp_tol = 1e-3; iteration =0;
        fbond = np.zeros([nnum]);      
        xc0_all = np.zeros([nnum]);    xs0_all = np.zeros([nnum]); 
        for j in range(nnum):
            xc0_all[j] = xc_all[j];    xs0_all[j] = xs_all[j];

        while disp_tol>1e-5 and iteration < 2:
            iteration += 1;
            for j in range(nnum):
                Ro_0 = Ro_all0[j];  xc0 = xc0_all[j];  xs0 = xs0_all[j];
                fa = fa_bond[j];    f = f_all[j];
                #xcj = xcoord_aRac_cytoll[j, i];  ycj = ycoord_all[j, i];
                xcj = xcoord[j];   ycj = ycoord[j];  
                Pf_iter = 0;  fdiff = 1.0;   ro_diff = 1.0;   ff_0 = 0;   rro_0 = 0;  Pf_imax = 4; 

                while (abs(fdiff) > 1e-6  or   abs(ro_diff) > 1e-6) and Pf_iter< Pf_imax:
                    Ro_ary = T.closed_bond_prob(alpha, zeta, ks, kc, fa, fcr, fc, f, Ro_0, dt) ;
                    Ro = Ro_ary[0];

                    beta = Ro*nc[j]*K_clutch;     F_stall = nm[j]*fm;    F_p= F_pro[j]; 
                    xc_xs=T.update_disps(xs0, xc0, Ro_ary, nc[j], K_clutch, gama_sub, K_a, K_l, K_s, F_stall, F_p, v0, dt, opt) ;
                    f = K_clutch*(xc_xs[0] - xc_xs[1]);
                    
                    fdiff = f - ff_0 ;   ro_diff = Ro - rro_0;
                    ff_0 = usf.assign_scalar(f);    rro_0 = usf.assign_scalar(Ro);
                    Pf_iter += 1;

                    if Ro<1e-4 or f<0.0 or f>100: f =0.; Ro = 0.; xc = 0.; xs = 0.; Pf_iter = Pf_imax*2;
                    else:   xc = xc_xs[0]; xs = xc_xs[1]; 
                
                Ro_all[j] = Ro;  xc_all[j] = xc;  xs_all[j] = xs; 
                Fsub = Ro*nc[j]*f;  
                F_st = Fst[j];  
                vf =  T.Retrograde_velocity(F_st, F_stall, v0);
                fbond[j] = f;   Fsub_val[j] = Fsub;
                Fstall[j]= F_stall;
                Vs[j] = Vp[j] - vf;
                

            
            #update membrane (xcoord_all) shape and outward-direction vector (vec)
            xycoord=usf.coord_direction_update(Vs, xcoord0, ycoord0, nm_pts_vx, nm_pts_vy, dt, nnum);
            xcoord = xycoord[0];    ycoord = xycoord[1];

            #if i < 1: Del_force_ck = 0;  # calculate microtubule forces
            #else: Del_force_ck = T.microtubule_force(K_ck, Aa0, Aa, A_soft, R_len_old, R_len_new);
            if i < 1: Del_force_ck = 0;  # calculate microtubule forces
            else: Del_force_ck = T.microtubule_force(K_ck0, K_ck1, Aa0, Aa, A_soft, R_len_old, R_len_new);     

            Force_ck_new = Force_ck_old + Del_force_ck
            vec_edges=usf.edge_vectors(xcoord[:nnum], ycoord[:nnum]);
            edge_ang=usf.edge_angles(vec_edges, nnum); edge_ang_all[:,(i+1)%Step_inv] = edge_ang;
            ang_diff_all= usf.angle_diff(edge_ang, nnum);

            angles_between_fibers = T.calculate_angles(nm_pts_vx, nm_pts_vy);

            repulsive_force = T.repulsive_tangent_force_old(K_repulsive,  angles_between_fibers, nm_pts_vx, nm_pts_vy, thresh_angle);

            vm_eps =1.0;     vm_iter = 0;
            vm_xcoord = usf.assign_values(xcoord);
            vm_ycoord = usf.assign_values(ycoord);

            while vm_eps > 1e-6 and vm_iter < 4:
                vm_iter += 1;
                nuc_mem_vx = vm_xcoord[:nnum] - xnuc_pst; 
                nuc_mem_vy = vm_ycoord[:nnum] - ynuc_pst;
                [nm_pts_vx, nm_pts_vy, R_len, Aa] = usf.Normalize_vector(nuc_mem_vx, nuc_mem_vy)
	    
                Force_Vm=T.modified_friction_viscosity(vec_edges, K_memb, edge_L0, Force_ck_new, eta_memb, v0, Vs, Vm0, Vp, mu_coef, mu_coef, Fsub_val, Fstall, \
                       vm_xcoord, vm_ycoord, xnuc_pst, ynuc_pst, conf_R, conf_width, nm_pts_vx, nm_pts_vy, repulsive_force) ;
                       
                F_pro=Force_Vm[0] ;  
                Vm = Force_Vm[1] ; 
                F_wall = Force_Vm[2] ; 
                vm_eps = np.linalg.norm(Vm - Vm0) ; 
                xycoord=usf.coord_direction_update(Vm, xcoord, ycoord, nm_pts_vy, -nm_pts_vx, dt, nnum); 
                vm_xcoord = xycoord[0];
                vm_ycoord = xycoord[1];
                Vm0 = usf.assign_values(Vm);
            xcoord = xycoord[0];   ycoord = xycoord[1];

    
            #update nucleus velocity based on F_sub;
            Fst = (Fsub_val - F_pro);
            
            V_nuc = T.curved_nucleus_velocity_m(nm_pts_vx, nm_pts_vy, Fst, Force_ck_new, R_nuc0, R_nuc0, eta_nuc, xnuc_pst, ynuc_pst, conf_R, conf_width);
            
            #xnuc_all[i+1] = xnuc_all[i] + dt*V_nuc[0]*1e-3;
            #ynuc_all[i+1] = ynuc_all[i] + dt*V_nuc[1]*1e-3;
            xnuc_pst = xnuc_pst0 + dt*V_nuc[0]*1e-3;
            ynuc_pst = ynuc_pst0 + dt*V_nuc[1]*1e-3;
            
            #nuc_mem_vx = xcoord[:nnum] - xnuc_all[i+1]; 
            #nuc_mem_vy = ycoord[:nnum] - ynuc_all[i+1];
            nuc_mem_vx = xcoord[:nnum] - xnuc_pst; 
            nuc_mem_vy = ycoord[:nnum] - ynuc_pst;
            [nm_pts_vx, nm_pts_vy, R_len, Aa] = usf.Normalize_vector(nuc_mem_vx, nuc_mem_vy)
            #R_len_ary[:,i+1] = R_len;
            R_len_new = usf.assign_values(R_len)
            #Area[i+1] = Aa;
            #update nucleus viscosity 
            [eta_nuc, Area_nuc, R_nuc] = usf.nuc_eta_area(ksub, Aa, A_soft, Area_nuc0, lamb, eta_nuc0, conf_width)
            disp = np.sqrt((xcoord-xcoord_tem)**2 + (ycoord-ycoord_tem)**2) ;
            xcoord_tem = xycoord[0];   ycoord_tem = xycoord[1];
            disp_tol = abs(np.sum(disp)); 

        R_len_old = usf.assign_values(R_len_new)
        Force_ck_old = usf.assign_values(Force_ck_new)
        
        edge_len = usf.edge_length(xcoord[0:nnum], ycoord[0:nnum]);
        if np.amax(edge_len[0])>eglen_max:
            eglen_max = np.amax(edge_len[0]);
        f_all = usf.assign_values(fbond);
        Ro_all0 = usf.assign_values(Ro_all);
        xcoord0 = usf.assign_values(xcoord);
        ycoord0 = usf.assign_values(ycoord);
        xnuc_pst0 = usf.assign_scalar(xnuc_pst);
        ynuc_pst0 = usf.assign_scalar(ynuc_pst);
        Aa0 = usf.assign_scalar(Aa)
        
        if i%save_memb_inv < 1e-6:
            #pdb.set_trace()
            #if R_nuc>3: pdb.set_trace()
            memb_step += 1; 
            xcoord_all[:, memb_step] = xycoord[0];    ycoord_all[:, memb_step] = xycoord[1];
            Rnuc_all[memb_step] = R_nuc;

        if i%save_nuc_inv < 1e-6:
            nuc_step += 1;
            time[nuc_step]= tm
            Area[nuc_step] = Aa;
            xnuc_all[nuc_step] = xnuc_pst;
            ynuc_all[nuc_step] = ynuc_pst;
            for jj in range(nnum):
                Fsub_diff[nuc_step,0] += (xcoord0[jj]-xnuc_pst)/R_len[jj]*Fsub_val[jj];
                Fsub_diff[nuc_step,1] += (ycoord0[jj]-ynuc_pst)/R_len[jj]*Fsub_val[jj];
        
        #check fa now
        if zeta > 0.0:
            if i%zeta_step < 1e-3:
                for j in range(nnum):
                    F_p = F_pro[j] ;   ncj = nc[j];    nmj = nm[j];
                    fa=C.check_faval(alpha, epsilon, ks, kc, fc, ncj, nmj, fm, v0, opt, K_a, K_l, K_s, K_clutch, gama_sub, F_p);
                    
                    fa_bond[j] = fa;               
                #fa_bond_ary[:, i] = fa_bond;


        
    
    #strain_energy[2,:] = strain_energy[0,:] + strain_energy[1,:];
    print('K_memb',K_memb)
    print('eta_memb',eta_memb)
    print('K_a',K_a*nnum/16)
    print('gama_sub',gama_sub*nnum/16)
    print('K_s', K_s*nnum/16)


    x_delta = xnuc_all[1:nuc_step] - xnuc_all[0:nuc_step-1];
    y_delta = ynuc_all[1:nuc_step] - ynuc_all[0:nuc_step-1];
    len_delta = np.sqrt((x_delta**2 + y_delta**2));
    traj_length = np.sum(len_delta)
    avg_vel = traj_length/time[nuc_step-1]


    print('avg_vel', avg_vel)    
    

    Area_avg = np.mean(Area[int(nuc_step*0.5):nuc_step]);
    print('Area', Area_avg)

    aspect = np.zeros([memb_step])
    for ii in range(memb_step):
        xx1 = xcoord_all[0, ii]; xx2 = xcoord_all[int(nnum/2), ii]; yy1 = ycoord_all[0, ii]; yy2 = ycoord_all[int(nnum/2), ii];
        theta_1 = m.atan2(yy1, xx1);
        theta_2 = m.atan2(yy2, xx2);
        aspect[ii] = conf_R*(theta_1 - theta_2);
    aspect_avg = np.mean(aspect[int(memb_step*0.5):memb_step])/conf_width;


    write_to_file_ary = np.array([np.amax(K_a)*nnum/16, np.amax(K_l)*nnum/16, np.amax(K_s)*nnum/16, np.amax(gama_sub)*nnum/16, opt, avg_vel, Area_avg, aspect_avg, conf_width, conf_R, K_repulsive]);
    
    write_name = 'solution.txt'
    with open(write_name, "ab") as f:     np.savetxt(f, [write_to_file_ary])


    shade_width = conf_R/50;

    rs = np.array([[np.cos(phi), np.sin(phi), 0.] for phi in np.linspace(0., 2*np.pi, 100)])



  
    #plt.style.use('seaborn-pastel')
    
    filecenX=fname+'_Xcen_'+dd+'.dat'
    filecenY=fname+'_Ycen_'+dd+'.dat'
    filecenRo=fname+'_Ro_'+dd+'.dat'
    filecenfary=fname+'_fary_'+dd+'.dat'
    fileRacm=fname+'_Racmb_'+dd+'.dat'
    fileRhom=fname+'_Rhomb_'+dd+'.dat'
    fileRaccy=fname+'_Raccy_'+dd+'.dat'
    fileRhocy=fname+'_Rhocy_'+dd+'.dat'
    fileFpro=fname+'_Fpro_'+dd+'.dat'
    fileFsub=fname+'_Fsub_'+dd+'.dat'
    fileFsubdiff=fname+'_Fsub_diff_'+dd+'.dat'
    fileVr=fname+'_Vr_'+dd+'.dat'
    fileVp=fname+'_Vp_'+dd+'.dat'
    fileVt=fname+'_Vt_'+dd+'.dat'
    fileArea=fname+'_Area_'+dd+'.dat'
    fileEnerg=fname+'_Energ_'+dd+'.dat'


    np.savetxt(filetarray, time.transpose(),  fmt='%.6e', delimiter='\t');
    np.savetxt(filenucX, xnuc_all.transpose(),  fmt='%.6e', delimiter='\t');
    np.savetxt(filenucY, ynuc_all.transpose(),  fmt='%.6e', delimiter='\t');
    #np.savetxt(filecenX, xcen_all.transpose(),  fmt='%.6e', delimiter='\t');
    #np.savetxt(filecenY, xcen_all.transpose(),  fmt='%.6e', delimiter='\t');
    np.savetxt(filemembX, xcoord_all.transpose(),  fmt='%.6e', delimiter='\t');
    np.savetxt(filemembY, ycoord_all.transpose(),  fmt='%.6e', delimiter='\t');
    np.savetxt(fileFsubdiff, Fsub_diff.transpose(),  fmt='%.6e', delimiter='\t');

    return fname



nnum = int(16) ; 
opt = 1 ; 
K_s = 0.010; 
K_a = 20.00 ;  
K_l = 0.10 ;
gama_sub = 0.10 ;  
uniform = int(4) ;  
duro = 0 ;         

conf_width = 8.0

flog = migration_simulator(uniform, duro, K_s, K_a, K_l, conf_width, gama_sub, opt, nnum)

