import math as m
import numpy as np
import User_functions_R as usf
from random import random
import pdb


'''
Backward Euler to update fraction/probablity of closed bond
'''
def closed_bond_prob(alpha, zeta, ks, kc, fa, fcr, fc, f, rho_n, dt):
    if fa > fcr:
        kon = alpha*np.exp(zeta*(fa-fcr))   # + f 
    else:
        kon = alpha 
    koff = ks * np.exp(f) + kc * np.exp(-f/fc)
    rho = (rho_n + kon*dt)/(1 + kon*dt + koff*dt)
    return (rho, kon, koff)


'''
Retrograde velocity calculation
'''
def Retrograde_velocity(F_st, F_stall, v0):
    vf = v0*(1-F_st/F_stall)
    if vf < float(0.): vf = 0.
    #if vf > v0:  vf=v0
    return vf


'''
Update displacements of the substrate (xs) and actin bundle (xc)
'''
def update_disps(xs0, xc0, rho_ary, nc, Kc, eta, Ka, Kl, Ks, Fstall, F_p, v0, dt, opt):
    rho = rho_ary[0];
    kon = rho_ary[1];
    koff = rho_ary[2];
    if opt == 0:
        m11 = Kc*nc*rho;
        m12 = - (m11 + Ks);
        r1 = 0;
    else:
        m11 = -(Ka*Kc*nc*rho*dt + eta*Kc*nc*rho + eta*Kc*nc*dt*((1-rho)*kon - rho*koff)) 
        m12 = ( (Ka + Kl)*eta + Ka*Kl*dt + Ka*Kc*nc*rho*dt + eta*Kc*nc*rho + eta*Kc*nc*dt*((1-rho)*kon - rho*koff))
        r1 = (Ka + Kl)*eta*xs0 - eta*Kc*nc*rho*(xc0 - xs0)        
    m21 = 1 + v0*dt*Kc*nc*rho/Fstall
    m22 = - v0*dt*Kc*nc*rho/Fstall
    matrix = np.array([[m11, m12], [m21, m22]])
    r2 = v0*dt + xc0 + v0*dt*F_p/Fstall
    rside = np.array([r1, r2])
    xcs = np.linalg.solve(matrix, rside)
    return xcs



def max_angle_with_origin(points):
    max_angle = 0
    fidx = [13, 14, 15, 0, 1, 2, 3, 4, 5]
    rjdx = [5,  6,  7,  8, 9, 10, 11, 12, 13]
    for i in range(9):
        for j in range(9):
            idx = fidx[i];
            jdx = rjdx[j]
            p1 = np.array(points[idx,:])
            p2 = np.array(points[jdx,:])
            # Calculate the dot product and magnitudes using NumPy
            dot_product = np.dot(p1, p2)
            mag1 = np.linalg.norm(p1)
            mag2 = np.linalg.norm(p2)
            # Calculate the angle in radians using the dot product and magnitudes
            angle_radians = np.arccos(dot_product / (mag1 * mag2))
            # Convert angle to degrees and update the maximum angle if necessary
            max_angle = max(max_angle, np.degrees(angle_radians))
    return max_angle




'''
calculate nucleus velocity based on traction force input and nucleus parameters
'''
def nucleus_velocity(pst_vx, pst_vy, Ftr, Force_ck, R_nuc, eta_nuc):
    Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
    Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy); 
    coef = 6*np.pi*R_nuc*eta_nuc*1e3;
    Vnuc_x = Fx_sum/coef;  Vnuc_y = Fy_sum/coef;  
    return (Vnuc_x, Vnuc_y)

'''
calculate nucleus velocity in the curved channel --- considering the contact force with the wall
'''
def curved_nucleus_velocity_m(pst_vx, pst_vy, Ftr, Force_ck, R_nuc, Rnuc, eta_nuc, xnuc_pst, ynuc_pst, conf_R, conf_width):
    eps = 1e-6;   gap = 1.0;
    #if ynuc_pst > 0.0:   xri = 2*conf_R + int(xnuc_pst/(4*conf_R))*4*conf_R;
    #else:                xri = round(xnuc_pst/(4*conf_R))*4*conf_R;
    xri = 0.0;  

    radius = np.sqrt( (xnuc_pst-xri)**2.0 + (ynuc_pst)**2.0 );
    if radius-conf_R+conf_width/2 <= R_nuc+gap+eps or conf_R+conf_width/2-radius <= R_nuc+gap+eps:
        #pdb.set_trace()
        ewall = np.array([(xnuc_pst-xri)/radius, ynuc_pst/radius]);
        coef = 6*np.pi*Rnuc*eta_nuc*1e3;
        Coef_matrix = np.array([[-coef, 0, ewall[0]], [0, -coef, ewall[1]], [ewall[0], ewall[1], 0]]);
        if abs(np.linalg.det(Coef_matrix))>eps:
            Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
            Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy);
            Rhs = np.array([ -Fx_sum,  -Fy_sum, 0]);
            solu = np.linalg.solve(Coef_matrix, Rhs);
            Vnuc_x = solu[0];   Vnuc_y = solu[1];    Fwall = solu[2];
        else:
            print('Singular Coef_matrix in Nucleus contact detection!')
            print('xnuc_pst', xnuc_pst);
            print('ynuc_pst', ynuc_pst);
            print('xri', xri);
            print('Coef_matrix', Coef_matrix);
            Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
            Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy); 
            coef = 6*np.pi*Rnuc*eta_nuc*1e3;
            Vnuc_x = Fx_sum/coef;  Vnuc_y = Fy_sum/coef;
            Fwall = 0.;         
    else:
        Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
        Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy); 
        coef = 6*np.pi*Rnuc*eta_nuc*1e3;
        Vnuc_x = Fx_sum/coef;  Vnuc_y = Fy_sum/coef;
        Fwall = 0.;         
    return (Vnuc_x, Vnuc_y)



def curved_nucleus_velocity(pst_vx, pst_vy, Ftr, Force_ck, R_nuc, eta_nuc, xnuc_pst, ynuc_pst, conf_R, conf_width):
    eps = 1e-6;
    if ynuc_pst > 0.0:   xri = 2*conf_R + int(xnuc_pst/(4*conf_R))*4*conf_R;
    else:                xri = round(xnuc_pst/(4*conf_R))*4*conf_R;

    radius = np.sqrt( (xnuc_pst-xri)**2.0 + (ynuc_pst)**2.0 );
    if radius-conf_R+conf_width/2 <= R_nuc+eps or conf_R+conf_width/2-radius <= R_nuc+eps:
        #pdb.set_trace()
        ewall = np.array([(xnuc_pst-xri)/radius, ynuc_pst/radius]);
        coef = 6*np.pi*R_nuc*eta_nuc*1e3;
        Coef_matrix = np.array([[-coef, 0, ewall[0]], [0, -coef, ewall[1]], [ewall[0], ewall[1], 0]]);
        if abs(np.linalg.det(Coef_matrix))>eps:
            Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
            Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy);
            Rhs = np.array([ -Fx_sum,  -Fy_sum, 0]);
            solu = np.linalg.solve(Coef_matrix, Rhs);
            Vnuc_x = solu[0];   Vnuc_y = solu[1];    Fwall = solu[2];
        else:
            print('Singular Coef_matrix in Nucleus contact detection!')
            print('xnuc_pst', xnuc_pst);
            print('ynuc_pst', ynuc_pst);
            print('xri', xri);
            print('Coef_matrix', Coef_matrix);
            Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
            Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy); 
            coef = 6*np.pi*R_nuc*eta_nuc*1e3;
            Vnuc_x = Fx_sum/coef;  Vnuc_y = Fy_sum/coef;
            Fwall = 0.;         
    else:
        Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
        Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy); 
        coef = 6*np.pi*R_nuc*eta_nuc*1e3;
        Vnuc_x = Fx_sum/coef;  Vnuc_y = Fy_sum/coef;
        Fwall = 0.;         
    return (Vnuc_x, Vnuc_y)



'''
calculate microtubule compressive force
<R> = [Rx, Ry] = [nm_pts_vx, nm_pts_vy] --- Direction vector
<R+> =[Ry, -Rx] = [nm_pts_vy, -nm_pts_vx] --- Orthogonal vector
def microtubule_force(R_len, k_ck, Rcell, Rnuc, Rnuc0):  
    aa = 0.3523;   bb = 0.7068;    alph=1.45;
    nlen = np.size(R_len);  Force_ck=np.zeros(nlen);
    energ_ck = np.zeros(nlen);
    for i in range(nlen):
        stretch_change = (Rcell-Rnuc0)-(R_len[i]-Rnuc);
        #Force_ck[i]= k_ck*stretch_change
        if stretch_change > 0:
            Force_ck[i]= k_ck*stretch_change;
        else:
            stretch = (R_len[i]-Rnuc)/(Rcell-Rnuc0);
            Force_ck[i]= -(alph*k_ck*(Rcell-Rnuc0)*(stretch-1) + k_ck*(Rcell-Rnuc0)*alph*(np.exp(usf.macualay(stretch-2.7))-1))
            #-aa*k_ck*Rcell/bb*(np.exp(bb*stretch)-np.exp(bb))
    energ_ck[i]= 0.50*k_ck*stretch_change**2.0
    ck_energy = np.sum(energ_ck)
    return (Force_ck, ck_energy)
'''
def area_stiff(Area, kck0, kck1):
    aa = 0.00185;
    Kck = kck0 + kck1*np.exp(aa*Area);
    return Kck


def microtubule_force(k_ck0, k_ck1, Area1, Area2, Asoft, R_len1, R_len2):
    nlen = np.size(R_len1);  del_force_ck=np.zeros(nlen);
    energ_ck = np.zeros(nlen); 
    for i in range(nlen): 
        kck1 = area_stiff(Area1, k_ck0, k_ck1); 
        kck2 = area_stiff(Area2, k_ck0, k_ck1); 
        del_len = R_len1[i]-R_len2[i]; 
        del_force_ck[i] = del_len*(kck1+kck2)/2; 
    #energ_ck[i]= 0.50*k_ck*stretch_change**2.0
    #ck_energy = np.sum(energ_ck)  , ck_energy
    return del_force_ck



'''
calculate angles between two stress fibers
'''
def calculate_angles(nm_pts_vx, nm_pts_vy):
    nlen = np.shape(nm_pts_vx)[0];   angles_between_fibers = np.zeros([nlen]);
    for i in range(nlen):
        angle1 = m.atan2(nm_pts_vy[i], nm_pts_vx[i]);
        if i < nlen -1:
            angle2 = m.atan2(nm_pts_vy[i+1], nm_pts_vx[i+1]);
        else:
            angle2 = m.atan2(nm_pts_vy[0], nm_pts_vx[0]);
        #print('angle1', angle1);      print('angle2', angle2);   
        del_angle = angle2 - angle1;  
        if del_angle < - np.pi:   del_angle = del_angle + 2*np.pi
        if del_angle > 1.5*np.pi:   del_angle = del_angle - 2*np.pi
        angles_between_fibers[i] = del_angle;
    return angles_between_fibers
    




'''
Repulsive force implemented to prevent cross of two stress fibers 
'''
def repulsive_tangent_force_old(K_repulsive,  angles_between_fibers, nm_pts_vx, nm_pts_vy, thresh_angle):
    nlen = np.shape(nm_pts_vx)[0];   repulsive_force = np.zeros([nlen, 2]);
    for i in range(nlen):
        del_angle = angles_between_fibers[i] ; 
        if del_angle < thresh_angle:
            repulsive_force[i, 0 ] += K_repulsive*((thresh_angle-del_angle)/thresh_angle)*nm_pts_vy[i];
            repulsive_force[i, 1 ] += -K_repulsive*((thresh_angle-del_angle)/thresh_angle)*nm_pts_vx[i];
            if i < nlen -1:
                repulsive_force[i+1, 0 ] += -K_repulsive*((thresh_angle-del_angle)/thresh_angle)*nm_pts_vy[i+1];
                repulsive_force[i+1, 1 ] += K_repulsive*((thresh_angle-del_angle)/thresh_angle)*nm_pts_vx[i+1];
            else:
                repulsive_force[0, 0 ] += -K_repulsive*((thresh_angle-del_angle)/thresh_angle)*nm_pts_vy[0];
                repulsive_force[0, 1 ] += K_repulsive*((thresh_angle-del_angle)/thresh_angle)*nm_pts_vx[0];
    return repulsive_force




'''
Repulsive force implemented to prevent cross of two stress fibers 
'''
def repulsive_tangent_force(K_repulsive,  angles_between_fibers, nm_pts_vx, nm_pts_vy, thresh_angle):
    nlen = np.shape(nm_pts_vx)[0];   repulsive_force = np.zeros([nlen, 2]);
    for i in range(nlen):
        del_angle = angles_between_fibers[i] ;
        magnitude = np.exp(10*(thresh_angle-del_angle)/thresh_angle)
        if del_angle < thresh_angle:
            repulsive_force[i, 0 ] += K_repulsive*magnitude*nm_pts_vy[i];
            repulsive_force[i, 1 ] += -K_repulsive*magnitude*nm_pts_vx[i];
            if i < nlen -1:
                repulsive_force[i+1, 0 ] += -K_repulsive*magnitude*nm_pts_vy[i+1];
                repulsive_force[i+1, 1 ] += K_repulsive*magnitude*nm_pts_vx[i+1];
            else:
                repulsive_force[0, 0 ] += -K_repulsive*magnitude*nm_pts_vy[0];
                repulsive_force[0, 1 ] += K_repulsive*magnitude*nm_pts_vx[0];
    return repulsive_force



'''
solve the tagent coordinate using newton's method
'''
def find_tagent_pts_newton(xri, yri, x0, y0, x1, y1, RR):
    xy = np.array([(x0+x1)/2, (y0+y1)/2]);
    eps_val = 1;
    iteration = 0;
    while iteration < 20 and eps_val>1e-6:
        tag_matrix = np.array([[2*(xy[0]-xri), 2*(xy[1]-yri)], [2*xy[0]-xri-x1, 2*xy[1]-yri-y1]]);
        res_vec = np.array([RR**2 - (xy[0]-xri)**2 -(xy[1]-yri)**2, -(xy[1]-yri)*(xy[1]-y1)-(xy[0]-xri)*(xy[0]-x1)])
        del_xy = np.linalg.solve(tag_matrix, res_vec);
        xy += del_xy
        iteration += 1;
        eps_val = np.linalg.norm(del_xy)
    return xy



'''
solve the tagent coordinate using bisection
'''
def find_tagent_pts_bisection(xri, yri, x0, y0, x1, y1, RR, conf_R):
    search_times = 200;
    t1 = 1/search_times;
    #xv = x1*t1 + x0*(1-t1);
    xv = x0*t1 + x1*(1-t1);
    #yv0 = y0*t1 + y1*(1-t1);
    if xri%(4*conf_R)<1e-3: yv_sign = -1;
    else: yv_sign = 1; #sign of the tagent point 
    
    
    if RR**2-(xv-xri)**2<0:
       #xv = x0;
       yv = y0*t1 + y1*(1-t1);
    else:
       y_try = np.sqrt(RR**2-(xv-xri)**2)
       if y_try<=max(y0, y1) and y_try>=min(y0, y1) and yv_sign*y_try>=0:   yv = y_try;
       else: yv = -y_try;
    vec1 = np.array([xv-xri, yv-yri]);
    vec2 = np.array([x1-xv, y1-yv]);
    vec_prod1 = np.dot(vec1, vec2);
    tt = 0;
    for ii in range(search_times-1):
        t2 = t1 + 1/search_times;
        xv = x0*t2 + x1*(1-t2);  #xv = x1*t2 + x0*(1-t2);
        if RR**2-(xv-xri)**2<0:
           yv = y0*t2 + y1*(1-t2);
        else:
           y_try = np.sqrt(RR**2-(xv-xri)**2)
           if y_try<=max(y0, y1) and y_try>=min(y0, y1) and yv_sign*y_try>=0:   yv = y_try;
           else: yv = -y_try;
        vec1 = np.array([xv-xri, yv-yri]);
        vec2 = np.array([x1-xv, y1-yv]);
        vec_prod2 = np.dot(vec1, vec2);
        if vec_prod1*vec_prod2<=0: tt = t2;   break;
        vec_prod1 = vec_prod2;
        t1 = t2;
    #if tt==0 or tt==1:
        #print('cannot find correct tt value!');
       
    #xv = x0*tt + x1*(1-tt);  #xv = x1*tt + x0*(1-tt);
    #if RR**2-(xv-xri)**2<0:
    #   xv = x0;
    #   yv = y0;
    #else:
    #   y_try = np.sqrt(RR**2-(xv-xri)**2)
    #   if y_try<=max(y0, y1) and y_try>=min(y0, y1) and yv_sign*y_try>0:   yv = y_try;
    #   else: yv = -y_try;
    xy = np.array([xv, yv]);    
    return (tt, xy)



'''
solve the tagent coordinate using bisection using polar coordinate expression   ----  for circular channel
'''
def polar_tagent_pts_bisection(xri, yri, x0, y0, x1, y1, xnuc, ynuc, RR, conf_R):
    search_times = 200;
    angle1 = m.atan2(y0-yri, x0-xri);
    angle2 = m.atan2(y1-yri, x1-xri);
    ang_nuc = m.atan2(ynuc-yri, xnuc-xri);
    if angle1*angle2<0 and abs(ang_nuc)>np.pi/2:
        if angle1<0: angle1 += 2*np.pi;
        if angle2<0: angle2 += 2*np.pi;

    ang_ary = np.linspace(angle1, angle2, search_times);

    angle = angle1;

    xv = RR*np.cos(ang_ary[0]);
    yv = RR*np.sin(ang_ary[0]);

    vec1 = np.array([xv-xri, yv-yri]);
    vec2 = np.array([x1-xv, y1-yv]);
    vec_prod1 = np.dot(vec1, vec2);
    tt = 0;
    for ii in range(search_times-1):
        xv = RR*np.cos(ang_ary[ii+1]);
        yv = RR*np.sin(ang_ary[ii+1]);
    
        vec1 = np.array([xv-xri, yv-yri]);
        vec2 = np.array([x1-xv, y1-yv]);
        
        vec_prod2 = np.dot(vec1, vec2);
        if vec_prod1*vec_prod2<=0: angle = 0.5*(ang_ary[ii]+ang_ary[ii+1]);   tt += 1/search_times;    break;
    
        vec_prod1 = vec_prod2;

    xy = np.array([xv, yv]);
    
    return (tt, angle, xy)



'''
plot part of circle based on two points and a center (raidus)
'''
def plot_part_of_circle(xri, yri, x0, y0, x1, y1, RR, conf_R):
    pts_num = 50;
    if xri%(4*conf_R)<1e-3: yv_sign = -1;
    else: yv_sign = 1; #sign of the tagent point 
    xx_ary = np.zeros([pts_num+1]);   yy_ary = np.zeros([pts_num+1]); 
    for ii in range(pts_num+1):
        tt = ii/pts_num;
        xv = x1*tt + x0*(1-tt);
        yv0 = y1*tt + y0*(1-tt);
        if RR**2-(xv-xri)**2>0:
           y_try = np.sqrt(RR**2-(xv-xri)**2)
           if  yv_sign*y_try>0:   yv = y_try;
           else: yv = -y_try;
        else:
           yv = yv0;

        xx_ary[ii] = xv;
        yy_ary[ii] = yv;
    return (xx_ary, yy_ary)



'''
plot part of circle based on two points and a center (raidus)  --- based on the polar coordinate for the circular channel
'''
def polar_plot_part_of_circle(xri, yri, x0, y0, x1, y1, xnuc, ynuc, RR, conf_R):
    
    pts_num = 50;
    angle1 = m.atan2(y0, x0);
    angle2 = m.atan2(y1, x1);
    ang_nuc = m.atan2(ynuc, xnuc);
    if angle1*angle2<0 and abs(ang_nuc)>np.pi/2:
        if angle1<0: angle1 += 2*np.pi;
        if angle2<0: angle2 += 2*np.pi;
    ang_ary = np.linspace(angle1, angle2, pts_num);

    xx_ary = xri + RR*np.cos(ang_ary);
    
    yy_ary = yri + RR*np.sin(ang_ary);

    return (xx_ary, yy_ary)

        


'''
describe a circle bewteen two vertices
input: vectors from two vertices to circle center (origin) and a vector between two vertices
'''
def characterize_circle_inbetween(vertex_to_ori,  vertex_to_ori1, vertices_vec, RR):
     [angle, unit_vertex_ori, unit_vertex_ori1] = usf.vectors_form_angle(vertex_to_ori, vertex_to_ori1);
     
     arc_len = angle*RR; # length of the edge
     tangent_vec1 = np.array([unit_vertex_ori[1], -unit_vertex_ori[0]])
     tangent_vec2 = np.array([unit_vertex_ori1[1], -unit_vertex_ori1[0]])
     if np.dot(tangent_vec1, vertices_vec)<0: tangent_vec1 = -tangent_vec1;
     if np.dot(tangent_vec2, -vertices_vec)<0: tangent_vec2 = -tangent_vec2;
     return (arc_len, tangent_vec1, tangent_vec2)



'''
check the penetraion and calculate the tagent curves based on two vertices
'''
def check_line_circle(vector_edge, xcoord, ycoord, xnuc, ynuc, conf_R, conf_width):
    tol = 1e-6;
    [nlen, nwid] = np.shape(vector_edge);
    vector_edge_new = vector_edge;   
    RR = conf_R - conf_width/2
    #line_type_ary = np.zeros([nlen]);  #0 for a line, 1 for circle, 2 for a line + a circle, 3 for two lines+circle
    for i in range(nlen):
        #print('xcoord0',xcoord0);
        xx1= xcoord[int((i+1)%nlen)];    yy1= ycoord[int((i+1)%nlen)];
        xx = xcoord[i];      yy = ycoord[i];
        #xx2= xcoord[i-1];    yy2= ycoord[i-1];
        edge_len = vector_edge[i, 0]
        
        #if yy > 0.0:   xri = 2*conf_R + int(xx/(4*conf_R))*4*conf_R;
        #else: xri = round(xx/(4*conf_R))*4*conf_R;
        xri = 0.0;   xri1 = 0.0;
        yri = 0.0;   yri1 = 0.0;

        #if yy1 > 0.0:   xri1 = 2*conf_R + int(xx1/(4*conf_R))*4*conf_R;
        #else: xri1 = round(xx1/(4*conf_R))*4*conf_R;        
        
        #radius2 = np.sqrt( (xx1-xri1)**2.0 + (yy1-yri1)**2.0 )      
        radius = np.sqrt( (xx-xri)**2.0 + (yy-yri)**2.0 )

        #if abs(xri1-xri)>conf_R and radius2<radius and radius2<conf_R:
        #   xri = xri1;
        #   radius = np.sqrt( (xx-xri)**2.0 + (yy-yri)**2.0 )


        radius1 = np.sqrt( (xx1-xri)**2.0 + (yy1-yri)**2.0 )

        vertex_to_ori = np.array([xx-xri, yy-yri]);  #define a vector from circle center(origin) to vertex (xx)
        vertex_to_ori1 = np.array([xx1-xri, yy1-yri]);  #define a vector from circle center(origin) to vertex (xx1)
        dot_prod1 = np.dot(vector_edge[i,1:3], vertex_to_ori);
        dot_prod2 = np.dot(vector_edge[i,1:3], vertex_to_ori1);

        len_tagent1 = abs(dot_prod1)  # project of vertex_to_ori on cell edge
        
        #len_tagent2 = abs(np.dot(vector_edge[i,4:6], vertex_to_ori))  # project of vertex_to_ori on cell edge
        
        if len_tagent1< edge_len and dot_prod1*dot_prod2<0: #conf_R - conf_width + tol:
           shortest_dist1 = np.sqrt(radius**2.0 - len_tagent1**2.0) ;

           if shortest_dist1 < RR - 1e-3:  #shortest_dist1 < np.sqrt(RR**2.0 - (edge_len/10)**2.0):
           
             if  radius<RR + 1e-3 and radius1<RR + 1e-3: # two vertices get contacts to channel wall
                
                [edge_len, edge_tagent1, edge_tagent2] = characterize_circle_inbetween(vertex_to_ori,  vertex_to_ori1, vector_edge[i,1:3], RR);
                
                if np.isnan(edge_len): print('label',1);   #pdb.set_trace();
                tt = 0.5;
                #line_type_ary[i] = 1;     
                
             elif radius<RR + 1e-3: # one vertex gets contacts to channel wall        
                [tt, angle, xy] = polar_tagent_pts_bisection(xri, yri, xx, yy, xx1, yy1, xnuc, ynuc, RR, conf_R);
                if tt>0 and tt<1:
                    xy_to_ori1 = np.array([xy[0]-xri, xy[1]-yri]);  #
                    dist_xy_xx1 = np.sqrt((xy[0]-xx1)**2.0+(xy[1]-yy1)**2.0)
                    [arc_len, edge_tagent1, edge_tagent2] = characterize_circle_inbetween(vertex_to_ori,  xy_to_ori1, vector_edge[i,1:3], RR);    
                    #line_type_ary[i] = 2;
                    edge_len = arc_len + dist_xy_xx1;
                    if np.isnan(edge_len): print('label',2);  #pdb.set_trace();


             elif radius1<RR + 1e-3: # one vertex gets contacts to channel wall  
                
                [tt, angle, xy] = polar_tagent_pts_bisection(xri, yri, xx1, yy1, xx, yy, xnuc, ynuc, RR, conf_R);
                if tt>0 and tt<1:
                    xy_to_ori1 = np.array([xy[0]-xri, xy[1]-yri]);  #d
                    dist_xy_xx = np.sqrt((xy[0]-xx)**2.0+(xy[1]-yy)**2.0)
                    [arc_len, edge_tagent1, edge_tagent2] = characterize_circle_inbetween(xy_to_ori1,  vertex_to_ori1, vector_edge[i,1:3], RR);    
                    #line_type_ary[i] = 2;
                    edge_len = arc_len + dist_xy_xx;
                    if np.isnan(edge_len): print('label',3);  # pdb.set_trace();
               
             else:
                
                [t1, angle1, xy1]  = polar_tagent_pts_bisection(xri, yri, xx, yy, xx1, yy1, xnuc, ynuc, RR, conf_R);
                xy_to_ori1 = np.array([xy1[0]-xri, xy1[1]-yri]);  #
                dist_xy_xx = np.sqrt((xy1[0]-xx)**2.0+(xy1[1]-yy)**2.0)

                [t2, angle2, xy2] = polar_tagent_pts_bisection(xri, yri, xx1, yy1, xx, yy, xnuc, ynuc, RR, conf_R);   
                xy_to_ori2 = np.array([xy2[0]-xri, xy2[1]-yri]);  #            
                dist_xy_xx1 = np.sqrt((xy2[0]-xx1)**2.0+(xy2[1]-yy1)**2.0)                
                #line_type_ary[i] = 3;
                tt = min(t1,t2); 
                if tt>0  and max(t1,t2)<1:
                    [arc_len, edge_tagent1, edge_tagent2] = characterize_circle_inbetween(xy_to_ori1,  xy_to_ori2, vector_edge[i,1:3], RR);    
                    edge_len = arc_len + dist_xy_xx + dist_xy_xx1;
                    if np.isnan(edge_len): print('label',4);   #pdb.set_trace(); 

                
             if tt>0 and tt<1:
                vector_edge_new[i, 0] = edge_len;   vector_edge_new[i, 1:3] = edge_tagent1;
                vector_edge_new[int((i+1)%nlen), 3] = edge_len;   vector_edge_new[int((i+1)%nlen), 4:6] = edge_tagent2;

    return vector_edge_new



'''
plot curved edges 
'''
def plot_curved_edges(xcoord, ycoord, xnuc, ynuc, conf_R, conf_width):
    tol = 1e-6;
    nlen = xcoord.size;
    vector_edge = usf.edge_vectors(xcoord, ycoord);

    RR = conf_R - conf_width/2
 
    line_type_ary = np.zeros([nlen]);  #0 for a line, 1 for circle, 2 for a line + a circle, 3 for two lines+circle
    for i in range(nlen):
        xx1= xcoord[int((i+1)%nlen)];    yy1= ycoord[int((i+1)%nlen)]; 
        xx = xcoord[i];      yy = ycoord[i]; 
        edge_len = vector_edge[i, 0] ; 

        xri = 0.0;    xri1 = 0.0; 
        yri = 0.0;    yri1 = 0.0; 
  
        radius = np.sqrt( (xx-xri)**2.0 + (yy-yri)**2.0 )

        radius1 = np.sqrt( (xx1-xri)**2.0 + (yy1-yri)**2.0 )

        vertex_to_ori = np.array([xx-xri, yy-yri]);  #define a vector from circle center(origin) to vertex (xx)
        vertex_to_ori1 = np.array([xx1-xri, yy1-yri]);  #define a vector from circle center(origin) to vertex (xx1)
        dot_prod1 = np.dot(vector_edge[i,1:3], vertex_to_ori);
        dot_prod2 = np.dot(vector_edge[i,1:3], vertex_to_ori1);

        len_tagent1 = abs(dot_prod1)  # project of vertex_to_ori on cell edge
        
        #len_tagent2 = abs(np.dot(vector_edge[i,4:6], vertex_to_ori))  # project of vertex_to_ori on cell edge
        
        if len_tagent1< edge_len and dot_prod1*dot_prod2<0: #conf_R - conf_width + tol:

           shortest_dist1 = np.sqrt(radius**2.0 - len_tagent1**2.0) ;

           if shortest_dist1 < RR - 0.1:  #np.sqrt(RR**2.0 - (edge_len/10)**2.0):
           
             if  radius<RR+0.05 and radius1<RR+0.05: # two vertices get contacts to channel wall
                 [xx_ary, yy_ary] = polar_plot_part_of_circle(xri, yri, xx, yy, xx1, yy1, xnuc, ynuc, RR, conf_R) 
                 plt.plot(xx_ary,  yy_ary, 'b-', lw=3); 
                 line_type_ary[i] = 1;     
                
             elif radius<RR+0.05: # one vertice gets contacts to channel wall  or radius1<conf_R - conf_width + tol               
                 #[tt, xy] = find_tagent_pts_bisection(xri, yri, xx, yy, xx1, yy1, RR, conf_R);
                 [tt, angle, xy] = polar_tagent_pts_bisection(xri, yri, xx, yy, xx1, yy1, xnuc, ynuc, RR, conf_R);
                 plt.plot(np.array([xx1, xy[0]]), np.array([yy1, xy[1]]),  'b-', lw=3); 
                 [xx_ary, yy_ary] = polar_plot_part_of_circle(xri, yri, xx, yy, xy[0], xy[1], xnuc, ynuc, RR, conf_R)   
                 plt.plot(xx_ary,  yy_ary, 'b-', lw=3);
                 line_type_ary[i] = 2;     

             elif radius1<RR+0.05: # one vertice gets contacts to channel wall  or radius1<conf_R - conf_width + tol
                 #[tt, xy] = find_tagent_pts_bisection(xri, yri, xx1, yy1, xx, yy, RR, conf_R);
                 [tt, angle, xy] = polar_tagent_pts_bisection(xri, yri, xx1, yy1, xx, yy, xnuc, ynuc, RR, conf_R);
                 plt.plot(np.array([xx, xy[0]]), np.array([yy, xy[1]]),  'b-', lw=3); 
                 [xx_ary, yy_ary] = polar_plot_part_of_circle(xri, yri, xx1, yy1, xy[0], xy[1], xnuc, ynuc, RR, conf_R)     
                 plt.plot(xx_ary,  yy_ary, 'b-', lw=3);
                 line_type_ary[i] = 2;     
               
             else:
                 
                 #[tt, xy1] = find_tagent_pts_bisection(xri, yri, xx, yy, xx1, yy1, RR, conf_R);
                 [tt, angle, xy1] = polar_tagent_pts_bisection(xri, yri, xx, yy, xx1, yy1, xnuc, ynuc, RR, conf_R);

                 plt.plot(np.array([xx1, xy1[0]]), np.array([yy1, xy1[1]]),  'b-', lw=3);

                 #[tt, xy2] = find_tagent_pts_bisection(xri, yri, xx1, yy1, xx, yy, RR, conf_R);
                 [tt, angle, xy2] = polar_tagent_pts_bisection(xri, yri, xx1, yy1, xx, yy, xnuc, ynuc, RR, conf_R);

                 plt.plot(np.array([xx, xy2[0]]), np.array([yy, xy2[1]]),  'b-', lw=3);
                            
                 [xx_ary, yy_ary] = polar_plot_part_of_circle(xri, yri, xy1[0], xy1[1], xy2[0], xy2[1], xnuc, ynuc, RR, conf_R)        
                     
                 line_type_ary[i] = 3;
                 
           else:
             plt.plot(np.array([xx, xx1]), np.array([yy, yy1]),  'b-', lw=3);
             
        else:
            plt.plot(np.array([xx, xx1]), np.array([yy, yy1]),  'b-', lw=3);
            
    return line_type_ary

    
'''
calculate membrane force and then get protrusion force based on equilibrium
<R> = [Rx, Ry] = [nm_pts_vx, nm_pts_vy] --- Direction vector
<R+> =[Ry, -Rx] = [nm_pts_vy, -nm_pts_vx] --- Orthogonal vector
modify log: add Vtn for friction force direction calculation;
            add fr_coef for friction coefficient --- 04/13/2023 
'''
def curved_protrusion_fv(vector_edge, k_mem, edge_L0, Force_ck, eta_mem, v0, Vsn, Vtn, Vp, fr_coef, Fsub, Fstall, xcoord0, ycoord0, xnuc, ynuc, conf_R, conf_width, nm_pts_vx, nm_pts_vy, repulsive_force):
    eps= 1e-6;
    nlen = np.shape(vector_edge)[0];  protrusion_force=np.zeros([nlen,2]); lavg = np.zeros([nlen]);   trigger =np.zeros([nlen]); 
    energ_mb = np.zeros([nlen]);   F_pro = np.zeros([nlen]);   vt = np.zeros([nlen]);   F_wall = np.zeros([nlen]);

    vector_edge_new = check_line_circle(vector_edge, xcoord0, ycoord0, xnuc, ynuc, conf_R, conf_width);
    
    for i in range(nlen):
        ext_disp1 = (vector_edge_new[i,0]-edge_L0); #/edge_L0
        ext_disp2 = (vector_edge_new[i,3]-edge_L0);
        #if ext_disp1<0: ext_disp1=0.0; 
        #if ext_disp2<0: ext_disp2=0.0; 
        protrusion_force[i,0]= k_mem*(ext_disp1*vector_edge_new[i,1] + ext_disp2*vector_edge_new[i,4]); 
        protrusion_force[i,1]= k_mem*(ext_disp1*vector_edge_new[i,2] + ext_disp2*vector_edge_new[i,5]);
        lavg[i] = (vector_edge_new[i,0] + vector_edge_new[i,3])/2;
        energ_mb[i] = 0.50*k_mem[i]*ext_disp2**2.0
    # membrane force in polar and tagent directions 
    Fmemb_nm = protrusion_force[:,0]*nm_pts_vx+protrusion_force[:,1]*nm_pts_vy 
    Fmemb_tg = protrusion_force[:,0]*nm_pts_vy-protrusion_force[:,1]*nm_pts_vx
    repulsive_tg = repulsive_force[:,0]*nm_pts_vy - repulsive_force[:,1]*nm_pts_vx
    # protrusion forces in nucleus-membrane direction
    #F_pro = eta_mem*vn*lavg-Fmemb_nm-Force_ck;
    
    for i in range(nlen):
        #print('xcoord0',xcoord0);
        xx= xcoord0[i];    yy= ycoord0[i];
        xri = 0.0;
        Vix = Vsn[i]*nm_pts_vx[i]  +  Vtn[i]*nm_pts_vy[i];   #Calculate the velocity direction
        Viy = Vsn[i]*nm_pts_vy[i]  -  Vtn[i]*nm_pts_vx[i];   #Calculate the velocity direction
        radius = np.sqrt( (xx-xri)**2.0 + (yy)**2.0 )
        if abs(radius-conf_R) > conf_width/2-eps:
            trigger[i] = conf_width/2;   
            e_wall = np.array([(xx-xri)/radius, yy/radius]);
            if radius > conf_R:  e_wall = -e_wall;     
            ewall = e_wall/np.linalg.norm(e_wall);    
            twall = np.array([-ewall[1], ewall[0]]);   # assume the friction force direction
            dot_vt = Vix*twall[0] + Viy*twall[1];     # the firction force direction must be opposite to the velocity direction
            #pdb.set_trace()
            
            if dot_vt > 0:  twall = np.array([ewall[1], -ewall[0]]);
                
            Coef_matrix = np.array([[1+eta_mem*lavg[i]*v0/Fstall[i], ewall[0]*nm_pts_vx[i] + ewall[1]*nm_pts_vy[i] + fr_coef*(twall[0]*nm_pts_vx[i]+twall[1]*nm_pts_vy[i]), 0], \
                                    [0, ewall[0]*nm_pts_vy[i]-ewall[1]*nm_pts_vx[i] + fr_coef*(twall[0]*nm_pts_vy[i]-twall[1]*nm_pts_vx[i]),  -eta_mem*lavg[i]],   \
                                    [-v0/Fstall[i]*(ewall[0]*nm_pts_vx[i]+ewall[1]*nm_pts_vy[i]), 0, ewall[0]*nm_pts_vy[i]-ewall[1]*nm_pts_vx[i]]]) ; 
            Rhs = np.array([eta_mem*lavg[i]*(Vp[i]-v0*(1-Fsub[i]/Fstall[i]))-Fmemb_nm[i]-Force_ck[i],  -Fmemb_tg[i]-repulsive_tg[i], (v0*(1-Fsub[i]/Fstall[i])-Vp[i])*(ewall[0]*nm_pts_vx[i]+ewall[1]*nm_pts_vy[i])]);
            solu = np.linalg.solve(Coef_matrix, Rhs);
            F_pro[i] = solu[0];
            F_wall[i] = solu[1]  #max(solu[1], 0);
            vt[i] = solu[2];
        else:
            trigger[i] = 0;
            F_pro[i] = (eta_mem*lavg[i]*(Vp[i]-v0*(1-Fsub[i]/Fstall[i]))-Fmemb_nm[i]-Force_ck[i])/(1+eta_mem*lavg[i]*v0/Fstall[i]) ;
            vt[i] = (Fmemb_tg[i]+repulsive_tg[i])/(eta_mem*lavg[i]);      #vt[i] = Fmemb_tg[i]/(eta_mem*lavg[i])
    memb_energy = np.sum(energ_mb)
    #Vs_updated = Vp - v0*(1-(Fsub-F_pro)/Fstall)
    #pdb.set_trace()
    return (F_pro, vt, F_wall, Fmemb_nm, Fmemb_tg, memb_energy, trigger)





'''
calculate membrane force and then get protrusion force based on equilibrium
<R> = [Rx, Ry] = [nm_pts_vx, nm_pts_vy] --- Direction vector
<R+> =[Ry, -Rx] = [nm_pts_vy, -nm_pts_vx] --- Orthogonal vector
modify log: add Vtn for friction force direction calculation;
            add fr_coef for friction coefficient --- 04/13/2023 
'''
def modified_curved_protrusion(vector_edge, k_mem, edge_L0, Force_ck, eta_mem, v0, Vsn, Vtn, Vp, fr_coef, Fw, Fsub, Fstall, xcoord0, ycoord0, xnuc, ynuc, conf_R, conf_width, nm_pts_vx, nm_pts_vy, repulsive_force):
    eps= 1e-6;
    nlen = np.shape(vector_edge)[0];  protrusion_force=np.zeros([nlen,2]); lavg = np.zeros([nlen]);   trigger =np.zeros([nlen]); 
    energ_mb = np.zeros([nlen]);   F_pro = np.zeros([nlen]);   vt = np.zeros([nlen]);   F_wall = np.zeros([nlen]);

    vector_edge_new = check_line_circle(vector_edge, xcoord0, ycoord0, xnuc, ynuc, conf_R, conf_width);
    
    for i in range(nlen):
        ext_disp1 = (vector_edge_new[i,0]-edge_L0); #/edge_L0
        ext_disp2 = (vector_edge_new[i,3]-edge_L0);
        #if ext_disp1<0: ext_disp1=0.0; 
        #if ext_disp2<0: ext_disp2=0.0; 
        protrusion_force[i,0]= k_mem*(ext_disp1*vector_edge_new[i,1] + ext_disp2*vector_edge_new[i,4]); 
        protrusion_force[i,1]= k_mem*(ext_disp1*vector_edge_new[i,2] + ext_disp2*vector_edge_new[i,5]);
        lavg[i] = (vector_edge_new[i,0] + vector_edge_new[i,3])/2;
        energ_mb[i] = 0.50*k_mem*ext_disp2**2.0
    # membrane force in polar and tagent directions 
    Fmemb_nm = protrusion_force[:,0]*nm_pts_vx+protrusion_force[:,1]*nm_pts_vy 
    Fmemb_tg = protrusion_force[:,0]*nm_pts_vy-protrusion_force[:,1]*nm_pts_vx
    repulsive_tg = repulsive_force[:,0]*nm_pts_vy - repulsive_force[:,1]*nm_pts_vx
    # protrusion forces in nucleus-membrane direction
    #F_pro = eta_mem*vn*lavg-Fmemb_nm-Force_ck;
    
    for i in range(nlen):
        #print('xcoord0',xcoord0);
        xx= xcoord0[i];    yy= ycoord0[i];
        xri = 0.0;
        Vix = Vsn[i]*nm_pts_vx[i]  +  Vtn[i]*nm_pts_vy[i];   #Calculate the velocity direction
        Viy = Vsn[i]*nm_pts_vy[i]  -  Vtn[i]*nm_pts_vx[i];   #Calculate the velocity direction
        radius = np.sqrt( (xx-xri)**2.0 + (yy)**2.0 )
        if abs(radius-conf_R) > conf_width/2-eps:
            trigger[i] = conf_width/2;
            e_wall = np.array([(xx-xri)/radius, yy/radius]);
            if radius > conf_R:  e_wall = -e_wall;     
            ewall = e_wall/np.linalg.norm(e_wall);  
            twall = np.array([-ewall[1], ewall[0]]);   # assume the friction force direction
            #dot_vt = Vix*twall[0] + Viy*twall[1];     # the firction force direction must be opposite to the velocity direction
            ##pdb.set_trace()
            #if dot_vt > 0:  twall = np.array([ewall[1], -ewall[0]]);

            #Frn = fr_coef*abs(Fw[i])*(twall[0]*nm_pts_vx[i]+twall[1]*nm_pts_vy[i])
            #Frt = fr_coef*abs(Fw[i])*(twall[0]*nm_pts_vy[i]-twall[1]*nm_pts_vx[i])
            Frn = abs(fr_coef*Fw[i]*(twall[0]*nm_pts_vx[i]+twall[1]*nm_pts_vy[i]))*(-np.sign(Vsn[i]))
            Frt = abs(fr_coef*Fw[i]*(twall[0]*nm_pts_vy[i]-twall[1]*nm_pts_vx[i]))*(-np.sign(Vtn[i]))
            
            Coef_matrix = np.array([[1+eta_mem*lavg[i]*v0/Fstall[i], ewall[0]*nm_pts_vx[i] + ewall[1]*nm_pts_vy[i], 0], \
                                    [0, ewall[0]*nm_pts_vy[i]-ewall[1]*nm_pts_vx[i],  -eta_mem*lavg[i]],   \
                                    [-v0/Fstall[i]*(ewall[0]*nm_pts_vx[i]+ewall[1]*nm_pts_vy[i]), 0, ewall[0]*nm_pts_vy[i]-ewall[1]*nm_pts_vx[i]]]) ; 
            Rhs = np.array([eta_mem*lavg[i]*(Vp[i]-v0*(1-Fsub[i]/Fstall[i]))-Fmemb_nm[i]-Force_ck[i]-Frn,  -Fmemb_tg[i]-repulsive_tg[i]-Frt, (v0*(1-Fsub[i]/Fstall[i])-Vp[i])*(ewall[0]*nm_pts_vx[i]+ewall[1]*nm_pts_vy[i])]);
            solu = np.linalg.solve(Coef_matrix, Rhs);
            F_pro[i] = solu[0];
            F_wall[i] = solu[1]  #max(solu[1], 0);
            vt[i] = solu[2];
        else:
            trigger[i] = 0;
            F_pro[i] = (eta_mem*lavg[i]*(Vp[i]-v0*(1-Fsub[i]/Fstall[i]))-Fmemb_nm[i]-Force_ck[i])/(1+eta_mem*lavg[i]*v0/Fstall[i]) ;
            vt[i] = (Fmemb_tg[i]+repulsive_tg[i])/(eta_mem*lavg[i]);      #vt[i] = Fmemb_tg[i]/(eta_mem*lavg[i])
    memb_energy = np.sum(energ_mb)
    #Vs_updated = Vp - v0*(1-(Fsub-F_pro)/Fstall)
    #pdb.set_trace()
    return (F_pro, vt, F_wall, Fmemb_nm, Fmemb_tg, memb_energy, trigger)




'''
calculate membrane force and then get protrusion force based on equilibrium
<R> = [Rx, Ry] = [nm_pts_vx, nm_pts_vy] --- Direction vector
<R+> =[Ry, -Rx] = [nm_pts_vy, -nm_pts_vx] --- Orthogonal vector
modify log: add Vtn for friction force direction calculation;
            add fr_coef for friction coefficient --- 04/13/2023 
'''
def modified_friction_viscosity(vector_edge, k_mem0, edge_L0, Force_ck, eta_mem0, v0, Vsn, Vtn, Vp, mu_coef, fr_coef, Fsub, Fstall, xcoord0, ycoord0, xnuc, ynuc, conf_R, conf_width, nm_pts_vx, nm_pts_vy, repulsive_force):
    eps= 1e-6
    nlen = np.shape(vector_edge)[0];  protrusion_force=np.zeros([nlen,2]); lavg = np.zeros([nlen]);   trigger =np.zeros([nlen]); 
    energ_mb = np.zeros([nlen]);   F_pro = np.zeros([nlen]);   vt = np.zeros([nlen]);   F_wall = np.zeros([nlen]);

    vector_edge_new = check_line_circle(vector_edge, xcoord0, ycoord0, xnuc, ynuc, conf_R, conf_width);

    # calculate cell length in the channel
    points = np.zeros([nlen, 2]);
    points[:,0] = xcoord0[0:nlen];  points[:,1] = ycoord0[0:nlen];  
    max_angle = max_angle_with_origin(points)
    arc_len = m.radians(max_angle)*conf_R;

    # update the  stiffness of the  membrane
    k_mem = np.zeros([nlen]);
    for i in range(nlen):
        xx= xcoord0[i];    yy= ycoord0[i];   xri = 0.0 ; 
        radius = np.sqrt( (xx-xri)**2.0 + (yy)**2.0 ) ; 
        if abs(radius-conf_R) > conf_width/2 - 0.1:
           #k_mem[i] = (1 + 2*mu_coef*np.sin(arc_len/2/conf_R))*k_mem0
           k_mem[i] = (1 +  mu_coef*(100/conf_R)**2.0 + 0.5)*k_mem0
        else:
           k_mem[i] = k_mem0

        
    for i in range(nlen):
        ext_disp1 = (vector_edge_new[i,0]-edge_L0); #/edge_L0
        ext_disp2 = (vector_edge_new[i,3]-edge_L0);
        #if ext_disp1<0: ext_disp1=0.0; 
        #if ext_disp2<0: ext_disp2=0.0; 
        protrusion_force[i,0]= k_mem[i]*(ext_disp1*vector_edge_new[i,1] + ext_disp2*vector_edge_new[i,4]); 
        protrusion_force[i,1]= k_mem[i]*(ext_disp1*vector_edge_new[i,2] + ext_disp2*vector_edge_new[i,5]);
        lavg[i] = (vector_edge_new[i,0] + vector_edge_new[i,3])/2;
        energ_mb[i] = 0.50*k_mem[i]*ext_disp2**2.0
    # membrane force in polar and tagent directions 
    Fmemb_nm = protrusion_force[:,0]*nm_pts_vx+protrusion_force[:,1]*nm_pts_vy 
    Fmemb_tg = protrusion_force[:,0]*nm_pts_vy-protrusion_force[:,1]*nm_pts_vx
    repulsive_tg = repulsive_force[:,0]*nm_pts_vy - repulsive_force[:,1]*nm_pts_vx
    # protrusion forces in nucleus-membrane direction 
    # F_pro = eta_mem*vn*lavg-Fmemb_nm-Force_ck; 
    
    for i in range(nlen):
        #print('xcoord0',xcoord0);
        xx= xcoord0[i];    yy= ycoord0[i];
        xri = 0.0;
        Vix = Vsn[i]*nm_pts_vx[i]  +  Vtn[i]*nm_pts_vy[i] ; #Calculate the velocity direction
        Viy = Vsn[i]*nm_pts_vy[i]  -  Vtn[i]*nm_pts_vx[i] ; #Calculate the velocity direction
        radius = np.sqrt( (xx-xri)**2.0 + (yy)**2.0 ) ; 
        if abs(radius-conf_R) > conf_width/2-eps: 
            trigger[i] = conf_width/2; 
            e_wall = np.array([(xx-xri)/radius, yy/radius]);
            if radius > conf_R:  e_wall = -e_wall;     
            ewall = e_wall/np.linalg.norm(e_wall);  

            eta_friction =   fr_coef*(100/conf_R)**2.0; #fr_coef/conf_R*100;
            eta_mem = eta_mem0*(1 + eta_friction + 0.5);
            
            Coef_matrix = np.array([[1+eta_mem*lavg[i]*v0/Fstall[i], ewall[0]*nm_pts_vx[i] + ewall[1]*nm_pts_vy[i], 0], \
                                    [0, ewall[0]*nm_pts_vy[i]-ewall[1]*nm_pts_vx[i],  -eta_mem*lavg[i]],   \
                                    [-v0/Fstall[i]*(ewall[0]*nm_pts_vx[i]+ewall[1]*nm_pts_vy[i]), 0, ewall[0]*nm_pts_vy[i]-ewall[1]*nm_pts_vx[i]]]) ; 
            Rhs = np.array([eta_mem*lavg[i]*(Vp[i]-v0*(1-Fsub[i]/Fstall[i]))-Fmemb_nm[i]-Force_ck[i],  -Fmemb_tg[i]-repulsive_tg[i], (v0*(1-Fsub[i]/Fstall[i])-Vp[i])*(ewall[0]*nm_pts_vx[i]+ewall[1]*nm_pts_vy[i])]);
            solu = np.linalg.solve(Coef_matrix, Rhs);
            F_pro[i] = solu[0];
            F_wall[i] = solu[1]  #max(solu[1], 0);
            vt[i] = solu[2];
        else:
            
            trigger[i] = 0;
            F_pro[i] = (eta_mem0*lavg[i]*(Vp[i]-v0*(1-Fsub[i]/Fstall[i]))-Fmemb_nm[i]-Force_ck[i])/(1+eta_mem0*lavg[i]*v0/Fstall[i]) ;
            vt[i] = (Fmemb_tg[i]+repulsive_tg[i])/(eta_mem0*lavg[i]);      
    memb_energy = np.sum(energ_mb)
    #Vs_updated = Vp - v0*(1-(Fsub-F_pro)/Fstall)
    #pdb.set_trace()
    return (F_pro, vt, F_wall, Fmemb_nm, Fmemb_tg, k_mem, arc_len)





def Protrusion_force_correction(Force_pro, Force_ck, Fstall, Fsub):
    nlen = np.shape(Force_ck)[0];  
    for i in range(nlen):
        Fst = Fsub[i]-Force_pro[i]
        if Force_pro[i]>0 and Fst<0:
           Force_ck[i]=Force_ck[i]+(Force_pro[i]-Fsub[i]); Force_pro[i]=Fsub[i]
        elif Force_pro[i]<0 and Fst>Fstall[i]:
           Force_ck[i]=Fstall[i]-Fst; Force_pro[i]=Fsub[i] - Fstall[i];
    return (Force_pro, Force_ck)


'''
Area conservation forces
'''
def Area_conservation_force(Aa, A0, K_Area, dvd_vec_all):
    if Aa>A0:
        F_Area = 0.
    else:
        F_Area = K_Area*(A0 - Aa) #/A0
    F_Area_vec = F_Area*dvd_vec_all
    #print('F_Area', F_Area)
    #print('dvd_vec_all', dvd_vec_all)
    return F_Area_vec

