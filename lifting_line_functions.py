import numpy as np
import matplotlib.pyplot as plt
import json
from numpy import pi, sin, cos, arccos

class lifting_line_class:

    def __init__(self, filename):

        '''Reads inputs from json file specified by filename'''
        
        json_string=open(filename + ".json").read()
        json_vals = json.loads(json_string)
        
        self.planform_type = json_vals["wing"]["planform"]["type"]
        self.R_A = json_vals["wing"]["planform"]["aspect_ratio"]
        self.R_T = json_vals["wing"]["planform"]["taper_ratio"]
        
        self.washout_distribution = json_vals["wing"]["washout"]["distribution"]
        self.Omega = json_vals["wing"]["washout"]["magnitude[deg]"]
        self.CL_design = json_vals["wing"]["washout"]["CL_design"]
        
        self.ail_begin_z = json_vals["wing"]["aileron"]["begin[z/b]"]
        self.ail_end_z = json_vals["wing"]["aileron"]["end[z/b]"]
        self.ail_begin_c = json_vals["wing"]["aileron"]["begin[cf/c]"]
        self.ail_end_c = json_vals["wing"]["aileron"]["end[cf/c]"]
        self.hinge_ef = json_vals["wing"]["aileron"]["hinge_efficiency"]
        self.def_ef = 1.0
        
        self.lift_slope = json_vals["wing"]["airfoil_lift_slope"]
        self.nodes_semi = json_vals["wing"]["nodes_per_semispan"]
        
        self.alphad_root = json_vals["condition"]["alpha_root[deg]"]
        self.ail_deflec = json_vals["condition"]["aileron_deflection[deg]"]
        self.pbar = json_vals["condition"]["pbar"]

        self.view_planform = json_vals["view"]["planform"]
        self.view_washout = json_vals["view"]["washout_distribution"]
        self.view_aileron = json_vals["view"]["aileron_distribution"]

    def theta_distribution(self, num_points):

        spacing = np.linspace(0., 1., num_points)
        theta_R = spacing*(pi/2)
        theta_L = (spacing + 1)*(pi/2)

        theta_dist = np.hstack((theta_R[:-1],theta_L)).T
    
        return theta_dist
    
    def z_distribution(self, theta_dist):

        z_dist = -0.5*cos(theta_dist)

        return z_dist

    def chord_distribution_tapered(self, theta_dist):

        chord_dist = (2/(self.R_A*(1 + self.R_T)))*(1 - (1 - self.R_T)*abs(cos(theta_dist)))
        
        if isinstance(chord_dist, np.ndarray):
            chord_dist[chord_dist[:] <= 0.001] = 0.001
        else:
            if chord_dist <= 0.001:
                chord_dist = 0.001
        
        return chord_dist

    def chord_distribution_elliptic(self, theta_dist):

        chord_dist = (4/(pi*self.R_A))*sin(theta_dist)
        
        if isinstance(chord_dist, np.ndarray):
            chord_dist[chord_dist[:] <= 0.001] = 0.001
        else:
            if chord_dist <= 0.001:
                chord_dist = 0.001

        return chord_dist
    
    def washout_optimum(self, theta_dist, chord_function):
        
        omega = 1 - ((sin(theta_dist))/(chord_function(theta_dist)/chord_function(pi/2.)))
        
        return omega

    def washout_linear(self, theta_dist):

        omega = abs(cos(theta_dist))
        
        return omega
    
    def chi_distribution(self, theta, chord_func):
        
        z = self.z_distribution(theta)
        
        if isinstance(z, np.ndarray):
            chi_dist = np.zeros(len(z))
            n = len(z)
            for i in range(n):
                if z[i] < -self.ail_end_z:
                    chi = 0.
                elif z[i] >= -self.ail_end_z and z[i] <= -self.ail_begin_z:
                    chi = self.flap_effectiveness(z[i], chord_func)
                elif z[i] > -self.ail_begin_z and z[i] < self.ail_begin_z:
                    chi = 0.
                elif z[i] >= self.ail_begin_z and z[i] <= self.ail_end_z:
                    chi = -self.flap_effectiveness(z[i], chord_func)
                elif z[i] > self.ail_end_z:
                    chi = 0.
                chi_dist[i] = chi
        else:
            if z < -self.ail_end_z:
                chi = 0.
            elif z >= -self.ail_end_z and z <= -self.ail_begin_z:
                chi = self.flap_effectiveness(z, chord_func)
            elif z > -self.ail_begin_z and z < self.ail_begin_z:
                chi = 0.
            elif z >= self.ail_begin_z and z <= self.ail_end_z:
                chi = -self.flap_effectiveness(z, chord_func)
            elif z > self.ail_end_z:
                chi = 0.
            chi_dist = chi
            
        return chi_dist
    
    def flap_effectiveness(self, z, chord_func):
        
        cf_c, _ = self.interp_aileron(z, chord_func)
        
        theta_f = arccos(2*(cf_c) - 1)
        efi = 1 - (theta_f - sin(theta_f))/pi
        
        ef = self.hinge_ef*self.def_ef*efi
        return ef
    
    def interp_aileron(self, z, chord_func):
        
        theta = arccos(-2*z)
        theta_end = arccos(-2*self.ail_end_z)
        theta_begin = arccos(-2*self.ail_begin_z)

        c4_dist_end = (0.75 - self.ail_end_c)*chord_func(theta_end)
        c4_dist_begin = (0.75 - self.ail_begin_c)*chord_func(theta_begin)
        
        c4_dist_intrp = (((abs(z) - self.ail_begin_z)/(self.ail_end_z - self.ail_begin_z))*(c4_dist_end - c4_dist_begin) + c4_dist_begin)
        c_z = chord_func(theta)
        cf_c = (0.75*c_z - c4_dist_intrp)/c_z
        
        return cf_c, c_z
        
        
    def solve_C_matrix(self, theta, chord):
        
        N = 2*self.nodes_semi - 1
        index = np.arange(1, N+1, 1)
        C = np.zeros((N,N))
        theta_mat = theta[1:-1,None]*np.ones((N-2,N))
        chord_mat = chord[1:-1,None]*np.ones((N-2,N))
        index_mat = index[None,:]*np.ones((N-2,N))

        C[0,:] = index*index
        C[-1,:] = ((-1)**(index + 1))*index*index

        C[1:-1,:] = ((4/(self.lift_slope*chord_mat)) + (index_mat/sin(theta_mat)))*sin(index_mat*theta_mat)

        return C
    
    def solve_an_coeff(self, C):

        rhs = np.ones(len(C[:,0]))
        a = np.linalg.solve(C, rhs)
        
        return a
    
    def solve_bn_coeff(self, C, omega):

        b = np.linalg.solve(C, omega)
        
        return b
    
    def solve_cn_coeff(self, C, chi):

        c = np.linalg.solve(C, chi)
        
        return c

    def solve_dn_coeff(self, C, theta):

        d = np.linalg.solve(C, cos(theta))
        
        return d
    
    def solve_factors(self,a, b):

        kappa_L = (1 - (1 + pi*self.R_A/self.lift_slope)*a[0])/((1 + pi*self.R_A/self.lift_slope)*a[0])

        e_omg = b[0]/a[0]

        index = np.arange(2,len(a)+1,1)

        kappa_D = np.sum(index[:]*((a[1:]**2)/(a[0]**2)))

        kappa_DL = 2*e_omg*np.sum(index[:]*(a[1:]/a[0])*((b[1:]/b[0]) - (a[1:]/a[0])))
        kappa_Domg = e_omg*e_omg*np.sum(index[:]*((b[1:]/b[0]) - (a[1:]/a[0]))**2)
        kappa_D0 = kappa_D - (kappa_DL*kappa_DL)/(4*kappa_Domg)

        return kappa_L, kappa_D, kappa_DL, kappa_Domg, kappa_D0, e_omg

    def solve_aero_coeff(self, kappa_L, kappa_D, kappa_DL, kappa_Domg, e_omg):
        
        CL_a = self.lift_slope/((1 + self.lift_slope/(pi*self.R_A))*(1 + kappa_L))
        
        if self.washout_distribution == 'optimum':
            self.Omega = (180./pi)*(kappa_DL*self.CL_design)/(2*kappa_Domg*CL_a)

        CL = CL_a*((self.alphad_root) - e_omg*self.Omega)*(pi/180.)
        
        CDi = (CL*CL*(1 + kappa_D) - kappa_DL*CL*CL_a*self.Omega*(pi/180.) + kappa_Domg*(CL_a*self.Omega*(pi/180.))**2)/(pi*self.R_A)
        
        Cl_da = (-pi*self.R_A/4)*self.c[1]
        
        Cl_pbar = (-pi*self.R_A/4)*self.d[1]

        if self.pbar == 'steady':
            self.pbar = -(Cl_da/Cl_pbar)*self.ail_deflec*(pi/180.)
        
        Cl = Cl_da*self.ail_deflec*(pi/180.) + Cl_pbar*self.pbar
        
        return CL_a, CL, CDi, Cl_da, Cl_pbar, Cl
    
    def solve_yaw_coeff(self, A):

        N = 2*self.nodes_semi - 1
        index = 2*np.arange(2, N+1, 1) - 1
        An = A[1:]
        An_1 = A[:-1]
        
        sum_vec = np.sum(index*An_1*An)
        c1 = (pi*self.R_A/4.)
        c2 = (pi*self.R_A*self.pbar/8.)
        c3 = (A[0] + A[2])
        Cn = c1*sum_vec - c2*c3
        
        return Cn
    
    def solve_CD_2(self, A):
        
        N = 2*self.nodes_semi - 1
        index = np.arange(1, N+1, 1)
        
        CD = pi*self.R_A*np.sum(index*A*A) - (pi*self.R_A*self.pbar*A[1]/2)
        
        return CD

    def plot_planform(self, z, chord_function):
        
        theta = arccos(-2*z)
        chord = chord_function(theta)
        quarter_chord = chord/4.
        leading_edge_x = quarter_chord
        trailing_edge_x = -chord + quarter_chord
        quarter_chord_x = np.zeros((len(chord),1))

        # ailerons points
        theta_a_begin = arccos(-2*self.ail_begin_z)
        theta_a_end = arccos(-2*self.ail_end_z)
        c_a_begin = chord_function(theta_a_begin)
        c_a_end = chord_function(theta_a_end)
        
        a_le_x = np.asarray([self.ail_begin_z, self.ail_end_z])
        a_le_y = np.asarray([-(0.75 - self.ail_begin_c)*c_a_begin, -(0.75 - self.ail_end_c)*c_a_end])
        # a_te_x = 
        a_te_y = np.asarray([-(0.75)*c_a_begin, -(0.75)*c_a_end])
        
        z_in_ail = z[z[:] >= self.ail_begin_z]
        z_in_ail = z_in_ail[z_in_ail[:] <= self.ail_end_z]

        cf_in_ail, c_in_ail = self.interp_aileron(z_in_ail, chord_function)
        
        te_in_ail = -0.75*c_in_ail
        le_in_ail = -(0.75 - cf_in_ail)*c_in_ail

        plt.figure(1)
        plt.axis('scaled')
        plt.plot(z,leading_edge_x, color = 'k', linestyle = '-')
        plt.plot(z,trailing_edge_x, color = 'k', linestyle = '-')
        plt.plot(z,quarter_chord_x, color = 'k', linestyle = '-')
        for i in range(len(chord)):
            plt.plot((z[i], z[i]), (leading_edge_x[i], trailing_edge_x[i]), color = 'b', linestyle = '-')
        # plot ailerons
        plt.plot(a_le_x, a_le_y, color = 'k', linestyle = '-')
        plt.plot(-a_le_x, a_le_y, color = 'k', linestyle = '-')
        plt.plot([a_le_x[0], a_le_x[0]], [a_le_y[0], a_te_y[0]], color = 'k', linestyle = '-')
        plt.plot([a_le_x[1], a_le_x[1]], [a_le_y[1], a_te_y[1]], color = 'k', linestyle = '-')
        plt.plot([-a_le_x[0], -a_le_x[0]], [a_le_y[0], a_te_y[0]], color = 'k', linestyle = '-')
        plt.plot([-a_le_x[1], -a_le_x[1]], [a_le_y[1], a_te_y[1]], color = 'k', linestyle = '-')
        for j in range(len(z_in_ail)):
            plt.plot([z_in_ail[j], z_in_ail[j]], [le_in_ail[j], te_in_ail[j]], color = 'r', linestyle = '-')
            plt.plot([-z_in_ail[j], -z_in_ail[j]], [le_in_ail[j], te_in_ail[j]], color = 'r', linestyle = '-')

        plt.ylabel('c/b')
        plt.xlabel('z/b')
        plt.title('Planform')
        plt.xlim((1.1*min(z), 1.1*max(z)))
        plt.ylim((1.25*min(trailing_edge_x), 1.25*max(leading_edge_x)))
        plt.show()

    def plot_washout(self, z, omega):
          
        plt.figure(2)
        # plt.axis('scaled')
        plt.plot(z, omega, color = 'k', linestyle = '-')
        plt.ylabel('omega')
        plt.xlabel('z/b')
        plt.title('Washout Distribution')
        plt.xlim((1.1*min(z), 1.1*max(z)))
        # plt.ylim((1.1*min(omega), 1.1*max(omega)))
        plt.show()
        
    def plot_aileron(self, z, chi):
          
        plt.figure(3)
        # plt.axis('scaled')
        plt.plot(z, chi, color = 'k', linestyle = '-')
        plt.ylabel('Chi')
        plt.xlabel('z/b')
        plt.title('Aileron Distribution')
        plt.xlim((1.1*min(z), 1.1*max(z)))
        # plt.ylim((1.1*min(omega), 1.1*max(omega)))
        plt.show()

    def write_results(self, C, C_inv, a, b, c, d):
        
        with(open('solution.txt', 'w+', encoding='utf-8')) as f:
            
            f.write('{0:<}'.format("C = " + '\n'))
            for i in range(len(C[:,0])):
                for j in range(len(C[0,:])):
                    f.write('{:<16.12f}'.format(C[i,j]))
                f.write('{:}'.format("\n"))
                
            f.write('{:}'.format("\n"))
            f.write('{0:<}'.format("C_inv = " + '\n'))
            for i in range(len(C_inv[:,0])):
                for j in range(len(C_inv[0,:])):
                    f.write('{:<16.12f}'.format(C_inv[i,j]))
                f.write('{:}'.format("\n"))
                
            f.write('{:}'.format("\n"))
            f.write('{0:<}'.format("an= " + '\n'))
            for i in range(len(a)):
                f.write('{:<16.12f}'.format(a[i]))
                f.write('{:}'.format("\n"))
            f.write('{:}'.format("\n"))
            
            f.write('{:}'.format("\n"))
            f.write('{0:<}'.format("bn= " + '\n'))
            for i in range(len(b)):
                f.write('{:<16.12f}'.format(b[i]))
                f.write('{:}'.format("\n"))
            f.write('{:}'.format("\n"))
            
            f.write('{:}'.format("\n"))
            f.write('{0:<}'.format("cn= " + '\n'))
            for i in range(len(c)):
                f.write('{:<16.12f}'.format(c[i]))
                f.write('{:}'.format("\n"))
            f.write('{:}'.format("\n"))
            
            f.write('{:}'.format("\n"))
            f.write('{0:<}'.format("dn= " + '\n'))
            for i in range(len(d)):
                f.write('{:<16.12f}'.format(d[i]))
                f.write('{:}'.format("\n"))
            f.write('{:}'.format("\n"))
        f.close()

    def run(self):

        theta = self.theta_distribution(num_points = self.nodes_semi)
        z_over_b = self.z_distribution(theta)
        
        if self.planform_type == 'tapered':
            
            c_over_b = self.chord_distribution_tapered(theta)
            self.c_root = self.chord_distribution_tapered(pi/2)
            
            if self.washout_distribution == 'optimum':
                omega = self.washout_optimum(theta, self.chord_distribution_tapered)
            elif self.washout_distribution == 'linear':
                omega = self.washout_linear(theta)
                
        elif self.planform_type == 'elliptic':
            
            c_over_b = self.chord_distribution_elliptic(theta)
            self.c_root = self.chord_distribution_elliptic(pi/2)
            
            if self.washout_distribution == 'optimum':
                omega = self.washout_optimum(theta, self.chord_distribution_elliptic)
            elif self.washout_distribution == 'linear':
                omega = self.washout_linear(theta)

        C = self.solve_C_matrix(theta, c_over_b)
        C_inv = np.linalg.inv(C)
        
        if self.planform_type == 'tapered':
            chord_function = self.chord_distribution_tapered
        elif self.planform_type == 'elliptic':
            chord_function = self.chord_distribution_elliptic
    
        chi = self.chi_distribution(theta, chord_function)
        a = self.solve_an_coeff(C)
        b = self.solve_bn_coeff(C, omega)
        self.c = self.solve_cn_coeff(C, chi)
        self.d = self.solve_dn_coeff(C, theta)

        KL, KD, KDL, KDomg, KD0, e_omg = self.solve_factors(a, b)

        CL_a, CL, CDi, Cl_da, Cl_pbar, Cl = self.solve_aero_coeff(KL, KD, KDL, KDomg, e_omg)
        
        A = self.alphad_root*(pi/180.)*a - self.Omega*(pi/180.)*b + self.ail_deflec*(pi/180.)*self.c + self.pbar*self.d
        
        Cn = self.solve_yaw_coeff(A)
        CDi_2 = self.solve_CD_2(A)
        
        print('Angle of Attack [deg]: ', self.alphad_root)
        print('\u039AL: ', KL)
        print('\u039AD: ', KD)
        print('\u039ADL: ', KDL)
        print('\u039AD\u03C9: ', KDomg)
        print('\u039AD0', KD0)
        print('ew: ', e_omg)
        print('\n')
        print('CL: ', CL)
        print('CL,\u03B1: ', CL_a)
        print('CDi: ', CDi)
        print('CDi (corrected): ', CDi_2)
        print('\n')
        print('Cl_da: ', Cl_da)
        print('Cl_pbar: ', Cl_pbar)
        print('Cl: ', Cl)
        print('Cn: ', Cn)
        

        

        self.plot_planform(z_over_b, chord_function)
        self.plot_washout(z_over_b, omega)
        self.plot_aileron(z_over_b, chi)
        self.write_results(C, C_inv, a, b, self.c, self.d)
        