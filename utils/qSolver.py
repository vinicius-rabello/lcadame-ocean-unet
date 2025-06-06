import numpy as np
#from saveNCfile import savenc
#import matplotlib.pyplot as plt

########################################################################################
# N is in x & N2 is in Y
# psiIN must be in lat,lon,2
########################################################################################
def drymodel(psiIN,finalT):   
    opt = 3 # 1 = just the linear parts, 2 = just the nonlinear parts, 3 = full model

    N = np.size(psiIN,axis=0) #zonal size of spectral decomposition
    N2 = np.size(psiIN,axis=1) #meridional size of spectral decomposition
    Lx = 96 #46. #size of x -- stick to multiples of 10
    Ly = 192 #68. #size of y -- stick to multiples of 10

    nu = 2.3*pow( 10., -6. ) #viscous dissipation
    tau_d = 100. #Newtonian relaxation time-scale for interface
    tau_f = 15. #surface friction 
    beta = 0.196 #beta 
    sigma = 3.5
    U_1 = 1.

    g = 0.04 #leapfrog filter coefficient

    y = np.linspace( -Ly / 2, Ly / 2, N2 ) 

#Wavenumbers:
    kk = np.fft.rfftfreq( N, Lx / float(N) / 2. / np.pi ) #zonal wavenumbers
    ll = np.fft.fftfreq( N2, Ly / float(N2) / 2. / np.pi ) #meridional wavenumbers


    tot_time = finalT #1#750 #Length of run
    dt = 0.025 #Timestep
    ts = int(tot_time / dt ) #Total timesteps
    lim = 0#int(650 / dt )#int(ts / 10 ) #Start saving
    st = int( 1. / dt ) #How often to save data
    
    psiAllTime = np.zeros((int(ts/st)+2,N,N2,2))
    #psiAllTime = np.zeros((int(int(3*365 / dt )/st)+2,N,N2,2))
    

#######################################################
#  Declare arrays

#Spectral arrays, only need 3 time-steps

    psic_1 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
    psic_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
    qc_1 = np.zeros( ( ( 3 , N2, int(N / 2 + 1) ) ) ).astype( complex )
    qc_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
    vorc_1 = np.zeros( ( ( 3 , N2, int(N / 2 + 1)  ) ) ).astype( complex )
    vorc_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )

    print('complex spectral array shapes: ' + str(psic_1.shape))
    
    #Real arrays, only need 3 time-steps
    psi_1 = np.zeros( ( ( 3 , N2 , N ) ) )
    psi_2 = np.zeros( ( ( 3 , N2 , N ) ) )
    q_1 = np.zeros( ( ( 3 , N2, N ) ) )
    q_2 = np.zeros( ( ( 3 , N2 , N ) ) )
    
    print('real array shapes: ' + str(psi_1.shape))
    
    
    #######################################################
    #  Define equilibrium interface height + sponge
    
    sponge = np.zeros( N2)
    u_eq = np.zeros( N2)
    
    for i in range( N2 ):
        y1 = float( i - N2 /2) * (y[1] - y[0] )
        y2 = float(min(i, N2 -i - 1)) * (y[1] - y[0] )
        sponge[i] = U_1 / (np.cosh(abs(y2/sigma)))**2 
        u_eq[i] = U_1 * ( 1. / (np.cosh(abs(y1/sigma)))**2 - 1. / (np.cosh(abs(y2/sigma)))**2  )

    psi_Rc = -np.fft.fft(  u_eq ) / 1.j / ll
    psi_Rc[0] = 0.
    psi_R = np.fft.ifft(psi_Rc )
    
    
    #######################################################
    #  Spectral functions
    
    def ptq(ps1, ps2):
        """Calculate PV"""
        q1 = -(ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2 ) * ps1 - (ps1 - ps2) # -(k^2 + l^2) * psi_1 -0.5*(psi_1-psi_2)
        q2 = -(ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2 ) * ps2 + (ps1 - ps2) # -(k^2 + l^2) * psi_2 +0.5*(psi_1-psi_2)
        return q1, q2
    
    def qtp(q1_s, q2_s):
        """Invert PV"""
        # Checked
        psi_bt = -(q1_s + q2_s) / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2) / 2.  # (psi_1 + psi_2)/2
        psi_bc = -(q1_s - q2_s) / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2
        psi_bt[0, 0] = 0.
        psi1 = psi_bt + psi_bc
        psi2 = psi_bt - psi_bc
    
        return psi1, psi2
    
    #######################################################
    #  Initial conditions:
    
    psi1 = np.transpose(psiIN[:, : , 0 ])
    psi2 = np.transpose(psiIN[:, : , 1 ])
    
    
    psic_1[0] = np.fft.rfft2(psi1)
    psic_2[0] = np.fft.rfft2(psi2)
    		
    #Transfer values:
    psic_1[ 1 , : , : ] = psic_1[ 0 , : , : ]
    psic_2[ 1 , : , : ] = psic_2[ 0 , : , : ]
    
    #Calculate initial PV
    for i in range( 2 ):
    	vorc_1[i], vorc_2[i] = ptq(psic_1[i], psic_2[i]) 
    	q_1[i] = np.fft.irfft2( vorc_1[i]) + beta * y[:, np.newaxis]
    	q_2[i] = np.fft.irfft2( vorc_2[i]) + beta * y[:, np.newaxis]
    	qc_1[i] = np.fft.rfft2( q_1[i] )
    	qc_2[i] = np.fft.rfft2( q_2[i] )
    
    
    
    #######################################################
    # Time-stepping functions
    
    def calc_nl( psi, qc ):
        """"Calculate non-linear terms, with Orszag 3/2 de-aliasing"""
    
        N2, N = np.shape( psi )
        ex = int(N *  3 / 2)# - 1
        ex2 = int(N2 * 3 / 2)# - 1
        temp1 = np.zeros( ( ex2, ex ) ).astype( complex )
        temp2 = np.zeros( ( ex2, ex ) ).astype( complex )
        temp4 = np.zeros( ( N2, N ) ).astype( complex )	#Final array
    
        #Pad values:
        temp1[:N2//2, :N] = psi[:N2//2, :N]
        temp1[ex2-N2//2:, :N] = psi[N2//2:, :N]
    
        temp2[:N2//2, :N] = qc[:N2//2, :N]
        temp2[ex2-N2//2:, :N] = qc[N2//2:, :N]
    
        #Fourier transform product, normalize, and filter:
        temp3 = np.fft.rfft2( np.fft.irfft2( temp1 ) * np.fft.irfft2( temp2 ) ) * 9. / 4.
        temp4[:N2//2, :N] = temp3[:N2//2, :N]
        temp4[N2//2:, :N] = temp3[ex2-N2//2:, :N]
    
        return temp4
    
    def nlterm(kk, ll, psi, qc):
        """"Calculate Jacobian"""
    
        dpsi_dx = 1.j * kk[np.newaxis, :] * psi 
        dpsi_dy = 1.j * ll[:, np.newaxis] * psi 
    
        dq_dx = 1.j * kk[np.newaxis, :] * qc
        dq_dy = 1.j * ll[:, np.newaxis] * qc 
    
        return  calc_nl( dpsi_dx, dq_dy ) - calc_nl( dpsi_dy, dq_dx )
    
    def fs(ovar, rhs, det, nu, kk, ll):
        # backward Euler of dq/dt=RHS-aq
        """Forward Step: q^t-1 / ( 1 + 2. dt * nu * (k^4 + l^4 ) ) + RHS"""
        mult = det / ( 1. + det * nu * (np.expand_dims(kk, 0) ** 8 + np.expand_dims(ll, 1) ** 8) )
    
        return mult * (ovar / det + rhs)
    
    def lf(oovar, rhs, det, nu, kk, ll):
        """Leap frog timestepping: q^t-2 / ( 1 + 2. * dt * nu * (k^4 + l^4 ) ) + RHS"""
        mult = 2. * det / ( 1. + 2. * det * nu * (np.expand_dims(kk, 0) ** 8 + np.expand_dims(ll, 1) ** 8) )
        return mult * (oovar / det / 2. + rhs)
    
    def filt(var, ovar, nvar, g):
        # see Analysis of time filters used with the leapfrog scheme
    	"""Leapfrog filtering"""
    	return var + g * (ovar - 2. * var + nvar )
    
    
    #######################################################
    #  Main time-stepping loop
    
    forc1 = np.zeros( ( N2, N ) )
    forc2 = np.zeros( ( N2, N ) )
    cforc1 = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)
    cforc2 = np.zeros( ( N2, N // 2 + 1  ) ).astype(complex)
    
    nl1 = np.zeros( ( N2, N // 2 + 1  ) ).astype(complex)
    nl2 = np.zeros( ( N2, N // 2 + 1 ) ).astype(complex)
    
    psiAll = np.zeros( ( ( ( N, N2 , 2 ) ) ) )
    #Timestepping:
    cnt = 0
    cnt_ave = 0
    ave_u1,ave_u2,ave_uv1,ave_uv2,ave_T1,ave_T2 = np.zeros(N2),np.zeros(N2),np.zeros(N2),np.zeros(N2),np.zeros(N2),np.zeros(N2)
    for i in range( 1, ts+1):
        if i % 100 == 0:
            print("Timestep:", i, "/", ts)
    
        if opt > 1:
    	#NL terms -J(psi, qc) - beta * v <<### Still we have problem here
            nl1[:, :] = -nlterm( kk, ll, psic_1[1, :, :], vorc_1[1, :, :]) - beta * 1.j * kk[np.newaxis, :] * psic_1[1, :, :]
            nl2[:, :] = -nlterm( kk, ll, psic_2[1, :, :], vorc_2[1, :, :]) - beta * 1.j * kk[np.newaxis, :] * psic_2[1, :, :]
            #nl1[:, :] = -nlterm( kk, ll, psic_1[1, :, :], vorc_1[1, :, :]) + beta * 1.j * kk[np.newaxis, :] * psic_1[1, :, :]
            #nl2[:, :] = -nlterm( kk, ll, psic_2[1, :, :], vorc_2[1, :, :]) + beta * 1.j * kk[np.newaxis, :] * psic_2[1, :, :]
    
        if opt != 2:
    	#Linear terms
    	#Relax interface
            forc1[:, :] = (psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) / tau_d 
            forc2[:, :] = -(psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) / tau_d
    
    	#Sponge
            forc1[:, :] -= sponge[:, np.newaxis] * (q_1[1] - np.mean( q_1[1], axis = 1)[:, np.newaxis] )
            forc2[:, :] -= sponge[:, np.newaxis] * (q_2[1] - np.mean( q_2[1], axis = 1)[:, np.newaxis] )
    
            #Convert to spectral space + add friction
            cforc1 = np.fft.rfft2( forc1 )
            cforc2 = np.fft.rfft2( forc2 ) + ( kk[np.newaxis, :] ** 2  + ll[:, np.newaxis] ** 2 ) * psic_2[1] / tau_f
    
        rhs1 = nl1[:] + cforc1[:]
        rhs2 = nl2[:] + cforc2[:]
        #mrhs = mnl[:]
    	
        if i == 1:
    	#Forward step
            qc_1[2, :] = fs(qc_1[1, :, :], rhs1[:], dt, nu, kk, ll)
            qc_2[2, :] = fs(qc_2[1, :, :], rhs2[:], dt, nu, kk, ll)
        else:
    	#Leapfrog step
            qc_1[2, :, :] = lf(qc_1[0, :, :], rhs1[:], dt, nu, kk, ll)
            qc_2[2, :, :] = lf(qc_2[0, :, :], rhs2[:], dt, nu, kk, ll)
    
        if i > 1:
    	#Leapfrog filter (our target is qc_1[1] which is time n)
            qc_1[1, :] = filt( qc_1[1, :], qc_1[0, :], qc_1[2, :], g)
            qc_2[1, :] = filt( qc_2[1, :], qc_2[0, :], qc_2[2, :], g)
    
        ## Calcs
        dPsidyC_1 = - 1.j * ll[:,np.newaxis] * psic_1[1, :, :]
        dPsidyC_2 = - 1.j * ll[:,np.newaxis] * psic_2[1, :, :]
        dPsidxC_1 =   1.j * kk[np.newaxis,:] * psic_1[1, :, :]
        dPsidxC_2 =   1.j * kk[np.newaxis,:] * psic_2[1, :, :]        
        dPsidy1 = np.fft.irfft2( dPsidyC_1 )
        dPsidy2 = np.fft.irfft2( dPsidyC_2 )
        dPsidx1 = np.fft.irfft2( dPsidxC_1 )
        dPsidx2 = np.fft.irfft2( dPsidxC_2 )
        
        umean1 = np.mean(dPsidy1, axis=1)
        umean2 = np.mean(dPsidy2, axis=1)
        uvmean1 = np.mean(dPsidy1*dPsidx1, axis=1)
        uvmean2 = np.mean(dPsidy2*dPsidx2, axis=1)
        ave_u1 = ave_u1 + umean1
        ave_u2 = ave_u2 + umean2
        ave_uv1 = ave_uv1 + uvmean1
        ave_uv2 = ave_uv2 + uvmean2
        cnt_ave = cnt_ave + 1
        
        for j in range( 2 ):
            q_1[j] = np.fft.irfft2( qc_1[j + 1] )
            q_2[j] = np.fft.irfft2( qc_2[j + 1] )
    
    	#Subtract off beta and invert
            vorc_1[j] = np.fft.rfft2( q_1[j] - beta * y[:, np.newaxis])
            vorc_2[j] = np.fft.rfft2( q_2[j] - beta * y[:, np.newaxis])
            psic_1[j], psic_2[j] = qtp( vorc_1[j], vorc_2[j] )
            psi_1[j] = np.fft.irfft2( psic_1[j] )
            psi_2[j] = np.fft.irfft2( psic_2[j] )
    
            #Transfer values:
            qc_1[j, :, :] = qc_1[j + 1, :, :]
            qc_2[j, :, :] = qc_2[j + 1, :, :]
    
        ave_T1 = ave_T1 + np.mean(dPsidx1*(psi_2[1]-psi_1[1]),axis=1)
        ave_T2 = ave_T2 + np.mean(dPsidx2*(psi_2[1]-psi_1[1]),axis=1)

        if i > lim:
           if i % st == 0:
                psiAll[: , : , 0 ] = np.transpose(psi_1[1])
                psiAll[: , : , 1 ] = np.transpose(psi_2[1])
                psiAllTime[cnt,:,:,0] = np.transpose(psi_1[1])
                psiAllTime[cnt,:,:,1] = np.transpose(psi_2[1])
                cnt = cnt + 1
    return psiAll,psiAllTime,cnt_ave,ave_u1,ave_u2,ave_uv1,ave_uv2,ave_T1,ave_T2
########################################################################################

# Initial settings
Nlat,Nlon = 46,68 #96,192
sig_m= 1.0 #0.15
# Initial condition
psi0=(np.random.normal(0,sig_m,[2*Nlat*Nlon,])).reshape([Nlat,Nlon,2])
print(psi0.shape)

# If necessary to restart from somewhere
#psi0 = np.load('ICs/matrix.npy')

T = 360*2 #3605
output=np.zeros([T+1,Nlon,Nlat,2])

# Call sover for specified amount of time
op,output,c1,c2,c3,c4,c5,c6,c7 = drymodel(psi0,T)

# save resutls in numpy for furture use
np.save('ICs/matrix_plus.npy', op)
np.save('ICs/results.npy',output)
np.save('ICs/c2.npy',c2)
np.save('ICs/c3.npy',c3)
np.save('ICs/c4.npy',c4)
np.save('ICs/c5.npy',c5)
np.save('ICs/c6.npy',c6)
np.save('ICs/c7.npy',c7)
print("End at ",c1)
exit()

# Save in netCFD4 format
lat = np.zeros(Nlat)
lon = np.zeros(Nlon)
#savenc(output,lon,lat,'ICs/IC.nc')

import matplotlib.pyplot as plt
x = np.linspace(0,46,Nlat)
y = np.linspace(0,68,Nlon)
X,Y = np.meshgrid(x,y)


## plot average
c2 = c2 / c1
c3 = c3 / c1
c4 = c4 / c1
c5 = c5 / c1
c6 = c6 / c1
c7 = c7 / c1
plt.plot(y,c2,'b')
plt.plot(y,c3,'r')
plt.savefig("uave.png", dpi=300)
plt.show()

plt.clf()
plt.plot(y,c4,'b')
plt.plot(y,c5,'r')
plt.plot(y,c6,'g')
plt.plot(y,c7,'k')
plt.savefig("uvave.png", dpi=300)
plt.show()
#np.save('ICs/zonal_wind.npy', op)
## plot average
plt.clf()
#exit()

levels = np.linspace(-2.5, 2.5, 10)
for i in range(output.shape[0]):
    plt.set_cmap('bwr')
    Z1 = np.transpose(output[i,:,:,0])
    Z2 = np.transpose(output[i,:,:,1])
    contour = plt.contourf(X,Y,Z2,levels=levels)
    #colorbar = plt.colorbar.set_clim(-2.5, 2.5)
    if i==0:
        plt.colorbar(contour, label='Value')
    filename = f'images/plot_{i+1:0{3}}.png'
    plt.savefig(filename, dpi=300)
    plt.show()
    print(i)

print("Finished!")

