import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import broyden1 , bisect


class FVMG:
    def __init__(self,case,MGType,Courant,Ncells,beta,its):
        """"
        FVMG: is a finit-volume multigrid solver for the 1-D Euler equations
        Test problems:
                Subsonic channel flow
                Transonic channel flow
        Inputs: 
        case: subsonic or transonic
        Ncells: number of grid cells
        Courant: Courant number
        beta: smoothing factor
        MGType: Multigrid type: V cycle or W cycle and number of grids

        Output:
            Mach, pressure, density, sound speed, energy (exact and numerical)
        """
        self.case =  case
        self.N = Ncells   # 95
        self.C =  Courant
        self.beta = beta
        self.schedule = FVMG.schedules(self,MGType)
        self.gamma = 1.4 
        self.k2 = 0.5
        self.k4 = 1/32
        self.file = 'transonic.txt' # only for the transonic
        self.its = its
        xl = 0
        xm = 5
        xr = 10  
        R = 287
        T0 = 300
        p01 = 100000
        
        self.x , self.dx , self.S , self.Dx , self.dS = FVMG.grid(self,xl,xm,xr)
        
        
        # Exact solution
        if self.case == 'subsonic':
            Sstar = 0.8
            self.M_ex, T_ex, self.p_ex, rho_ex, a_ex, u_ex, e_ex = FVMG.subsonic(self,R,T0,p01,Sstar)
        elif self.case == 'transonic':
            xshock = 7
            SstarL = 1
            self.M_ex, T_ex, self.p_ex, rho_ex, a_ex, u_ex, e_ex = FVMG.transonic(self,xshock,SstarL,T0,p01,R)
        Q_ex = np.concatenate((rho_ex[1:-1],
                                rho_ex[1:-1]*u_ex[1:-1],
                                e_ex[1:-1]),axis=None)
        QS_ex = np.concatenate((rho_ex[1:-1]* self.S[1:-1],
                                rho_ex[1:-1]*u_ex[1:-1]* self.S[1:-1],
                                e_ex[1:-1]*self.S[1:-1]),axis=None)
        self.R2L,self.R3L = FVMG.GetRiemannInv(self,rho_ex[0],rho_ex[0]*u_ex[0],e_ex[0])[1:]
        self.R1R = FVMG.GetRiemannInv(self,rho_ex[-1],rho_ex[-1]*u_ex[-1],e_ex[-1])[0]
        rho0 = ((rho_ex[-1]-rho_ex[0])/(xr-xl))*self.x[1:-1] + rho_ex[0] 
        u0 = ((u_ex[-1]-u_ex[0])/(xr-xl))*self.x[1:-1] + u_ex[0] 
        e0 = ((e_ex[-1]-e_ex[0])/(xr-xl))*self.x[1:-1] + e_ex[0]
        # Initial conditions
        QS = np.array([rho0 *self.S[1:-1],
                        rho0*u0 *self.S[1:-1],
                        e0 *self.S[1:-1]]).flatten()
        # Initial residual
        Rnorm = np.sqrt(sum((FVMG.ResidualVector(self,QS))**2))
        self.QS1 , self.residuals1  = FVMG.multistage(self,QS,self.its,return_his=True)
        self.residuals1.insert(0,Rnorm)
        self.residuals1 = self.residuals1 / Rnorm
        self.QS2 , self.residuals2  = FVMG.Multigrid(self,QS,self.its,return_his=True)
        self.residuals2.insert(0,Rnorm)
        self.residuals2 = self.residuals2 / Rnorm
        QS = np.copy(self.QS2)
        bdyQS = FVMG.extrapolate1st(self,QS)
        self.rhoS = np.pad(QS[:self.N],(1,1),'constant',
                            constant_values=(bdyQS[0],bdyQS[3]))
        self.rhouS = np.pad(QS[self.N:2*self.N],(1,1),'constant',
                            constant_values=(bdyQS[1],bdyQS[4]))
        self.eS = np.pad(QS[2*self.N:],(1,1),'constant',
                        constant_values=(bdyQS[2],bdyQS[5]))
        self.pres  = FVMG.pressure(self,self.rhoS,self.rhouS,self.eS)/self.S
        self.Mach = self.rhouS/self.rhoS/FVMG.sound(self,self.rhoS,self.rhouS,self.eS)


    def plotting(self,X,Y,axis,labels,colors,lines, title,
                 logscale = False,location = 'upper right',lw= 1):
        plt.rc('font', family='serif')
        plt.rc('font',size=10)
        plt.rc('axes',labelsize=10)
        fig = plt.figure(figsize = (4,4))
        ax = fig.add_subplot(111)
        for i, x in enumerate(X):
            if logscale:
                ax.semilogy(Y[i], lines[i], color = colors[i],
                            label = labels[i], markevery = 2 ,linewidth = lw)
            else:
                ax.plot(x, Y[i] , lines[i], color = colors[i],
                        label = labels[i], markevery = 2 ,linewidth = lw)
        ax.set_title(title)
        ax.set_xlabel(axis[0])
        ax.set_ylabel(axis[1])
        ax.legend(numpoints = 1, loc = location,fontsize = 9, frameon = False)
        fig.tight_layout()
        plt.show()
             
    def grid(self,xl, xm, xr):
        """
        Computes the grid vector, the grid spacing, the difference operator,
        and the channel area
        """
        x1 = np.linspace(xl, xm, num=int((self.N - 1) / 2 + 2), endpoint=True)  # including xl
        x2 = np.linspace(xm, xr, num=int((self.N - 1) / 2 + 2), endpoint=True)  # including xr
        x = np.concatenate((x1[:-1], x2), axis=None)  # including xl and xr
        dx = x[1] - x[0]  # spacing

        # Channel area variation
        S1 = 1 + 1.5 * (1 - x1[:-1] / 5) ** 2
        S2 = 1 + 0.5 * (1 - x2 / 5) ** 2
        S = np.concatenate((S1, S2), axis=None)
        Dx1 = (np.diag(np.ones(self.N - 1), 1) + np.diag(-np.ones(self.N - 1), -1)) / 2 / dx
        Dx = np.block([[Dx1, np.zeros((self.N, self.N)), np.zeros((self.N, self.N))],
                    [np.zeros((self.N, self.N)), Dx1, np.zeros((self.N, self.N))],
                    [np.zeros((self.N, self.N)), np.zeros((self.N, self.N)), Dx1]])
        dS = Dx1 @ S[1:-1]
        dS[0] -= S[0] / 2 / dx
        dS[-1] += S[-1] / 2 / dx
        return x, dx, S, Dx, dS

    def Evector(self,Q):
        """
        Computes vector E given the vector Q
        """
        N = int(len(Q)/3)
        P = FVMG.pressure(self,Q[:N],Q[N:2*N],Q[2*N:])
        E = np.empty(np.shape(Q))
        E[:N] = Q[N:2*N]
        E[N:2*N] = Q[N:2*N]**2/Q[:N]+P
        E[2*N:] = Q[N:2*N]/Q[:N]*(Q[2*N:]+P)
        return E
    
    def Fvector(self,QS):
        """
        Compute the source term F given QS
        """
        M = int(len(QS)/3)
        dS = FVMG.RestrictdS(self,M)
        S = FVMG.RestrictS(self,M)
        P = FVMG.pressure(self,QS[:M],QS[M:2*M],QS[2*M:]) /S[1:-1]
        F = np.zeros(3*M)
        F[M:2*M] = P * dS
        return F

    def D_x(self,N):
        """
        Make Dx in block form
        """
        Dx1 = (np.diag(np.ones(N-1),1)+np.diag(-np.ones(N-1),-1))/2/self.dx
        return np.block([[Dx1,np.zeros((N,N)),np.zeros((N,N))],
                    [np.zeros((N,N)),Dx1,np.zeros((N,N))],
                    [np.zeros((N,N)),np.zeros((N,N)),Dx1]])
                    
    def ResidualVector(self,QS,P=0):
        """
        Construct the residual vector given QS
        """
        M = int(len(QS)/3)
        bdyQS = FVMG.extrapolate1st(self,QS)
        D = FVMG.ArtifDiss(self,QS,*bdyQS)
        return - FVMG.D_x(self,M) @ FVMG.Evector(self,QS) - FVMG.BdyDiffE(self,*bdyQS,M) + FVMG.Fvector(self,QS) + D + P
    
    def RestrictS(self,M):
        R = np.zeros((M+2,self.N+2))
        skip = int((self.N+1)/(M+1))
        for i in range(M+2):
            R[i,i*skip] = 1
        return R@ self.S
    
    def RestrictdS(self,M):
        R = np.zeros((M,self.N))
        skip = int((self.N+1)/(M+1))
        if skip ==1:
            R = np.identity(M)
        else:
            for i in range(M):
                R[i,i*skip+1] = 1
        return R @ self.dS

    def ArtifDiss(self,QS,QS1L,QS2L,QS3L,QS1R,QS2R,QS3R):
        """
        Artificial dissipation routine. Computes the second and fourth order
        dissipation terms 
        """
        M = int(len(QS)/3)
        S = FVMG.RestrictS(self,M)  
        rho = np.pad(QS[:M],(1,1),'constant',
                     constant_values=(QS1L,QS1R)) /S
        rhou = np.pad(QS[M:2*M],(1,1),'constant',
                      constant_values=(QS2L,QS2R)) /S
        e = np.pad(QS[2*M:],(1,1),'constant',
                   constant_values=(QS3L,QS3R)) /S
        P = FVMG.pressure(self,rho,rhou,e)
        a = FVMG.sound(self,rho,rhou,e)

        Y = np.pad(abs((P[2:]-2*P[1:-1]+P[:-2])/(P[2:]+2*P[1:-1]+P[:-2])),
                (2,2),'constant',constant_values=(0, 0))
        eps2 = self.k2*np.maximum.reduce([Y[2:],Y[1:-1],Y[:-2]]) 
        eps4 = np.maximum(np.zeros(M+2),self.k4-eps2)
        # Second order dissipation term D2
        C = eps2*(abs(rhou/rho) + a)*S
        C1 = C[1:-1]+C[2:]
        C2 = C[1:-1]+C[:-2]
        C3 = C[2:]+2*C[1:-1]+C[:-2]
        D2 = np.array([(C1*rho[2:] + C2*rho[:-2] -C3*rho[1:-1]),
                    (C1*rhou[2:] + C2*rhou[:-2] -C3*rhou[1:-1]),
                    (C1*e[2:] + C2*e[:-2] -C3*e[1:-1])]).flatten() /2/ self.dx
    
        # Fourth order dissipation term D4
        C = eps4*(abs(rhou/rho) + a)*S 
        C1 = C[1:-1]+C[2:]
        C2 = C[1:-1]+C[:-2]
        C3 = 4*C[1:-1]+C[:-2]+3*C[2:]
        C4 = 4*C[1:-1]+3*C[:-2]+C[2:]
        C5 = 6*C[1:-1]+3*C[:-2]+3*C[2:]
        C4[0] = C[0]+2*C[1]+C[2]
        C5[0] = 2*C[0]+5*C[1]+3*C[2]

        C3[-1] = C[-1]+2*C[-2]+C[-3]
        C5[-1] = 2*C[-1]+5*C[-2]+3*C[-3]
        drho = np.empty(M)
        drhou = np.empty(M)
        de = np.empty(M)
        drho[1:-1] = -C1[1:-1]*rho[4:] \
                     -C2[1:-1]*rho[:-4] \
                     +C3[1:-1]*rho[3:-1] \
                     +C4[1:-1]*rho[1:-3] \
                     -C5[1:-1]*rho[2:-2]
        drhou[1:-1] = -C1[1:-1]*rhou[4:] \
                      -C2[1:-1]*rhou[:-4] \
                      +C3[1:-1]*rhou[3:-1] \
                      +C4[1:-1]*rhou[1:-3] \
                      -C5[1:-1]*rhou[2:-2] 
        de[1:-1] = -C1[1:-1]*e[4:] \
                   -C2[1:-1]*e[:-4] \
                   +C3[1:-1]*e[3:-1] \
                   +C4[1:-1]*e[1:-3] \
                   -C5[1:-1]*e[2:-2]

        drho[0] = C4[0]*rho[0] \
                  -C5[0]*rho[1] \
                  +C3[0]*rho[2] \
                  -C1[0]*rho[3]
        drhou[0] = C4[0]*rhou[0] \
                   -C5[0]*rhou[1] \
                   +C3[0]*rhou[2] \
                   -C1[0]*rhou[3]
        de[0] = C4[0]*e[0] \
                -C5[0]*e[1] \
                +C3[0]*e[2] \
                -C1[0]*e[3]
        drho[-1] = C3[-1]*rho[-1] \
                   -C5[-1]*rho[-2] \
                   +C4[-1]*rho[-3] \
                   -C2[-1]*rho[-4]
        drhou[-1] = C3[-1]*rhou[-1] \
                    -C5[-1]*rhou[-2] \
                    +C4[-1]*rhou[-3] \
                    -C2[-1]*rhou[-4]
        de[-1] = C3[-1]*e[-1] \
                 -C5[-1]*e[-2] \
                 +C4[-1]*e[-3] \
                 -C2[-1]*e[-4]
        D4 = np.array([drho,drhou,de]).flatten() /2/ self.dx
        return D2+D4
    
    def BdyDiffE(self,Q1L,Q2L,Q3L,Q1R,Q2R,Q3R,N):
        """
        Boundary vector for difference operator Dx given
        Q1, Q2, and Q3 at xL and xR nodes
        """
        bcE = np.zeros(3*N)
        bcE[0] = -Q2L/2/self.dx
        bcE[N-1] = Q2R/2/self.dx
        bcE[N] = -((self.gamma-1)*Q3L + (3-self.gamma)*Q2L**2/Q1L/2)/2/self.dx
        bcE[2*N-1] = ((self.gamma-1)*Q3R + (3-self.gamma)*Q2R**2/Q1R/2)/2/self.dx
        bcE[2*N] = -(self.gamma*Q3L*Q2L/Q1L - (self.gamma-1)*Q2L**3/Q1L**2/2)/2/self.dx
        bcE[3*N-1] = (self.gamma*Q3R*Q2R/Q1R - (self.gamma-1)*Q2R**3/Q1R**2/2)/2/self.dx
        return bcE

    def GetRiemannInv(self,rho,rhou,e):
        """
        Rieman invariants where R1 corresponds to u-a,
        R2 corresponds to u+a, and R3 to u
        """
        u = rhou/rho
        R = 2/(self.gamma-1)*FVMG.sound(self,rho,rhou,e)
        return u-R, u+R, np.log(FVMG.pressure(self,rho,rhou,e)/rho**self.gamma)

    def RiemanntoQ(self,R1,R2,R3):
        k = ((self.gamma-1)*(R2-R1)**2 + 2*self.gamma*(R1+R2)**2)/16/self.gamma
        Q3 = k*((self.gamma-1)**2*(R2-R1)**2/16/self.gamma/np.e**R3)**(1/(self.gamma-1))
        Q1 = Q3/k
        Q2 = Q1/2*(R1+R2)
        return Q1 , Q2 , Q3

    def PermuteForward(self,Q):
        N = int(len(Q)/3)
        I = np.zeros((3*N,3*N))
        for i in range(N):
            I[3*i,i],I[3*i+1,i+N],I[3*i+2,i+2*N] = 1,1,1 
        if len(np.shape(Q))==1: # vector
            return I@Q
        if len(np.shape(Q))==2: # matrix
            return I@Q@I.T
    
    def PermuteBackward(self,Q):
        N = int(len(Q)/3)
        I = np.zeros((3*N,3*N))
        for i in range(N):
            I[3*i,i],I[3*i+1,i+N],I[3*i+2,i+2*N] = 1,1,1
        if len(np.shape(Q))==1: # vector
            return I.T@Q
        if len(np.shape(Q))==2: # matrix
            return I.T@Q@I

    def BuildSmoothing(self,N):
        """ Smoothing operator for beta"""

        B = (np.diag(np.ones(N)+2*self.beta) - np.diag(self.beta*np.ones(N-1),1)
            - np.diag(self.beta*np.ones(N-1),-1))
        B[0,0] , B[-1,-1] = 1 , 1
        B[0,1] , B[-1,-2] = 0 , 0
        Binv = np.linalg.inv(B)
        return np.block([[Binv,np.zeros((N,N)),np.zeros((N,N))],
                    [np.zeros((N,N)),Binv,np.zeros((N,N))],
                    [np.zeros((N,N)),np.zeros((N,N)),Binv]])
    
    def multistage(self,QS,its,P=0,return_his=False):
        """Multistage timemarching routine"""
        QS = np.copy(QS)
        
        if return_his: residuals = []

        for i in range(its):
            N = int(len(QS)/3)
            h = self.C*self.dx/(abs(QS[N:2*N]/QS[:N]) + FVMG.sound(self,QS[:N],QS[N:2*N],QS[2*N:]))
            h = np.concatenate((h,h,h), axis=None)
            
            B = FVMG.BuildSmoothing(self,N)
            Dx = FVMG.D_x(self,N)
            
            # first stage
            bdyQS = FVMG.extrapolate1st(self,QS)
            D0 = FVMG.ArtifDiss(self,QS,*bdyQS)
            R0 = B@(h*(-Dx@ FVMG.Evector(self,QS) \
                       - FVMG.BdyDiffE(self,*bdyQS,N) \
                       + FVMG.Fvector(self,QS) + D0 + P))
            QS1 = QS + 0.25*R0
            
            # second stage
            bdyQS1 = FVMG.extrapolate1st(self,QS1) 
            R1 = B@(h*(-Dx@FVMG.Evector(self,QS1) \
                       - FVMG.BdyDiffE(self,*bdyQS1,N) \
                       + FVMG.Fvector(self,QS1) + D0 + P))
            QS2 = QS + R1/6
            
            # third stage
            bdyQS2 = FVMG.extrapolate1st(self,QS2) # Get boundary QS values using Riemann extrapolation
            D2 = FVMG.ArtifDiss(self,QS2,*bdyQS2)
            R2 = B@(h*(-Dx@ FVMG.Evector(self,QS2) \
                       - FVMG.BdyDiffE(self,*bdyQS2,N) \
                       + FVMG.Fvector(self,QS2) \
                       + 0.44*D0 + 0.56*D2 + P))
            QS3 = QS + 3/8*R2
            
            # fourth stage
            bdyQS3 = FVMG.extrapolate1st(self,QS3) # Get boundary QS values using Riemann extrapolation
            R3 = B@(h*(-Dx@ FVMG.Evector(self,QS3) \
                       - FVMG.BdyDiffE(self,*bdyQS3,N) \
                       + FVMG.Fvector(self,QS3) \
                       + 0.44*D0 + 0.56*D2 + P))
            QS4 = QS + 0.5*R3
            
            # fifth stage
            bdyQS4 = FVMG.extrapolate1st(self,QS4) # Get boundary QS values using Riemann extrapolation
            D4 = FVMG.ArtifDiss(self,QS4,*bdyQS4)
            R4 = B@(h*(-Dx@ FVMG.Evector(self,QS4) - FVMG.BdyDiffE(self,*bdyQS4,N) + FVMG.Fvector(self,QS4) + 0.2464*D0 + 0.3136*D2 + 0.44*D4 + P))
            QS = QS + R4
        
            if return_his: # return convergence history
                residuals.append(np.sqrt(sum((FVMG.ResidualVector(self,QS))**2)))
        
        if return_his: return QS , residuals
        else: return QS

    def Restrict(self,Q,weighted=False):
        # restrict function
        a = len(Q)/3
        R = np.zeros((int((a-1)/2),int(a)))
        if weighted:
            for i in range(int((a-1)/2)):
                R[i,(i+1)*2-2] = 0.25
                R[i,(i+1)*2] = 0.25
                R[i,(i+1)*2-1] = 0.5
        else:
            for i in range(int((a-1)/2)):
                R[i,(i+1)*2-1] = 1
        size = np.shape(R)
        R = np.block([[R,np.zeros(size),np.zeros(size)],
                    [np.zeros(size),R,np.zeros(size)],
                    [np.zeros(size),np.zeros(size),R]])
        return R@Q
    
    def Prolong(self,Q):
        aa = int(len(Q)/3)
        a = 2*aa + 1 
        I = np.zeros((a,aa))
        I[0,0]=0.5
        I[-1,-1]=0.5
        for i in range(1,a-1):
            if i%2==1:
                I[i,i//2] = 1
            elif i%2 ==0:
                I[i,int(i/2)] = 0.5
                I[i,int(i/2)-1] = 0.5
        size = np.shape(I)
        I = np.block([[I,np.zeros(size),np.zeros(size)],
                    [np.zeros(size),I,np.zeros(size)],
                    [np.zeros(size),np.zeros(size),I]])
        return I@Q

    def Multigrid(self,QS,its,return_his=False):
        """
        Multigrid approach:
        V and W cycles are implemented in a single routine
        """
        
        residuals = []
        global count
        
        def solve(QS,P):

            Q = np.copy(QS)
            global count
            # Step 1: check if first iteration. If so, save residual
            if count == 0: 
                if return_his:
                    Q, res = FVMG.multistage(self,Q,1,P=P,return_his=True)
                    residuals.append(res[0])
                else :
                    Q = FVMG.multistage(self,Q,1,P=P)
                R = FVMG.ResidualVector(self,Q,P=P) # compute residual vector

            else: 
                # if we are not on the first iteration, just call the method
                Q = FVMG.multistage(self,Q,1,P=P)
                R = FVMG.ResidualVector(self,Q,P=P)
                
            # Step 2: restrict r to a coarser grid with N2=(N-1)/2)
            Q2 = FVMG.Restrict(self,Q)
            P2 = FVMG.Restrict(self,R,weighted=True) - FVMG.ResidualVector(self,Q2)

            # Step 3: Solve the problem on the coarse grid
            done = False
            while not done:
                # if next grid is finer, and previous was finer or the same, 
                # call the method and prolong
                if (self.schedule[count+2] == self.schedule[count+1]-1) \
                    and ((self.schedule[count] == self.schedule[count+1]-1) \
                    or (self.schedule[count] == self.schedule[count+1])):
                    count += 1
                    Q2 = FVMG.multistage(self,Q2,1,P=P2)
                    done = True
                # if next grid is fine, and previous was courser, just prolong
                elif (self.schedule[count+2] == self.schedule[count+1]-1) \
                    and (self.schedule[count] == self.schedule[count+1]+1):
                    count += 1
                    done = True
                # if next grid is the same, call the method again
                elif (self.schedule[count+2] == self.schedule[count+1]): 
                    count += 1
                    Q2 = FVMG.multistage(self,Q2,1,P=P2)
                # if next grid is coarser, call a new round of solve
                elif self.schedule[count+2] == self.schedule[count+1]+1:
                    count += 1
                    Q2 = solve(Q2,P2) 
                else:
                    print('ERROR: Multigrid if statement failed')
                    break
            
            # Step 4: Prolong the error back to the fine grid and update solution    
            return Q + FVMG.Prolong(self,Q2- FVMG.Restrict(self,Q))

        while True:
            count = 0
            QS = solve(QS,0)
            if residuals[-1]/residuals[0] < 1e-11:
                break
        
        if return_his: return QS, residuals
        else: return QS

    def schedules(self,MGType):
        cycles = {"W-4": [1,2,3,4,3,4,3,2,3,4,3,4,3,2,1],
                     "W-5": [1,2,3,4,5,4,5,4,3,4,5,4,5,4,3,
                     2,3,4,5,4,5,4,3,4,5,4,5,4,3,2,1],
                     "W-6": [1,2,3,4,5,6,5,6,5,4,5,6,5,6,5,4,
                     3,4,5,6,5,6,5,4,3,2,3,4,5,6,5,6,5,4,
                     3,4,5,6,5,6,5,4,5,6,5,6,5,4,3,2,1],
                     "V-4": [1,2,3,4,3,2,1],
                     "V-5": [1,2,3,4,5,4,3,2,1],
                     "V-6": [1,2,3,4,5,6,5,4,3,2,1]
                     }

        return cycles[MGType]

    def extrapolate1st(self,QS):
        """
        First order Riemann extrapolation
        """
        M = int(len(QS)/3)
        e = int((self.N+1)/(M+1))
        R11 = FVMG.GetRiemannInv(self,QS[0]/self.S[e],QS[M]/self.S[e],QS[2*M]/self.S[e])[0]
        R12 = FVMG.GetRiemannInv(self,QS[1]/self.S[2*e],QS[M+1]/self.S[2*e],QS[2*M+1]/self.S[2*e])[0]
        R2M,R3M = FVMG.GetRiemannInv(self,QS[M-1]/self.S[-1-e],QS[2*M-1]/self.S[-1-e],QS[-1]/self.S[-1-e])[1:]
        R2M2,R3M2 = FVMG.GetRiemannInv(self,QS[M-2]/self.S[-1-2*e],QS[2*M-2]/self.S[-1-2*e],QS[-2]/self.S[-1-2*e])[1:]
        Q1L,Q2L,Q3L = FVMG.RiemanntoQ(self,2*R11-R12,self.R2L,self.R3L)
        Q1R,Q2R,Q3R = FVMG.RiemanntoQ(self,self.R1R,2*R2M-R2M2,2*R3M-R3M2)
        return [Q1L*self.S[0],Q2L*self.S[0],Q3L*self.S[0],Q1R*self.S[-1],Q2R*self.S[-1],Q3R*self.S[-1]]

    def pressure(self,rho,rhou,e):
        return (self.gamma-1)*(e-(rhou**2/2/rho))

    def sound(self,rho,rhou,e):
        return np.sqrt(self.gamma* FVMG.pressure(self,rho,rhou,e)/rho)

    def exact(self,M,S, Sstar):
        return M*S/Sstar - (2*(1+M**2*(self.gamma-1)/2)/(self.gamma+1))**((self.gamma+1)/(2*(self.gamma-1)))

    def subsonic(self,R,T0,p01,Sstar,):
        """
        Exact solution of the subsonic nozzle
        """ 
        eqn = lambda M : FVMG.exact(self,M,self.S,Sstar)
        M = broyden1(eqn,0.5*np.ones(len(self.S)),verbose=0,
                        maxiter=1000,f_tol=1E-14)
        T = T0/(1+M**2*(self.gamma-1)/2)
        p = p01*(1+M**2*(self.gamma-1)/2)**(-self.gamma/(self.gamma-1))
        rho = p/R/T
        a = np.sqrt(self.gamma*R*T)
        u = M*a
        e = p/(self.gamma-1) + rho*u**2/2
        return M , T , p , rho , a , u , e
    
    def transonic(self,xshock,SstarL,T0,p01,R):
        """
        Exact solution of the transonic case
        """
        def exactsol(M,S, Sstar):
            gamma = 1.4
            return M*S/Sstar - (2*(1+M**2*(gamma-1)/2)/(gamma+1))**((gamma+1)/(2*(gamma-1)))
        
        xshocki = np.where((abs(self.x-xshock))==min(abs(self.x-xshock)))[0][0]
        transi = np.where((abs(self.S-SstarL))==min(abs(self.S-SstarL)))[0][0]
        M = np.empty(len(self.S))
        T = np.empty(len(self.S))
        p = np.empty(len(self.S))
        for i in range(transi):
            M[i] = bisect(exactsol,0,1,args=(self.S[i],SstarL),xtol=1e-12,
                        maxiter=1000)
        for i in range(xshocki-transi):
            M[i+transi] = bisect(exactsol,1,2,args=(self.S[i+transi],SstarL),
                                    xtol=1e-12,maxiter=1000)

        T[:xshocki] = T0/(1+M[:xshocki]**2*(self.gamma-1)/2)
        p[:xshocki] = p01*(1+M[:xshocki]**2*(self.gamma-1)/2)**(-self.gamma/(self.gamma-1))
        
        ML = M[xshocki-1]
        T0R = T0
        p0L = p01
        p0R = p0L*(((self.gamma+1)*ML**2/(2+(self.gamma-1)*ML**2))**(self.gamma/(self.gamma-1))) \
            /((2*self.gamma*ML**2/(self.gamma+1) - (self.gamma-1)/(self.gamma+1))**(1/(self.gamma-1)))
        rho01 = p01/R/T0
        rho0R = p0R/R/T0
        a01 = np.sqrt(self.gamma*p01/rho01)
        a0R = np.sqrt(self.gamma*p0R/rho0R)
        rhostarLastarL = rho01*a01*(2/(self.gamma+1))**((self.gamma+1)/2/(self.gamma-1))
        rhostarRastarR = rho0R*a0R*(2/(self.gamma+1))**((self.gamma+1)/2/(self.gamma-1))
        SstarR = SstarL*rhostarLastarL/rhostarRastarR
        for i in range(len(self.S)-xshocki):
            M[i+xshocki] = bisect(exactsol,0,1,args=(self.S[i+xshocki],SstarR),
                                xtol=1e-12,maxiter=1000)
        T[xshocki:] = T0R/(1+M[xshocki:]**2*(self.gamma-1)/2)
        p[xshocki:] = p0R*(1+M[xshocki:]**2*(self.gamma-1)/2)**(-self.gamma/(self.gamma-1))
        rho = p/R/T
        a = np.sqrt(self.gamma*R*T)
        u = M*a
        e = p/(self.gamma-1) + rho*u**2/2
        
        return M , T , p , rho , a , u , e
