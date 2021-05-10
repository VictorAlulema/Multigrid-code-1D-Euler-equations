from solver import FVMG


Solution = FVMG('subsonic','W-4',7,223,0.5,10)

M_ex = Solution.M_ex
M_num = Solution.Mach
p_ex = Solution.p_ex
p_num = Solution.pres


X = [Solution.x,Solution.x]
Y = [M_ex,M_num]
labels = ['exact','numerical']
colors = ['b','r']
lines = ['-','+']
axis = ['x[m]','Mach number']
Solution.plotting(X,Y,axis,labels,colors,lines, 'Subsonic problem',
                 logscale = False,location = 'lower center',lw= 0.75)

Y = [p_ex,p_num]
axis = ['x[m]','Pressure']
Solution.plotting(X,Y,axis,labels,colors,lines, 'Subsonic problem',
                 logscale = False,location = 'upper center',lw= 0.75)


##"""
##Subsonic channel flow
##First run: Replicate figure 5.18 book
##"""
### Multistep solution with Courant = 3 (here the multigrid case is also run but it is omitted)
##case1 = FVMG('subsonic','W-4',3,103,0.6,600) 
### Multistep solution with Courant = 7 (here the multigrid case is also run but it is omitted)    
##case2 = FVMG('subsonic','W-4',7,103,0.6,420)
### Multigrid solution with Courant = 7 (here the multistep case is also run but it is omitted)
##case3 = FVMG('subsonic','W-4',7,103,0.6,100)
##res1 = case1.residuals2
##res2 = case2.residuals2
##res3 = case3.residuals2
##X = [case1.x,case2.x,case3.x]
##Y = [res1,res2,res3]
##labels = ['Multistep - C = 3','Multistep - C = 7', 'Multigrid W-cycle \n 4 grid, C = 7']
##colors = ['b','r','g']
##lines = ['-','-','-']
##axis = ['Iterations','L2 residual']
##case1.plotting(X,Y,axis,labels,colors,lines, 'Subsonic: beta = 0.6, N = 103',
##                 logscale = True,location = 'upper right',lw= 1)
##
##
##"""
##Subsonic channel flow
##Second run: Effect of Multigrid cycle
##Courant number fixed: Courant = 4
##Number of cells fixed: N = 223
##"""
##
### Multigrid solution W-cycle 2-3-4
##case1 = FVMG('subsonic','W-4',4,223,0.6,10) 
### Multigrid solution W-cycle 3-4-5  
##case2 = FVMG('subsonic','W-5',4,223,0.6,10)
### Multigrid solution W-cycle 4-5-6
##case3 = FVMG('subsonic','W-6',4,223,0.6,10)
### Multigrid solution V-cycle 4 grid
##case4 = FVMG('subsonic','V-4',4,223,0.6,10) 
### Multigrid solution V-cycle 5 grid 
##case5 = FVMG('subsonic','V-5',4,223,0.6,10)
### Multigrid solution V-cycle 6 grid
##case6 = FVMG('subsonic','V-6',4,223,0.6,10)
##res1 = case1.residuals2
##res2 = case2.residuals2
##res3 = case3.residuals2
##res4 = case4.residuals2
##res5 = case5.residuals2
##res6 = case6.residuals2
##X = [case1.x,case2.x,case3.x,case4.x,case5.x,case6.x]
##Y = [res1,res2,res3,res4,res5,res6]
##labels = ['W-cycle 2-3-4','W-cycle 3-4-5', 'W-cycle 4-5-6',
##          'V-cycle 4 grid','V-cycle 5 grid', 'V-cycle 6 grid']
##colors = ['b','r','g','b','r','g']
##lines = ['-','-','-',':',':',':']
##axis = ['Iterations','L2 residual']
##case1.plotting(X,Y,axis,labels,colors,lines, 'Subsonic Multigrid W-cycle VS V-cycle \n C = 4, beta = 0.6, N = 223',
##                 logscale = True,location = 'upper right',lw= 0.75)
##
##
##"""
##Subsonic channel flow
##Third run: Effect of Courant number
##Number of cells fixed, N = 223
##Multigrid cycles fixed: W-cycle 2-3-4 & 3-4-5
##
##"""
##case1 = FVMG('subsonic','W-4',1,223,0.6,10)
##case2 = FVMG('subsonic','W-4',4,223,0.6,10)
##case3 = FVMG('subsonic','W-4',7,223,0.6,10)
##case4 = FVMG('subsonic','W-5',1,223,0.6,10)
##case5 = FVMG('subsonic','W-5',4,223,0.6,10)
##case6 = FVMG('subsonic','W-5',7,223,0.6,10)
##res1 = case1.residuals2
##res2 = case2.residuals2
##res3 = case3.residuals2
##res4 = case4.residuals2
##res5 = case5.residuals2
##res6 = case6.residuals2
##X = [case1.x,case2.x,case3.x,case4.x,case5.x,case6.x]
##Y = [res1,res2,res3,res4,res5,res6]
##labels = ['Cycle 1, C = 1 ','Cycle 1, C = 4', 'Cycle 1, C = 7',
##          'Cycle 2, C = 1 ','Cycle 2, C = 4', 'Cycle 2, C = 7']
##colors = ['b','r','g','b','r','g']
##lines = ['-','-','-',':',':',':']
##axis = ['Iterations','L2 residual']
##case1.plotting(X,Y,axis,labels,colors,lines, 'Subsonic Multigrid beta = 0.6 \n N = 223 Cycle 1: W-cycle 2-3-4 \n Cycle 2: W-cycle 3-4-5',
##                 logscale = True,location = 'upper right',lw= 0.75)



##"""

##Subsonic channel flow
##Fourth run: Effect of smoothing factor Beta
##Number of cells fixed: N = 223
##Multigrid cycle fixed: W-cycle 3-4-5
##Courant fixed: C = 7
##"""
##case1 = FVMG('subsonic','W-4',7,223,0.5,10)
##case2 = FVMG('subsonic','W-4',7,223,0.6,10)
##case3 = FVMG('subsonic','W-4',7,223,0.7,10)
##res1 = case1.residuals2
##res2 = case2.residuals2
##res3 = case3.residuals2
##X = [case1.x,case2.x,case3.x]
##Y = [res1,res2,res3]
##labels = ['Beta = 0.5','Beta = 0.6', 'Beta = 0.7']
##colors = ['b','r','g']
##lines = ['-','-','-',':',':',':']
##axis = ['Iterations','L2 residual']
##case1.plotting(X,Y,axis,labels,colors,lines, 'Subsonic Multigrid N = 223 \n W-cycle 3-4-5, C = 7',
##                 logscale = True,location = 'upper right',lw= 0.75)



