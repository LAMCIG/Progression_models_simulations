import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Tuple, Optional, List
import sys


def visualize_meta_analysis(weights, 
                            SE_indiv, #expects normally-distributed effect sizes and their standard errors 
                            Beta_indiv, #expects normally-distributed effect sizes and their standard errors 
                            SE_meta, #expects normally-distributed effect sizes and their standard errors 
                            Beta_meta, #expects normally-distributed effect sizes and their standard errors 
                            study_names,
                            meta_type = 'CohensD', # other options: 'RiskRatio', 'Correlation'
                            figsize: Tuple[float, float] = None,
                            ax: plt.Axes = None)-> plt.Axes:
    
    if meta_type not in ['CohensD', 'RiskRatio', 'Correlation']:
        sys.exit("unknown \'meta_type\'\nKnown values: \'CohensD\', \'RiskRatio\', \'Correlation\'")
    
    if figsize is None:
        figsize = (7,np.int32(np.ceil(0.5*len(SE_indiv))))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    null_value = 0.0
    if(meta_type == 'RiskRatio'):
        null_value = 1.0

    max_wt = np.max(weights)
    min_wt = np.min(weights)

    ratio = np.fmin(max_wt/min_wt,10)
    scale = 2*ratio/(max_wt-min_wt)

    marksize = np.add(8-np.log(ratio),np.multiply(np.subtract(weights,min_wt),scale))

    Min = []
    Max = []
    
    ###Draw individual studies in the forest plot
    for se,beta,x,w in zip(SE_indiv, Beta_indiv, range(len(SE_indiv)), marksize):
        y = -1.0*x

        center = beta
        right = center + 1.96*se
        left = center - 1.96*se

        if(meta_type == 'Correlation'):
            center = corel_expit(center)
            right = corel_expit(right)
            left = corel_expit(left)

        if(meta_type == 'RiskRatio'):
            center = np.exp(center)
            right = np.exp(right)
            left = np.exp(left)

        Min.append(left)
        Max.append(right)

        ax.plot((left,right),(y,y),'-',color='orange')
        ax.plot((center),(y),'s',color='orange',markersize=w)

    ###Draw the meta-analysis result in the forest plot
    y_sum = -1.0*len(SE_indiv)

    center = Beta_meta
    right = Beta_meta + 1.96*SE_meta
    left = Beta_meta - 1.96*SE_meta

    if(meta_type == 'Correlation'):
        center = corel_expit(center)
        right = corel_expit(right)
        left = corel_expit(left)

    if(meta_type == 'RiskRatio'):
        center = np.exp(center)
        right = np.exp(right)
        left = np.exp(left)

    ax.plot((left,right),(y_sum,y_sum),'-',color='blue')
    ax.plot((center),(y_sum),'D',color='blue',markersize=10)

    #Draw the null hypothesis line   
    ax.plot((null_value,null_value),(y_sum-0.5,0.5),'--',color='red')
  
    #Print study names on the plot
    names = study_names.copy()
    names.append('Summary')
    name_ticks = np.append( np.multiply(-1.0,np.array(range(len(SE_indiv)))), [y_sum])
    ax.set_yticks(name_ticks, list(names))

    Max = np.fmax((np.asarray(Max)).max(),null_value)
    Min = np.fmin((np.asarray(Min)).min(),null_value)

    rng = Max - Min

    # plt.xlim([Min - 0.1*rng , Max + 0.1*rng])
    ax.set_xlim(Min - 0.1*rng, Max + 0.1*rng)
    plt.tight_layout()

    return ax

def corel_expit(val):
    return (np.exp(2.0*val) - 1.0)/(np.exp(2.0*val) + 1.0)





def simulate_data(sample_sizes = None,
                  analysis_type = 'CohensD', # other options: 'RiskRatio', 'Correlation'
                  random_effects = True,
                  random_seed = 100) -> Tuple[List, List, List]:
    
    if analysis_type not in ['CohensD', 'RiskRatio', 'Correlation']:
        sys.exit("unknown \'meta_type\'\nKnown values: \'CohensD\', \'RiskRatio\', \'Correlation\'")

    #Simulate several studies assuming a fixed (or random) effects model drawn from a multivariate Gaussian
    mean1_true = [4, 10, 2] #Means of group A
    mean2_true = [5, 6, 2]  #Means of group B

    cov = [[10, 3, 2], 
        [3, 15, 1], 
        [2, 1, 6]]

    Tau = [[0.15, 0, 0], 
        [0, 0.2, 0], 
        [0, 0, 0.2]]
    

    if sample_sizes is None:
        N = [[20,30], [15,17], [18,23], [24,40], [10,33], [67,80]]
    else:
        N = sample_sizes

    DF_all = []
    study = []

    if random_seed is not None:
        np.random.seed(random_seed)
        
    for n_sam, j in zip(N,range(len(N))):
        
        mean1 = mean1_true
        mean2 = mean2_true
        
        if(random_effects):
            dim = len(mean1_true)
            for kk in range(dim):
                mean1[kk] = np.random.normal(mean1_true[kk], Tau[kk][kk], 5*(1+dim))[2*(1+dim)-kk]
                mean2[kk] = np.random.normal(mean2_true[kk], Tau[kk][kk], 5*(1+dim))[2*(1+dim)-kk]
            

        x1, y1, z1 = np.random.multivariate_normal(mean1, cov, n_sam[0]).T
        x2, y2, z2 = np.random.multivariate_normal(mean2, cov, n_sam[1]).T

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
        z = np.concatenate((z1, z2))

        label = []
        for i in range(x1.size):
            label.append("Type A")
        for i in range(x2.size):
            label.append("Type B")

        ##Create a dataframe for the current study & add it to the array of studies
        df = pd.DataFrame(data={'x_var': x, 'y_var': y, 'z_var': z, 'class': label})
        DF_all.append(df)
        
        ##add the study name
        study.append("Study {0:d}".format(j+1))
        
        
    ##########Individual study mean difference computation between Type A & B      
    if (analysis_type == 'CohensD'):
        var = 'x_var'
        beta_all = []
        se_all = []
        for df in DF_all:
            df1 = df[df['class']=='Type A']
            df2 = df[df['class']=='Type B']
            data1 = df1[var]
            data2 = df2[var]
            
            n = []
            n.append(np.size(data1))
            n.append(np.size(data2))
            
            n1 = n[0] - 1
            n2 = n[1] - 1
            
            mean_dif = data1.mean() - data2.mean()
            sd = np.sqrt((data1.var()*n1 + data2.var()*n2)/(n1+n2))
            
            beta = mean_dif/sd
            se = np.sqrt( (n[0]+n[1])/(n[0]*n[1]) + beta*beta/( 2*(n[0]+n[1]) ) )
            
            beta_all.append(beta)
            se_all.append(se)
        

    ##########Individual study correlation Fischer's z for variables x & y 
    if (analysis_type == 'Correlation'):
        var1 = 'x_var'
        var2 = 'y_var'
        beta_all = []
        se_all = []
        for df,n in zip(DF_all,N):
            
            r = df[[var1,var2]].corr().to_numpy()[0,1]
            n_all = n[0] + n[1]
            
            beta = 0.5*np.log((1+r)/(1-r))
            se = 1/np.sqrt(n_all - 3)
            
            beta_all.append(beta)
            se_all.append(se)

    ##########Individual study Log Risk ratio for variable [x > true_mean(group A)] 
    if (analysis_type == 'RiskRatio'):
        
        var1 = 'x_var'
        thresh = mean1_true[0]

        beta_all = []
        se_all = []
        for df,n in zip(DF_all,N):

            A = sum(1*(df[df['class']=='Type A'][var1].to_numpy() > thresh))
            B = sum(1*(df[df['class']=='Type A'][var1].to_numpy() <= thresh))

            C = sum(1*(df[df['class']=='Type B'][var1].to_numpy() > thresh))
            D = sum(1*(df[df['class']=='Type B'][var1].to_numpy() <= thresh))

            RR = (A/n[0])/(C/n[1])
            beta = np.log(RR)
            se = np.sqrt(1.0/A - 1.0/n[0] + 1.0/C - 1.0/n[1])

            beta_all.append(beta)
            se_all.append(se)


    return beta_all, se_all, study