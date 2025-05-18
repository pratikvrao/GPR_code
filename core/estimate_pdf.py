import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def estimate_pdf(y1, y2, pts=2000, plot_fig=True, label1='distribution 1', label2= 'distribution 2'):

    """This function is used after the data (X,y) is scaled. For more information, check preprocess.py"""
    
    # Disecretise test space
    test = np.linspace(-5,5,pts)[:,np.newaxis]
 

    #KDE for first distribution
    kde1 = KernelDensity(kernel="gaussian", bandwidth="scott")
    kde1.fit(y1.reshape(-1,1))
    log_dens1 = kde1.score_samples(test)
    p_y1 = np.exp(log_dens1)


    #KDE for second distribution
    kde2 = KernelDensity(kernel="gaussian", bandwidth="scott")
    kde2.fit(y2.reshape(-1,1))
    log_dens2 = kde2.score_samples(test)
    p_y2 = np.exp(log_dens2)

    if plot_fig:
       fig = plt.figure()
       plt.title('Kernel densities of the two distributions')
       plt.plot(test, p_y1, color='blue', label=label1)
       plt.plot(test, p_y2, color='green', label=label2)
       plt.xlabel(r'$y$')
       plt.ylabel('Kernel density estimate')
       plt.legend(fontsize=14)
       plt.show()

    return p_y1, p_y2, test
