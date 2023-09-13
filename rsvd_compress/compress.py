"""
@author: Christopher Torres
"""
# Full working code
# Implementing web application funcionality in the future


import time
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

start = time.process_time()

# Methods
##############################################################################
def gen_err_approx(A,P,p,q,test_r):
    res = []
    counter = 0
    for i in  test_r:    
        P = genP(A, test_r[counter])
        U_test, Sig_test, Vt_test = rsvd_pit(A, P,q)
        Ap_test = U_test*Sig_test@Vt_test
        res.append(error(A,Ap_test))
        counter+= 1
    return res
        
def error(A, Ap):
    """
    

    Parameters
    ----------
    A : Original Matrix input
    Ap : Matrix Approximation of "A"

    Returns
    -------
    err : float

    """
    err = np.linalg.norm(A-Ap, ord ='fro')/np.linalg.norm(A, ord ='fro')
    return err
    
def convert(X):
    gray_im = im.fromarray(X)
    return gray_im

def genP(A,r):
    """

    Parameters
    ----------
    A : Matrix input
    
    r : Target rank of projection matrix

    Returns
    -------
    For A ϵ R^mxn returns value of P ϵ R^mxr  

    """
    return np.random.randn(A.shape[1], r)

def rsvd(A, P):
    """

    Parameters
    ----------
    A : Matrix Input
    
    P : Projection matrix to begin approximation
    
    Returns
    -------
    Ua : U matrix of A matrix approximation
    
    s : Singular Value Matrix of matrix input
    
    v : V matrix of SVD decomposition of matrix input
    """
    B = A @ P
    Q, R = np.linalg.qr(B, mode="reduced")
    Ap = Q.T @ A
    Uap, s, v = np.linalg.svd(Ap, full_matrices = 0)
    Ua = Q @ Uap
    return Ua, s, v

def rsvd_pit(A, P,q):
    """
    
    Parameters
    ----------
    A : Matrix Input
    
    P : Projection matrix to begin approximation
    
    q : Power iterations to perform

    Returns
    -------
    Ua : U matrix of A matrix approximation
    
    s : Singular Value Matrix of matrix input
    
    v : V matrix of SVD decomposition of matrix input
    
    """
    Q = p_it(A, P,q)
    Ap = Q.T @ A
    Uap, s, v = np.linalg.svd(Ap, full_matrices = 0)
    Ua = Q @ Uap
    return Ua, s, v

def p_it(A, P,q):
    B = A @ P
    
    for q in range(q):
        B = A @ (A.T @ B)
    Q, R = np.linalg.qr(B, mode = "reduced")
    return Q

def run_all(A,r,p,q):
    #Number of power iterations, set to 0 if none
    #Generate random P matrix with target rank r+p for oversampling
    #If no oversampling, set p = 0
    P = genP(A,r+p)

    #rSVD implementation 
    #Ua, Sig, Vt = rsvd(A, P, q)

    #rSVD implementation with power iterations
    Ua, Sig, Vt = rsvd_pit(A, P, q)


    Ap = Ua*Sig@Vt
    return Ap, P     
#############################################################################
#End methods

#Image path for URL
path = 'https://images.unsplash.com/photo-1516794840430-8d8c51e7c045?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8M3x8fGVufDB8fHx8&w=1000&q=80'

img = io.imread(path)

#If input is a color image, convert to Grayscale
if img.ndim == None:
    A = im.fromarray(img)

else:
#Convert Image to grayscale matrix
    A = np.mean(img, axis=2)    
#Target rank for approximation
r = 5
#Size to increase target rank by
step_size = 15
test_r = []
photo_album = []
#Oversampling amount for better approximation
p = 5
#Power iterations
q = 2
#Target error
tar = 0.05

check = False
i = 1
#Main method call
Ap,P = run_all(A,r,p,q)

#While target error has not been met, increase sampled space
while check == False:
    test_r.append(r)
    photo_album.append(Ap)
    if error(A,Ap) > tar:
        i += 1
        r += step_size
        Ap,P = run_all(A,r,p,q)       
    else:
        check = True

counter = 0
#Plot Images
for j in photo_album:
    gray_im = convert(photo_album[counter])   
    fig = plt.figure()
    plt.subplots_adjust(wspace = 0.1)
    fig.add_subplot(1,2,1)
    plt.title("Original")
    plt.imshow(convert(A))
    plt.axis("off")
    fig.add_subplot(1,2,2)
    plt.title("Approximation")
    plt.imshow(gray_im)
    plt.axis("off")
    counter += 1
    
#Generate Approximate error plot
results = gen_err_approx(A, P, p,q,test_r)
plt.figure()
plt.title("rSVD Approximation with Variable Rank")
plt.plot(0,1,results, 'ro')
plt.plot(0,1,results, 'r')
plt.xticks(range(len(test_r)), test_r)
plt.ylabel("Error Approximation")
plt.xlabel("Target Rank Size")
plt.show()



end = time.process_time()
runtime = (end-start) 
print("Runtime:", str(runtime), "seconds")