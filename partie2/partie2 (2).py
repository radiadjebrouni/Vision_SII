import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *

def loadImages():
    images = []
    # with open("/home/radia/Desktop/M2SII/VISON/TP/objet1PNG_SII_VISION/filenames.txt", "r") as a_file:
    with open("E:\Documents\MEGAsync\Mix code\TPM2\TPVision\Projet P2\\filenames.txt", "r") as a_file:
        for imgName in a_file:
            # lire l'image en uint16 et la convertir en float32
            # img = cv2.imread("/home/radia/Desktop/M2SII/VISON/TP/objet1PNG_SII_VISION/"+imgName.strip(), cv2.IMREAD_UNCHANGED).astype(np.float32)
            img = cv2.imread("E:\Documents\MEGAsync\Mix code\TPM2\TPVision\Projet P2\\"+imgName.strip(), cv2.IMREAD_UNCHANGED).astype(np.float32)
            # Changer l'intervalle de [0 2^16-1] vers [0 1]
            img /= 65535.0
            images.append(img)

    matrice=[]

    #Diviser chaque pixel sur l’intensité de la source (B/intB, G/intG, R/intR) 
    #Convertir les images en niveau de gris (NVG = 0.3 * R + 0.59 * G + 0.11 * B) .
    intensities = load_intensSources()
    images = np.array(images)
    j=0
    for image in images:    
        print(j)
        for pixels in image:
            i=0
            for p in pixels:
                p[0] = p[0]/float(intensities[i][2])
                p[1] = p[1]/float(intensities[i][1])
                p[2] = p[2]/float(intensities[i][0])
            i+=1
        j+=1
        #Redimensionner l’image telle que chaque image est représentée dans une seule ligne
        #ravel /flatten /reshape 
        image=(image[:,:,0]*0.3+image[:,:,1]*0.59 +image[:,:,2]*0.11)
        image= image.flatten()

        #Ajouter les images dans un tableau (pour former une matrice de N lignes et (h*w)*
        matrice.append(image)
    
    return matrice


def load_lightSources():
    lights = []
    # with open("/home/radia/Desktop/M2SII/VISON/TP/objet1PNG_SII_VISION/light_directions.txt", "r") as a_file:
    with open("E:\Documents\MEGAsync\Mix code\TPM2\TPVision\Projet P2\light_directions.txt", "r") as a_file:
        for light in a_file:
            l=[]
            for j in light.strip().split(' '):
                l.append(float(j))
            lights.append(l)
    
    return lights


def load_intensSources():
    lights = []
    # with open("/home/radia/Desktop/M2SII/VISON/TP/objet1PNG_SII_VISION/light_intensities.txt", "r") as a_file:
    with open("E:\Documents\MEGAsync\Mix code\TPM2\TPVision\Projet P2\light_intensities.txt", "r") as a_file:
        for light in a_file:
            lights.append(light.strip().split(' '))
    return lights


def load_objMask():
    #retourne une matrice (image) binaire tel que 1 :représente un pixel de l’objet et 0 : un pixel du fond.
    
    # img = cv2.imread("/home/radia/Desktop/M2SII/VISON/TP/objet1PNG_SII_VISION/mask.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("E:\Documents\MEGAsync\Mix code\TPM2\TPVision\Projet P2\mask.png", cv2.IMREAD_GRAYSCALE)
    #normalisation de limage
    cv2.threshold(img,0,1,  cv2.THRESH_BINARY, img)
    
    # img = np.array(img)
    return img


def calcul_needle_map():
    """calculer le vecteur normal pour chaque pixel"""
    obj_images = loadImages()
    light_sources = load_lightSources()
    mask = load_objMask()

    #inverser la matrice S
    light_source_inv= np.linalg.pinv(light_sources)

    # N =S^-1 * E
    n= ((np.dot(light_source_inv,obj_images)))
    
    h,w =mask.shape
    img=np.zeros((h,w,3),np.float32) 
    img[:,:,0]= n[0].reshape(-1, w)
    img[:,:,1]= n[1].reshape(-1, w) 
    img[:,:,2]= n[2].reshape(-1, w)

    for i in range(h):
        for j in range(w):
            if (mask[i,j]==1):
                s=sqrt((img[i,j,0]**2+img[i,j,1]**2 +img[i,j,2]**2))
                if not s==0.0:
                    img[i,j,:]=((img[i,j,:])/s +1.)/2.

    return img


def angle_between(v1, v2):
    """Angle entre deux vecteurs dans l'epace 3D"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def y_rotation(vector,theta):
    """Rotationner le vecteur sur l'axe y dans l'espace 3D"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)


def calcul_tan(normals):
    """Cette fonction calcule les tangeantes à partir du vecteur des normaux"""
    tangeantes = normals
    for i in range (len(normals[0])):
        x = normals[0][i]
        y = normals[1][i]
        z = normals[2][i]
        N = np.array([x, y, z])
        t = y_rotation(N, np.pi/2.0)
        tangeantes[0][i]= t[0]
        tangeantes[1][i]= t[1]
        tangeantes[2][i]= t[2]
    return tangeantes


def calcul_angle(T):
    """Cette fonction calcule l'angle entre un vecteur T et le vecteur X [1, 0, 0]"""
    X = np.array([1, 0, 0])
    A = []
    for i in range (len(T[0])):
        x = T[0][i]
        y = T[1][i]
        z = T[2][i]
        N = np.array([x, y, z])
        A.append(angle_between(N, X))
    return A


def show3D(Z):
    """Cette fonction affiche les profondeurs sur un espace 3D"""
    mask = load_objMask()
    h,w = mask.shape
    # afficher l'environnement 3D
    fig = plt.figure() #figsize= [12,8]

    ax = fig.gca(projection='3d')

    p=0;
    X=[]
    Y=[]
    Z = Z.flatten()
    # zz=np.zeros((h,w), np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            if (mask[i,j]==1):
                X.append(i)
                Y.append(j)
                # zz[i][j]=Z[p]
                # ax.scatter(i, j, Z[p], marker='o', color='black')
                p+=1
    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

    # X, Y = np.mgrid[0:512:512, 0:612:612]
    # Z = np.tile(Z, (len(X), 1))
    # ax.plot_surface(X, Y, zz, cmap='viridis', edgecolor='none')

    # X, Y = np.meshgrid(X, Y)
    # zs = Z
    # Z = zs.reshape(X.shape)
    # ax.plot_surface(X, Y, Z)
    plt.show()


def z_from_pq(img):
    p=np.zeros((512,612),np.float32) 
    q=np.zeros((512,612),np.float32)
    z=np.zeros((512,612),np.float32)  

    q[:, :]=-img[:,:,1]/img[:,:,2]
    p[: ,:]=-img[:,:,0]/img[:,:,2]

    z[0,0]=0
    #cimpute depth for the first column
    for y in range(1,w-1):
        z[0,y]=z[0,y-1] - q[0,y]


    #cimpute depth for each row
    for x in range(1,h-1):
        for y in range(1,w-1):
            z[x,y]=z[x-1,y] - p[x,y]

    # z=((z+1. )/2.)*255.
    return z


def dephtFromAngles(i):
    T = calcul_tan(i)
    A = calcul_angle(T)

    Z = []
    for o in A:
        Z.append(tan(o))

    return Z


if __name__ == "__main__":
    h,w = load_objMask().shape
    i = calcul_needle_map()
    # img = cv2.imread("E:\Documents\MEGAsync\Mix code\TPM2\TPVision\Projet P2\\normal.png", cv2.IMREAD_UNCHANGED)

    # Méthode 1
    # Z = dephtFromAngles(i)
    
    # Méthode 2
    Z = z_from_pq(i)



    
    cv2.imshow('Normal Image',i)
    show3D(Z)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   

 