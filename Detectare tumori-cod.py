import matplotlib.pyplot as plt
import numpy as np
import os
cale = r'C:\Users\strat\Downloads'
cale_img = os.path.join(cale, 'img1.jpeg')
img = plt.imread(cale_img)
def extindere(img, L, r):
    s = img.shape
    img_out = np.zeros([s[0], s[1],s[2]], dtype='float')
    img = img.astype('float')
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            img_out[i, j] = (L - 1) * (img[i, j] / (L - 1)) **r
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out
img_out = extindere(img, 255, 2)
plt.figure()
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Imaginea 1 Originala')
plt.subplot(1, 2, 2), plt.imshow(img_out, cmap='gray'), plt.title('Imaginea 1 cu Functia putere')
plt.show()




import matplotlib.pyplot as plt
import numpy as np
import os
cale = r'C:\Users\strat\Downloads'
cale_img = os.path.join(cale, 'img1.jpeg')
img = plt.imread(cale_img)
def exponentiala(img, L):
    s = img.shape
    img_out = np.zeros([s[0], s[1], s[2]], dtype='float')
    img = img.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            img_out[i, j] = L ** ((img[i, j] / (L - 1)) - 1)
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out
img_out = exponentiala(img, 100)
plt.figure()
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Imaginea Originala')
plt.subplot(1, 2, 2), plt.imshow(img_out, cmap='gray'), plt.title('Imaginea 1 Modificata cu functia exponentiala')
plt.show()







import matplotlib.pyplot as plt
import numpy as np
import os
cale = r'C:\Users\strat\Downloads'
cale_img = os.path.join(cale, 'img1.jpeg')
img = plt.imread(cale_img)
def logaritm(img, L):
    s = img.shape
    img_out = np.zeros([s[0], s[1], s[2]], dtype='float')
    img = img.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            img_out[i, j] = ((L - 1) / np.log(L)) * np.log(img[i, j] + 1)
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out
img_out = logaritm(img, 255)
plt.figure()
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Imaginea Originala')
plt.subplot(1, 2, 2), plt.imshow(img_out, cmap='gray'), plt.title('Imaginea1 cu functia log')
plt.show()











import numpy as np
import os
import matplotlib.pyplot as plt
cale= r'C:\Users\strat\Downloads'
cale_img=os.path.join(cale, 'img6.jpeg')
img_in=plt.imread(cale_img)
def binarizare(img_in,a,b,Ta,Tb,L_1):
    img_final=np.empty_like(img_in)
    img_in=img_in.astype(float)
    s=img_in.shape 
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if (img_in[i,j]<a).any():
                img_final[i,j]=Ta/a*img_in[i,j]
            elif (img_in[i,j]>a).any() and (img_in[i,j]<b).any():
                img_final[i,j]=Ta+((Tb-Ta)/(b-a))*(img_in[i,j]-a)
            else:
                img_final[i,j]=Tb+((L_1-Tb)/(L_1-b))*(img_in[i,j]-b)
    img_final=np.clip(img_final,0,255)
    img_final=img_final.astype('uint8')
    return img_final
img_final=binarizare(img_in,150,150,0,0,255)
plt.figure()
plt.subplot(1,2,1),plt.imshow(img_in,cmap='gray'),plt.title('Imaginea Originala')
plt.subplot(1,2,2),plt.imshow(img_final,cmap='gray'),plt.title('Imaginea 6 Binarizata')
plt.show()
print('===')









import numpy as np
import os
import matplotlib.pyplot as plt

import scipy.ndimage as sc 

cale = r'C:\Users\strat\Downloads'
cale_img = os.path.join(cale, 'img6.jpeg')
img_init = plt.imread(cale_img)

def rgb2gri(img_in, format):
    img_in = img_in.astype('float')   
    s = img_in.shape
    if len(s) == 3 and s[2] == 3:
        if format == 'png':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]) * 255
        elif format == 'jpg':
            img_out = 0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out.astype('uint8')
        return img_out
    else:
        print('Conversia nu a putut fi realizata deoarece imaginea de intrare nu este color!')
        return img_in 
    
img_init = rgb2gri(img_init, 'jpg')

def binarizare(img_init, a, b, Ta, Tb, L_1):
    img_final = np.empty_like(img_init)
    img_init = img_init.astype(float)
    s = img_init.shape
    
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if (img_init[i, j] < a).any():
                img_final[i, j] = Ta / a * img_init[i, j]
            elif (img_init[i, j] > a).any() and (img_init[i, j] < b).any():
                img_final[i, j] = Ta + ((Tb - Ta) / (b - a)) * (img_init[i, j] - a)
            else:
                img_final[i, j] = Tb + ((L_1 - Tb) / (L_1 - b)) * (img_init[i, j] - b)
    img_final = np.clip(img_final, 0, 255)
    img_final = img_final.astype('uint8')
    return img_final


v1 = np.array([[1, 1, 1, 1, 1]])
v2 = np.array([[1], [1], [1]])
v3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
v4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
v5 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
v6 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

y = binarizare(img_init, 120, 120, 0, 255, 255)

inchidere1 = sc.binary_closing(y, structure=v1)
inchidere2 = sc.binary_closing(y, structure=v2)
inchidere3 = sc.binary_closing(y, structure=v3)
inchidere4 = sc.binary_closing(y, structure=v4)
inchidere5 = sc.binary_closing(y, structure=v5)
inchidere6 = sc.binary_closing(y, structure=v6)

plt.figure('Total')

plt.subplot(3, 3, 1), plt.imshow(img_init, cmap='gray'), plt.title('imaginea initiala')
plt.subplot(3, 3, 3), plt.imshow(y, cmap='gray'), plt.title('binarizare')
plt.subplot(3, 3, 4), plt.imshow(inchidere1, cmap='gray'), plt.title('inchidere1')
plt.subplot(3, 3, 5), plt.imshow(inchidere2, cmap='gray'), plt.title('inchidere2')
plt.subplot(3, 3, 6), plt.imshow(inchidere3, cmap='gray'), plt.title('inchidere3')
plt.subplot(3, 3, 7), plt.imshow(inchidere4, cmap='gray'), plt.title('inchidere4')
plt.subplot(3, 3, 8), plt.imshow(inchidere5, cmap='gray'), plt.title('inchidere5')
plt.subplot(3, 3, 9), plt.imshow(inchidere6, cmap='gray'), plt.title('inchidere6')
plt.show()








import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage as sc 

cale = r'C:\Users\strat\Downloads'
cale_img = os.path.join(cale, 'img6.jpeg')
img_init = plt.imread(cale_img)

def rgb2gri(img_in, format):
    img_in = img_in.astype('float')   
    s = img_in.shape
    if len(s) == 3 and s[2] == 3:
        if format == 'png':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]) * 255
        elif format == 'jpg':
            img_out = 0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out.astype('uint8')
        return img_out
    else:
        print('Conversia nu a putut fi realizata deoarece imaginea de intrare nu este color!')
        return img_in 
    
img_init = rgb2gri(img_init, 'jpg')

def binarizare(img_init, a, b, Ta, Tb, L_1):
    img_final = np.empty_like(img_init)
    img_init = img_init.astype(float)
    s = img_init.shape
    
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if (img_init[i, j] < a).any():
                img_final[i, j] = Ta / a * img_init[i, j]
            elif (img_init[i, j] > a).any() and (img_init[i, j] < b).any():
                img_final[i, j] = Ta + ((Tb - Ta) / (b - a)) * (img_init[i, j] - a)
            else:
                img_final[i, j] = Tb + ((L_1 - Tb) / (L_1 - b)) * (img_init[i, j] - b)
    img_final = np.clip(img_final, 0, 255)
    img_final = img_final.astype('uint8')
    return img_final


v1 = np.array([[1, 1, 1, 1, 1]])
v2 = np.array([[1], [1], [1]])
v3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
v4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
v5 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
v6 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

y = binarizare(img_init, 150, 150, 0, 0, 255)

deschidere = sc.binary_opening(y, structure=v1)
deschidere2 = sc.binary_opening(y, structure=v2)
deschidere3 = sc.binary_opening(y, structure=v3)
deschidere4 = sc.binary_opening(y, structure=v4)
deschidere5 = sc.binary_opening(y, structure=v5)
deschidere6 = sc.binary_opening(y, structure=v6)

plt.figure('Total')

plt.subplot(3, 3, 1), plt.imshow(img_init, cmap='gray'), plt.title('imaginea initiala')
plt.subplot(3, 3, 2), plt.imshow(y, cmap='gray'), plt.title('binarizare')
plt.subplot(3, 3, 4), plt.imshow(deschidere, cmap='gray'), plt.title('deschidere1')
plt.subplot(3, 3, 5), plt.imshow(deschidere2, cmap='gray'), plt.title('deschidere2')
plt.subplot(3, 3, 6), plt.imshow(deschidere3, cmap='gray'), plt.title('deschidere3')
plt.subplot(3, 3, 7), plt.imshow(deschidere4, cmap='gray'), plt.title('deschidere4')
plt.subplot(3, 3, 8), plt.imshow(deschidere5, cmap='gray'), plt.title('deschidere5')
plt.subplot(3, 3, 9), plt.imshow(deschidere6, cmap='gray'), plt.title('deschidere6')

plt.show()

