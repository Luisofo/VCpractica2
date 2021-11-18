import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import uuid

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def leerimagenfloat(path):
    imagen = cv2.imread(path, 0)
    imagen = np.float64(imagen/255)
    return imagen


def CalculaTamanioMascara(sigma):
    tamanio = 2 * math.ceil(2.5 * sigma) + 1
    return tamanio


def CalculaSigma(tamanio):
    return (tamanio - 1) / 5


def Gaussiana(x, sigma):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))


def PrimeraDerivada(x, sigma):
    return -x / sigma * sigma * Gaussiana(x, sigma)


def SegundaDerivada(x, sigma):
    return ((-x / sigma * sigma) * PrimeraDerivada(x, sigma)) + ((-1 / sigma * sigma) * Gaussiana(x, sigma))


def CalculaKernelGaussiana1D(tamanio):
    sigma = CalculaSigma(tamanio)
    mascara = np.zeros(tamanio)
    inicio = tamanio // 2

    for i in range(tamanio):
        mascara[i] = Gaussiana(i - inicio, sigma)

    suma = np.sum(mascara)

    for i in range(tamanio):
        mascara[i] = mascara[i] / suma

    return mascara


def AnadirPaddingX(img, orden):
    pixeles_a_anadir = ((orden - 1) // 2)
    img = cv2.copyMakeBorder(img, 0, 0, pixeles_a_anadir, pixeles_a_anadir, cv2.BORDER_REFLECT_101)
    return img


def ConvolucionarK1D(kernel, img):
    kernel = np.float64(kernel)
    img = np.float64(img)

    orden = len(kernel)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_procesada = AnadirPaddingX(img, orden)
    img_final = np.zeros_like(img, dtype=np.float64)
    kernel_x = np.tile(kernel, (img_procesada.shape[0], 1))
    array_aux = np.zeros(shape=(img_procesada.shape[0], orden))

    for i in range(img.shape[1]):
        indices = []
        for j in range(orden):
            indices.append(i + j)
        array_aux = img_procesada[:, indices]
        img_final[:, i] = np.sum(array_aux * kernel_x, axis=1)
    return img_final


def MapearARango(img_u):
    alto = img_u.shape[0]
    ancho = img_u.shape[1]

    max = np.amax(img_u)
    min = np.amin(img_u)

    for i in range(alto):
        for j in range(ancho):
            img_u[i][j] = (img_u[i][j] - min) / (max - min) * (255 - 0) + 0

    return np.uint8(img_u)


def Convolucionar2D(kernel_x, kernel_y, img_inicial):
    img_aux = ConvolucionarK1D(kernel_x, img_inicial)
    img_final = np.transpose(ConvolucionarK1D(kernel_y, np.transpose(img_aux)))
    return img_final


def Subsampling(img, orden):
    img_aux = np.zeros(shape=(img.shape[0], img.shape[1] // orden))
    for i in range(img_aux.shape[1]):
        img_aux[:, i] = img[:, 0 + i * orden]
    img_aux_2 = np.zeros(shape=(img.shape[0] // orden, img.shape[1] // orden))
    for i in range(img_aux_2.shape[0]):
        img_aux_2[i, :] = img_aux[0 + i * orden, :]

    return img_aux_2


def Upsampling(img, orden):
    img_aux = np.zeros(shape=(img.shape[0], img.shape[1]*orden))
    for i in range(img.shape[1]):
        img_aux[:, i*orden] = img[:, i]
        img_aux[:, i*orden+1] = img[:, i]
    img_aux_2 = np.zeros(shape=(img.shape[0]*orden, img.shape[1]*orden))
    for i in range(img_aux.shape[0]):
        img_aux_2[i*orden, :] = img_aux[i, :]
        img_aux_2[i*orden+1, :] = img_aux[i, :]

    return img_aux_2


def CalculaSigmaResultante(sigma_img, sigma_aplicado):
    return np.sqrt(sigma_img ** 2 + sigma_aplicado ** 2)


def CalculaSigmaAAplicar(sigma_incial, sigma_final):
    return np.sqrt(sigma_final ** 2 - sigma_incial ** 2)


def ImagenSemilla(img):
    tamanio = CalculaTamanioMascara(1.519)
    return Convolucionar2D(CalculaKernelGaussiana1D(tamanio), CalculaKernelGaussiana1D(tamanio), img)

def ImagenSemillaSuboctava(img):
    tamanio = CalculaTamanioMascara(0.62)
    return Convolucionar2D(CalculaKernelGaussiana1D(tamanio), CalculaKernelGaussiana1D(tamanio), img)


def Escala(img, s, sigma):
    img_list = []
    img_list.append(np.copy(img))
    k = (2 ** (1 / s))
    # print("Sigma- 0=",sigma,"k=",0,CalculaSigmaAAplicar(sigma,sigma))
    for i in range(1, s + 2, 1):
        k_aplicar = k ** i
        sigma_aplicar = CalculaSigmaAAplicar(sigma, sigma * k_aplicar)
        sigma_resultante = CalculaSigmaResultante(sigma, sigma_aplicar)

        # print("Sigma-",i,"=",sigma_resultante,"k=",k_aplicar,sigma_aplicar)

        img_list.append(Convolucionar2D(CalculaKernelGaussiana1D(CalculaTamanioMascara(sigma_aplicar)),
                                        CalculaKernelGaussiana1D(CalculaTamanioMascara(sigma_aplicar)), img_list[0]))

    return img_list


def EscalaIncremental(img, s, sigma):
    img_list = []
    img_list.append(np.copy(img))
    # print("Sigma- 0=",sigma,"k=",0,CalculaSigmaAAplicar(sigma,sigma))
    sigma_anterior = sigma
    for i in range(1, s + 3, 1):
        k = np.sqrt((2 ** (2 * i / s)) - (2 ** ((2 * (i - 1)) / s)))
        sigma_usar = sigma * k
        sigma_resultante = CalculaSigmaResultante(sigma_anterior, sigma_usar)
        sigma_anterior = sigma_resultante
        # print("Sigma-",i,"=",sigma_resultante,"k=",k,sigma_usar)

        img_list.append(Convolucionar2D(CalculaKernelGaussiana1D(CalculaTamanioMascara(sigma_usar)),
                                        CalculaKernelGaussiana1D(CalculaTamanioMascara(sigma_usar)), img_list[i - 1]))

    return img_list


def CalculaSigmaEscala(escala, n_escalas, sigma_inicial):
    k = (2 ** (1 / n_escalas))
    k_aplicar = k ** escala
    sigma_escala = CalculaSigmaResultante(sigma_inicial,
                                          CalculaSigmaAAplicar(sigma_inicial, sigma_inicial * k_aplicar))
    return sigma_escala


def Octavas(n_octavas, n_escalas, img, sigma_incial):
    semilla = ImagenSemilla(img)
    octavas = []
    for i in range(n_octavas):
        sigma = sigma_incial * (2 ** i)
        escala = EscalaIncremental(semilla, n_escalas, sigma)
        octavas.append(escala)
        semilla = octavas[i][-3]
        semilla = Subsampling(semilla, 2)

    return octavas


def DiferenciaDeGaussianas(octavas):
    DoG = []
    for i in range(len(octavas)):
        DoG_octava = []
        for j in range(len(octavas[i]) - 1):
            DoG_octava.append(np.copy(octavas[i][j] - octavas[i][j + 1]))

        DoG.append(DoG_octava)

    return DoG


def HallarExtremosLocales(DoG, sigma_inicial):
    lista_maximos = []
    for i in range(len(DoG)):
        print("Octava:", i)
        for j in range(1, len(DoG[i]) - 1):
            print("Escala:", j)
            imagen = DoG[i][j]
            sigma = CalculaSigmaEscala(j, 3, sigma_inicial * (2 ** i))
            print(sigma)
            for k in range(1, imagen.shape[0] - 1):
                for l in range(1, imagen.shape[1] - 1):
                    # Comprobamos si es extremo de sus vecinos
                    vecinos = np.copy(imagen[k - 1:k + 2, l - 1:l + 2])
                    valor = imagen[k, l]
                    vecinos[1][1] = vecinos[1][0]

                    if valor < np.min(vecinos):
                        vecinos_abajo = DoG[i][j - 1][k - 1:k + 2, l - 1:l + 2]
                        vecinos_arriba = DoG[i][j + 1][k - 1:k + 2, l - 1:l + 2]
                        min_arriba = np.min(vecinos_arriba)
                        min_abajo = np.min(vecinos_abajo)
                        if valor < min_arriba and valor < min_abajo:
                            lista_maximos.append(
                                np.array([valor, cv2.KeyPoint(l * (2 ** i), k * (2 ** i), sigma * 6 * 2, sigma, i)]))

                    elif valor > np.max(vecinos):
                        vecinos_abajo = DoG[i][j - 1][k - 1:k + 2, l - 1:l + 2]
                        vecinos_arriba = DoG[i][j + 1][k - 1:k + 2, l - 1:l + 2]
                        max_arriba = np.max(vecinos_arriba)
                        max_abajo = np.max(vecinos_abajo)
                        if (valor > max_arriba) and (valor > max_abajo):
                            lista_maximos.append(
                                np.array([valor, cv2.KeyPoint(l * (2 ** i), k * (2 ** i), sigma * 6 * 2, sigma, i)]))

    return lista_maximos


def HallarExtremosLocalesV2(DoG, sigma_inicial):
    lista_maximos = []
    for i in range(len(DoG)):
        for j in range(1, len(DoG[i]) - 1):
            imagen = DoG[i][j]
            sigma = CalculaSigmaEscala(j, 3, sigma_inicial * (2 ** i))
            for k in range(1, imagen.shape[0] - 1):
                for l in range(1, imagen.shape[1] - 1):
                    maximo = False
                    minimo = False

                    vecinos = np.copy(imagen[k - 1:k + 2, l - 1:l + 2])
                    valor = imagen[k, l]

                    if (valor > vecinos[1][0]) and (valor > vecinos[1][2]):
                        if (valor > vecinos[0][0]) and (valor > vecinos[0][1]) and (valor > vecinos[0][2]):
                            if (valor > vecinos[2][0]) and (valor > vecinos[2][1]) and (valor > vecinos[2][2]):
                                maximo = True

                    if not maximo:
                        if (valor < vecinos[1][0]) and (valor < vecinos[1][2]):
                            if (valor < vecinos[0][0]) and (valor < vecinos[0][1]) and (valor < vecinos[0][2]):
                                if (valor < vecinos[2][0]) and (valor < vecinos[2][1]) and (valor < vecinos[2][2]):
                                    minimo = True

                    if maximo:
                        vecinos_abajo = DoG[i][j - 1][k - 1:k + 2, l - 1:l + 2]
                        vecinos_arriba = DoG[i][j + 1][k - 1:k + 2, l - 1:l + 2]
                        max_arriba = np.max(vecinos_arriba)
                        max_abajo = np.max(vecinos_abajo)
                        if (valor > max_arriba) and (valor > max_abajo):
                            lista_maximos.append(
                                np.array([valor, cv2.KeyPoint(l * (2 ** i), k * (2 ** i), 1.6 * (2 ** i), sigma, i)]))

                    if minimo:
                        vecinos_abajo = DoG[i][j - 1][k - 1:k + 2, l - 1:l + 2]
                        vecinos_arriba = DoG[i][j + 1][k - 1:k + 2, l - 1:l + 2]
                        min_arriba = np.min(vecinos_arriba)
                        min_abajo = np.min(vecinos_abajo)
                        if (valor < min_arriba) and (valor < min_abajo):
                            lista_maximos.append(np.array([valor, cv2.KeyPoint(l * (2 ** i), k * (2 ** i), sigma * 6)]))

    return lista_maximos


def DevolverValor(a):
    return a[0]


def ObtenerKeypointsSignificativos(lista_max):
    cien_mejores = np.copy(lista_max)
    cien_mejores_ordenador = cien_mejores[cien_mejores[:, 0].argsort()]
    cien_mejores_ordenador = np.concatenate((cien_mejores_ordenador[0:49,:], cien_mejores_ordenador[-50:,:]))
    keypoints=[]
    for i in range(cien_mejores_ordenador.shape[0]):
        keypoints.append(cien_mejores_ordenador[i][1])

    return keypoints


def DerivaDogX(dog):
    derivada_dog=[]

    for i in range(dog):
        for j in range(dog[i]):
            derivada_dog.append(ConvolucionarK1D([1,0,-1],dog[i][j]))

    return derivada_dog


def DerivaDogY(dog):
    derivada_dog = []

    for i in range(dog):
        for j in range(dog[i]):
            derivada_dog.append(np.transpose(ConvolucionarK1D([1, 0, -1], np.transpose(dog[i][j]))))

    return derivada_dog


def DerivaDogSigma(dog):
    derivada_dog = dog.copy()

    for i in range(dog):
        for j in range(1,len(dog[i])):
            for k in range(dog[i][j].shape[0]):
                for m in range(dog[i][j].shape[0]):
                    derivada_dog[i][j][k][m]=derivada_dog[i][j-1][k][m]-derivada_dog[i][j+1][k][m]

    return derivada_dog


# Debemos calcular para el punto x^ = (x,y,sigma) D(x^)=
def ObtenerKeypointsSignificativos(lista_max,dog_img, sigma_inicial):
    keypoints = []
    # Necesitamos la primera derivada y la segunda derivada del DoG, tanto en x, como en y, como en sigma
    # Primeras derivadas
    derivada_x = DerivaDogX(dog_img)
    derivada_y = DerivaDogY(dog_img)
    derivada_sigma = DerivaDogSigma(dog_img)
    # Segundas derivadas
    segunda_derivada_x = DerivaDogX(derivada_x)
    segunda_derivada_y = DerivaDogX(derivada_y)
    segunda_derivada_sigma = DerivaDogSigma(derivada_sigma)


    for i in range(lista_max):
        valor = lista_max[i][1]+




    return keypoints


def KeyPoints(img, oct, esc, sigma_inicial):
    img_semilla = ImagenSemilla(img)
    print("Creando espacio de escalas...")
    octavas_img = Octavas(oct, esc, img_semilla, sigma_inicial)
    print("Creando diferencia de gaussianas...")
    dog_img = DiferenciaDeGaussianas(octavas_img)
    print("Hallando extremos locales...")
    lista_maximos_img = HallarExtremosLocales(dog_img, sigma_inicial)
    print("Extremos locales hayados:", len(lista_maximos_img), ". Refinando...")
    keypoints_img = ObtenerKeypointsSignificativos(lista_maximos_img)
    # for i in range(len(keypoints_img)):
    #   print("Coord:",keypoints_img[i].pt,"Size calculo:",keypoints_img[i].size,"Size func.",keypoints_img[i].angle)
    img_final = cv2.drawKeypoints(img, keypoints_img, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_final

def ejercicio1():
    perro = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador\perro.jpg", 0)
    plt.figure(uuid.uuid4(), figsize=(10, 10))
    plt.imshow(ImagenSemilla(perro), cmap='gray')
    plt.show()


def ejercicio2():
    gato = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador\gato.jpg", 0)
    gato_EscalaGlobal = Escala(ImagenSemilla(gato), 3, 1.6)
    gato_EscalaIncremental = EscalaIncremental(ImagenSemilla(gato), 3, 1.6)

    fig = plt.figure(uuid.uuid4(), figsize=(20, 20))
    for i in range(len(gato_EscalaGlobal)):
        fig.add_subplot(1, 5, i + 1)
        plt.imshow(gato_EscalaGlobal[i], cmap='gray')
    plt.show()

    fig_2 = plt.figure(uuid.uuid4(), figsize=(20, 20))
    for i in range(len(gato_EscalaIncremental)):
        fig_2.add_subplot(1, 5, i + 1)
        plt.imshow(gato_EscalaIncremental[i], cmap='gray')
    plt.show()


def ejercicio3():
    perro = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador\perro.jpg", 0)
    octavas_perro = Octavas(3, 3, perro)

    for i in range(3):
        fig = plt.figure(uuid.uuid4(), figsize=(30, 30))
        for j in range(3):
            fig.add_subplot(1, 3, j + 1)
            plt.imshow(octavas_perro[i][j + 1], cmap='gray')

    plt.show()


def ejercicio4():
    perro = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador\perro.jpg", 0)
    octavas_perro = Octavas(3, 3, perro)
    Dog_perro = DiferenciaDeGaussianas(octavas_perro)

    for i in range(len(Dog_perro)):
        fig = plt.figure(uuid.uuid4(), figsize=(30, 30))
        for j in range(len(Dog_perro[i])):
            fig.add_subplot(1, len(Dog_perro[i]), j + 1)
            plt.imshow(Dog_perro[i][j], cmap='gray')

    plt.show()


def ejercicio5():
    perro = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador\perro.jpg", 0)
    octavas_perro = Octavas(3, 3, perro)
    Dog_perro = DiferenciaDeGaussianas(octavas_perro)
    maximos = HallarExtremosLocales(Dog_perro)
    maximos_v2 = HallarExtremosLocalesV2(Dog_perro)
    print(len(maximos))
    for i in range(98):
        print(maximos[i][1].pt,maximos[i][1].size,"////",maximos_v2[i][1].pt,maximos_v2[i][1].size)


def ejercicio6():
    perro = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador\perro.jpg", 0)
    octavas_perro = Octavas(3, 3, perro)
    Dog_perro = DiferenciaDeGaussianas(octavas_perro)

    maximosV2 = ObtenerKeypointsSognificativos(HallarExtremosLocalesV2(Dog_perro))
    maximos = ObtenerKeypointsSognificativos(HallarExtremosLocales(Dog_perro))
    for i in range(len(maximosV2)):
        print(maximosV2[i].pt, maximos[i].pt)


def ejercicio7():
    gato = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador\Yosemite1.jpg", 0)
    plt.figure(uuid.uuid4(), figsize=(20, 20))
    plt.imshow(KeyPoints(gato, 3, 3, 1.6))
    plt.show()


def ejercicio8():
    # Cargamos la imagen sigma=0.5
    gato = cv2.imread(r"C:\Users\Usuario\Desktop\Universidad\Informatica\Vision por Computador/Yosemite1.jpg", 0)
    # Creamos la imagen inicial del espacio de escalas, upscaling con delta=1 y computamos y sigma inicial de 0.8
    gato_upsampleado = Upsampling(gato, 2)
    gato_semilla = ImagenSemillaSuboctava(gato_upsampleado)
    # Creamos el espacio de escalas
    octavas_gato = Octavas(3, 4, gato_semilla, 0.8)
    # Creamos las Diferencias de Gaussianas
    dog_gato = DiferenciaDeGaussianas(octavas_gato)
    # Hallamos los extremos locales
    extremos = HallarExtremosLocales(dog_gato, 0.8)
    # Tomamos los significativos
    extremos_significativos = ObtenerKeypointsSognificativos(extremos)
    print(len(extremos_significativos))
    # Los pintamos sobre la imagen
    gato_keypoints = cv2.drawKeypoints(gato, extremos_significativos, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(uuid.uuid4(), figsize=(20, 20))
    plt.imshow(gato_keypoints)
    plt.show()

if __name__ == '__main__':
    # ejercicio1()
    # ejercicio2()
    # ejercicio3()
    # ejercicio4()
    # ejercicio5()
    # ejercicio6()
    ejercicio7()
    # ejercicio8()



