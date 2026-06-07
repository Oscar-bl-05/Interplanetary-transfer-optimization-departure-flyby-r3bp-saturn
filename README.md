# Introducción
El objetivo del trabajo es estudiar numéricamente una transferencia interplanetaria Tierra–Saturno en el marco de la aproximación de tres cuerpos restringida plana. La nave parte de una órbita circular baja alrededor de la Tierra y se propaga en un sistema heliocéntrico inercial, considerando en todo instante la atracción gravitatoria del Sol y de la Tierra. La gravedad de Saturno se desprecia durante la transferencia, de acuerdo con la simplificación propuesta en el guion. El destino no se modela como una llegada física a Saturno, sino como una llegada a una órbita circular heliocéntrica de radio igual a la distancia orbital media de Saturno, compatible con la órbita de sus puntos de Lagrange L4 o L5. Por tanto, la condición de llegada usada en la integración es:
|r(t_fin)|=R_saturno
Una vez alcanzada esta distancia, se calcula la maniobra impulsiva final necesaria para igualar la velocidad de la nave a la velocidad circular heliocéntrica de la órbita objetivo.
Se comparan dos estrategias:
## Caso I
Transferencia directa. Se optimiza el impulso inicial de escape y así como su dirección.
## Caso II
Transferencia con asistencia gravitacional terrestre. El impulso inicial debe ser menor que el del Caso I, pero suficiente para escapar de la Tierra. La nave realiza una trayectoria resonante alrededor del Sol, vuelve a pasar por la esfera de influencia terrestre y usa ese segundo encuentro como flyby antes de llegar a Saturno.
En **to-do.md** aparecen 22 subtareas a cumplir provenientes del guión del trabajo. El fichero se han añadido para facilitar su acceso y se referenciarán en este readme con corchetes.

# Estructura del código
**main.py**
Main contiene: algunos parámetros editables del código (steps, tolerancias…); la función F correspondiente a las ecuaciones dinámicas a resolver[2]; y el código que simula, optimiza y grafica los casos I y II.
**cts.py**
Contiene las constantes y parámetros relevantes.[1]
**IC.py**
Contiene las condiciones iniciales básicas, así como una función para convertir valores arbitrarios de ∆vign y θ arbitrarios en un vector Y0 listo para integrar.[3]
analitical.py
Contiene algunos valores y cálculos analíticos para obtener, entre otras cosas, las estimaciones iniciales de ∆vign y θ.[4,5]
**plotter.py**
Grafica los resultados y crea una animación utilizando la librería de matplotlib. [6,15,17,18,19]
err_check.py
No está dentro de ‘/include’ puesto que no es una dependencia del resto del código. Se creó para poder probar fácilmente diferentes valores de tolerancias y así escoger las adecuadas a utilizar en main.py. [7,16]
**optimizer.py**
Utiliza el método propuesto en [8-14] para hallar el resultado óptimo en ambos casos. Para el caso II prueba con varios valores de resonancia. Además cuenta con funciones que analizan el flyby de la tierra para observar si choca con la tierra; miden la energía antes y después de pasar por la esfera de influencia terrestre… con el fin de asistir en la búsqueda de mejores soluciones, pues frecuentemente haciendo barridos ‘sin asistencia’ el optimizador no encontraba soluciones válidas.
**test_case_II.py**
Al igual que err_check.py, no está dentro de ‘/include’ puesto que no es una dependencia del resto del código. Se creó para ayudar a debuggear el optimizador del caso II en su fase de desarrollo.
