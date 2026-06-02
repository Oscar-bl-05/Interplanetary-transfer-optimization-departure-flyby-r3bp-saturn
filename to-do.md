# TO-DO | Segundo ejercicio numérico MAO

1. Documentarse.

2. Escribir las ecuaciones dinámicas:

       Y_dot = F(t, Y)

3. Implementar las ecuaciones dinámicas y las condiciones iniciales como funciones en un código de Python para su solución numérica.

4. Usar las estimaciones analíticas del tiempo de transferencia en cada caso para elegir la duración de las simulaciones.

5. Probar el código solucionando las ecuaciones dinámicas en dos ejemplos no optimizados, uno del caso I y otro del II, eligiendo como condiciones iniciales las estimaciones analíticas de la sección B.5. En el caso II, elegir un valor intermedio de velocidad asintótica, por ejemplo:

       v_inf = 0.8 * delta_v_H

6. Representar las soluciones del punto anterior, obteniendo las gráficas de las variables dinámicas, posición y velocidad con respecto al Sol, y de sus errores.

7. Comprobar que los errores numéricos globales cuando se alcanza la distancia R_B sean aceptables.

   Si hay margen, variar las tolerancias para encontrar valores mayores que sigan dando errores aceptables, por ejemplo:

       error de longitud < 1000 km
       error de velocidad < 10 m/s

   Elegir así tolerancias que sean suficientes para tener errores aceptables, pero que aceleren los cálculos para los bucles de optimización.

8. En cada caso considerado, empezando con el caso I, definir un array de 10 valores de theta y un array de 10 valores de delta_v_ign, teniendo en cuenta las estimaciones de la sección B.5.1.

9. Solucionar numéricamente las ecuaciones dinámicas para cada pareja de valores de theta y delta_v_ign, usando un doble bucle y las tolerancias más laxas compatibles con los errores encontrados en el punto 7.

10. Para cada una de esas soluciones, determinar t_fin, definido como el primer tiempo en que r alcanza la distancia R_B, dentro de un margen de error.

    Si no se alcanza esa distancia, se descarta el par de valores de theta y delta_v_ign.

11. Usar los valores de delta_v_ign y de v(t_fin) para calcular el delta_v_tot correspondiente a cada par de valores de theta y delta_v_ign.

12. Elegir los valores de theta y delta_v_ign que den lugar al menor gasto en delta-v total.

13. Repetir los pasos del 8 al 12 incluidos con arrays refinados alrededor de los valores de theta y delta_v_ign obtenidos en el paso 12.

    Se obtiene así un cálculo más preciso de los valores óptimos de theta y delta_v_ign.

    Si es necesario, repetir de nuevo el refinamiento hasta obtener resultados con la precisión deseada.

14. Repetir los pasos del 8 al 13 para el caso II, cambiando oportunamente los arrays de condiciones iniciales y el tiempo de simulación.

15. Para las soluciones óptimas de los dos casos I y II obtenidas, representar las gráficas de la evolución temporal de las variables dinámicas, posición y velocidad con respecto al Sol, y de sus errores.

16. Comprobar que los errores numéricos sean aceptables.

17. Para las soluciones óptimas de los dos casos I y II obtenidas, incluir también:

    - gráficas de r(t), distancia al Sol;
    - gráficas de |r(t) - R(t)|, distancia al planeta A;
    - gráficas en dos dimensiones de la trayectoria vista desde el Sol, poniendo x, y en los ejes;
    - gráficas en dos dimensiones de la trayectoria vista desde el centro del planeta A, poniendo x - X, y - Y en los ejes, donde X, Y son las componentes del vector R(t).

18. En los dos casos óptimos, representar la evolución de los elementos orbitales calculados respecto al Sol y los relativos al planeta A.

19. Realizar una animación de la trayectoria.

20. Discutir los resultados de los puntos anteriores.

    En particular, comparar el consumo en delta-v total de los casos I y II optimizados, e interpretar el resultado.

21. Redactar las conclusiones.

22. Adjuntar los códigos y las figuras a la memoria.

    El nombre de los ficheros de cada grupo contendrá el nombre de los planetas A-B.

    En la primera página de la memoria aparecerán los nombres de los integrantes del grupo.

---

## Estado

1. check
2. check
3. check
4. check
5. check
6. check
7. check (revisar caso II)
8. check
9. check
10. check
11. check
12. check
13. check (n_refines = 2)
14. check (n_refines = 0 para debuggear, cambiar)
15. check
16. check
17. falta centrada en la Tierra
18. check
19. check
20. pendiente
21. pendiente
22. pendiente

---