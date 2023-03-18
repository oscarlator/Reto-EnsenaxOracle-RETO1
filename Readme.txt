Reto Enseña x Oracle - Clasificación¢n im†genes (Reto 1)

El código generado para solucionar el Reto Enseña x Oracle - Clasificación im†genes (Reto 1) es un ejemplo de cómo entrenar una red neuronal para clasificar im†genes en diferentes categorías. Para hacer esto, he utilizado una herramienta llamada PyTorch, que se utiliza para trabajar con redes neuronales.

Primero, se define la estructura de la red neuronal usando un modelo pre-entrenado llamado ResNet18. También se agregan dos capas personalizadas para clasificar las im†genes en 8 categorías diferentes.

Luego, se cargan las im†genes y las etiquetas desde un archivo CSV, utilizando una clase personalizada llamada MyDataset. En esta clase, se definen las transformaciones que se aplican a las im†genes para mejorar la capacidad de generalización¢n de la red neuronal.

después, se entrena la red neuronal utilizando el conjunto de entrenamiento y se ajustan los pesos de la red para minimizar la pérdida. Durante el entrenamiento, se utilizan lotes para mejorar la eficiencia del procesamiento.

Finalmente, se utiliza la red neuronal entrenada para realizar predicciones en el conjunto de prueba y se guardan los resultados en un archivo JSON. Los resultados pueden ser evaluados para ver cu†n precisas son las predicciones de la red neuronal.
