# Speech Recognition
Como trabajo práctico final de la materaia Procesamiento del Habla de la carreta Ingeniería Electrónica en FIUBA (Facultad de Ingeniería de la UBA), se optó la realización de un sistema reconocedor de habla basado en redes neuronales implementado en TensorFlow.
En este trabajo se abordan dos estrategias:
* Se diseña un clasificador de fonemas basado en redes recurrentes, donde se utilizan las transcripciones fonéticas alineadas provistas por TIMIT, para predecir en cada instante de tiempo el fonema expresado en la señal de habla.
* Se diseña un sistema de traducción de habla a texto basado en redes recurrentes, que hace uso de CTC (Connectionist Temporal Classification) para poder entrenar con las transcripciones no alineadas de TIMIT. 

El objetivo de este trabajo no es el de ganarle a Google ni a Siri en el reconocimiento de habla, sino el de interiorizarse en el uso de redes recurrentes y en el desarrollo en TensorFlow. Por otra parte, se busca también poder aplicar los conocimientos adquiridos en la materia a un ejemplo práctico.

## Instalación
Para este trabajo se utilizó el IDE PyCharm. Sin embargo, cualquier computadora con python3.6 y TensorFlow instalados (preferiblemente con soporte de CUDA) puede ejecutarlo.

Algunas de las librerías que se requieren tener instaladas son:
* TensorFlow 
* Numpy
* SMAC3
* python_speech_features
* scipy
* pickle

Se debe tener en cuenta que las rutas a los archivos y carpetas fueron configurados dentro del IDE, por lo que es posible que deban ser modificados para ejecutar el proyecto desde una terminal. Por otra parte, debido a un tema de licencia y a espacio requerido, no subí la base de datos de TIMIT utilizada.

## Ejecución
El clasificador de fonemas se encuentra en la carpeta **phoneme_classificator**. Antes de ejecutar la red, se debe generar la base de datos, lo cual se hace mediante el script **GenerateDatabase** que se encuentra en la carpeta **utils**. 
Una vez generada la base de datos, se puede ejecutar la red mediante el script **RNN_test** que está en la carpeta **NeuralNetwork**. 

Mediante el script **smac_optimization** es posible realizar iteraciones sucesivas de la red, con el objetivo de hallar los mejores hiperparámetros de la red.

En la carpeta **ctc_network** se tiene el traductor de habla a texto basado en CTC. Al igual que en el caso anterior, se debe generar la base de datos mediante **GenerateDatabase**, y la ejecucición y optimización de la red se realizan mediante los scripts **ctc_test** y **ctc_smac_optimization** respectivamente.


# Referencias

---
- Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures: ftp://ftp.idsia.ch/pub/juergen/nn_2005.pdf.
- Referencias Understanding LSTM Networks: http://colah.github.io/posts/2015-08-Understanding-LSTMs/.
- Referencias Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf.
- Referencias TIMIT Acoustic-Phonetic Continuous Speech Corpus: https://catalog.ldc.upenn.edu/ldc93s1.
- Referencias SPHERE Conversion Tools: https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools
- Referencias python_speech_features: https://github.com/jameslyons/python_speech_features.
- Referencias SMAC3: https://github.com/automl/SMAC3.
- Referencias Bidirectional Recurrent Neural Networks: https://pdfs.semanticscholar.org/4b80/89bc9b49f84de43acc2eb8900035f7d492b2.pdf.
- Referencias Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks [Alex Graves]: https://www.cs.toronto.edu/~graves/icml_2006.pdf.
- Referencias Supervised Sequence Labelling with Recurrent Neural Networks: https://www.cs.toronto.edu/~graves/preprint.pdf