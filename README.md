# Country_guesser

Ce projet a pour but d'implémenter et d'utiliser un perceptron multi-couches (PMC) afin de classifier 3 drapeaux nationaux.

Savez-vous différencier les 3 drapeaux suivants ? Que la réponse soit positive ou non, ce projet va remédier à cela. :ok_hand:

| Jordanie                                                                                                      | Palestine                                                                                                     | Soudan                                                                                                         |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| <img width="247" alt="1" src="https://user-images.githubusercontent.com/53021621/132264644-4c74c1d3-26f8-4120-b45a-b4efa4339e69.png"> | <img width="247" alt="2" src="https://user-images.githubusercontent.com/53021621/132264645-b3af7218-4e5f-49d1-8fd6-c74fbf2c7a44.png"> | <img width="245" alt="3" src="https://user-images.githubusercontent.com/53021621/132264966-884dec4c-7bc0-44ad-a8be-71ce9b89d00d.png">

## Installation

```
1. git clone https://github.com/0ENZO/Country_guesser/
2. pip install -r requirements.txt
```

### Ngrok

```
streamlit run "C:/Users/../Country_guesser/python/app.py" --server.port 80
```

### Librairie

Le perceptron a été implémenté en tant que librairie C++

### Dataset

Le dataset est constitué d'environ 250 images par drapeau, aux alentours de 200 pour l'entraînement et de 40 pour le test. 

### Modèles

Parmi les différentes consignes strictes, l'une d'entre une était d'entraîner jusqu'à convergence les 4 modèles d'architecture suivante : 
* PMC sans couche cachée
* PMC avec une couche cachée de 8 neurones
* PMC avec une couche cachée de 32 neurones
* PMC avec deux couches cachées de 32 neurones

Un modèle par architecture a été retenu, ces 4 modèles entraînés ont entre 81% et 83% d'accuracy sur le dataset de test.

### Application

Une petite web app streamlit a été développé et permet d'upload une image, de choisir parmi un des 4 modèles entrainés cités précedement et de finalement prédire à quel pays appartient le drapeau.
Pour déployer l'app en local et en ligne, ngrok a été utilisé

<img width="960" alt="app 2" src="https://user-images.githubusercontent.com/53021621/132266214-8b8d20de-e0bb-4071-a99f-b3d3f3029ada.PNG">

### Languages utilisés 

* Python
* C++
