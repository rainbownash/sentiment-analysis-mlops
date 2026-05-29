import numpy as np

def review_vector(tokens, model):
  # model = el modelo Word2Vec completo
  # model.wv = donde viven los vectores aprendidos (vocabulario)
  # model.wv["palabra"] = el embedding de esa palabra
  # t in model.wv = comprobar si la palabra existe en el vocabulario -> si hay palabras que aparecen en test que no estaban en train, el código rompería
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0) # promedio por columna (axis=0)