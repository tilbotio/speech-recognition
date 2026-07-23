import os

from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer, SimilarityFunction
from typing import Annotated
from urllib.parse import unquote

modelname = 'whisper-large-v3'
language = 'auto'

if 'MODELNAME' in os.environ:
    modelname = os.environ['MODELNAME']

if 'LANGUAGE' in os.environ:
    language = os.environ['LANGUAGE']

model = SentenceTransformer(modelname,
                            similarity_fn_name=SimilarityFunction.COSINE)

app = FastAPI()


@app.get("/")
async def get_intent(user_input: str, intent_options: Annotated[list[str] | None, Query()] = None):
    sentences1 = [unquote(user_input)]

    intent_options_decoded = []
    for io in intent_options:
        if unquote(io) != "candidates":
            intent_options_decoded.append(unquote(io))

    sentences2 = intent_options_decoded

    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    similarities = model.similarity(embeddings1, embeddings2)
    closest = [None, 0.0]
    secondclosest = [None, 0.0]
    thirdclosest = [None, 0.0]

    for idx, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <30}: {similarities[0][idx]:.4f}")
        if similarities[0][idx] > closest[1]:
            thirdclosest = secondclosest
            secondclosest = closest
            closest = [sentence2, similarities[0][idx]]
        elif similarities[0][idx] > secondclosest[1]:
            secondclosest = [sentence2, similarities[0][idx]]
        elif similarities[0][idx] > thirdclosest[1]:
            thirdclosest = [sentence2, similarities[0][idx]]

    if closest[1] > 0.4:
        if closest[1] - secondclosest[1] < 0.05:
            if secondclosest[1] - thirdclosest[1] < 0.05:
                return {"intent": "candidates", "connector_label": [closest[0], secondclosest[0], thirdclosest[0]]}
            return {"intent": "candidates", "connector_label": [closest[0], secondclosest[0]]}

        return {"intent": closest[0], "connector_label": closest[0]}
    else:
        return None

# from sentence_transformers import SentenceTransformer, SimilarityFunction
# sentences1 = ["Draagt de persoon een corrigerend ding gezichts?"]
# sentences2 = ["bril", "ogen", "haar", "leeftijd"]

# # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
# #                             similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
# model = SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers',
#                             similarity_fn_name=SimilarityFunction.COSINE)
# embeddings1 = model.encode(sentences1)
# embeddings2 = model.encode(sentences2)

# similarities = model.similarity(embeddings1, embeddings2)

# for idx_i, sentence1 in enumerate(sentences1):
#     print(sentence1)
#     for idx_j, sentence2 in enumerate(sentences2):
#         print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
