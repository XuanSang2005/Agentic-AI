"""Dense retrieval: sentence-transformers (multilingual-e5 hoặc bge-m3) + FAISS.

IndexFlatIP — chỉ ~111 POI nên brute-force exact, không cần ANN.
Embed một lần lúc khởi động, cache ra file .faiss.
"""

# TODO Ngày 1
