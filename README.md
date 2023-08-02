# baai-general-embedding-large-zh-instruction


Map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification,  clustering, or semantic search.
It also can be used in vector databases for  LLMs.
For more details please refer to our GitHub: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)


## Model List
|              Model              | Language | Description | query instruction for retrieval |
|:-------------------------------|:--------:| :--------:| :--------:|
|  [BAAI/baai-general-embedding-large-en-instruction](https://huggingface.co/BAAI/baai-general-embedding-large-en-instruction) |   English |  rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/baai-general-embedding-large-zh-instruction](https://huggingface.co/BAAI/baai-general-embedding-large-zh-instruction) |   Chinese | rank **1st** in [C-MTEB]() bechmark | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/baai-general-embedding-large-zh](https://huggingface.co/BAAI/baai-general-embedding-large-zh) |   Chinese | rank **2nd** in [C-MTEB]() bechmark | --  |


## Evaluation Results  

- **C-MTEB**:  
We create a benchmark C-MTEB for Chinese text embedding which consists of  31 datasets from 6 tasks. 
More details and evaluation scripts see [evaluation](evaluation/README.md).   
 
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**baai-general-embedding-large-zh-instruction**](https://huggingface.co/BAAI/baai-general-embedding-large-zh-instruction) | 1024 | **63.84** | **71.53** | **53.23** | **78.94** | 72.26 | 62.33 | 48.39 |  
| [baai-general-embedding-large-zh](https://huggingface.co/BAAI/baai-general-embedding-large-zh) | 1024 | 63.62 | 70.55 | 50.98 | 76.77 | **72.49** | **65.63** | **50.01** |   
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 |56.91 | 48.15 | 63.99 | 70.28 | 59.34 | 47.68 |  
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 |54.75 | 48.64 | 64.3 | 71.22 | 59.66 | 48.88 |  
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 40.61 | 69.56 | 67.38 | 54.28 | 45.68 |  
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 39.41 | 66.62 | 65.29 | 49.25 | 44.39 | 
| [text2vec](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 41.71 | 67.41 | 65.18 | 49.45 | 37.66 |  
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 41.98 | 70.86 | 63.42 | 49.16 | 30.02 |  
 


## Usage 

### Sentence-Transformers

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["样例数据-1", "样例数据-2"]
model = SentenceTransformer('BAAI/baai-general-embedding-large-zh-instruction')
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings)
```


### HuggingFace Transformers
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/baai-general-embedding-large-zh-instruction')
model = AutoModel.from_pretrained('BAAI/baai-general-embedding-large-zh-instruction')
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:")
print(sentence_embeddings)
```


### Retrieval Task
For retrieval task, when you use the model whose name ends with `-instruction`
each query should start with a instruction. 
```python
from sentence_transformers import SentenceTransformer
queries = ["手机开不了机怎么办？"]
passages = ["样例段落-1", "样例段落-2"]
instruction = "为这个句子生成表示以用于检索相关文章："
model = SentenceTransformer('BAAI/baai-general-embedding-large-zh-instruction')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```

## Limitations
This model only works for Chinese texts and long texts will be truncated to a maximum of 512 tokens.

