# Architecture

## Retrieval

### TD-IDF/BM25
Pros:
* More deterministic and explainable than embeddings

Cons:
* Vocabulary bound. no notion of semantics. cardiology does not get selected if the query has heart in it.


### Embedding Approach

1. Embed each document/row during pre processing
2. Embed the query at run time
3. Retrieve a variable number of documents. The threshold isn't a hard set rule rather set by a threshold for similarity

Pros:
* Captures semantics so heart and cardiology can be recognized as being close together

Cons:
* likely overkill for this specific application.

### Cutoff — relative threshold vs fixed top-k
pros: 
* we don't know ahead of time how many documents will be relevant. this makes it dynamic

cons:
* requires hand tuning/labeling that adds work/maintenance

## Augmentation/Generation
Feed all of the relevant rows into a model + the query

Pros: 
* Can rely on a very smart model to figure out queries
* More agnostic to different types of queries

cons:
* Could feed in a lot of junk and also won't necessarily guarantee precise calculations


## Evaluation


### Precision vs Recall
The current eval isn't set up to punish over retrieiving (no precision metric.) You could naively "cheat" and produce a 100% recall score but flood the context layer with irrelevant context (this is exemplified in evaluation/cheated_eval.py)


### Core Components get their own eval
As we they are two disjointed steps in our pipeline I've added an evaluation script for retrieval
though this is more cumbersome it's useful to have for two reasons:
1. It's strictly optional. You don't have to take it into consideration when deploying to Production as the 
user never actually sees the retrieved documents in the current set up
2. It enables us to improve a bad retrieval. Bad retrieval could cost more money or amke if we over retrieve and a good model might just be able to cover that up. It could also make a good model produce bad results by giving erroneous context. The big picture is this limits our visibility into what is actually going on with performance and where failures happen.


## Observability

We log the following for observability/traceability into the model
* User queries
* The retrieved documents (rows) by their ID
* The response
* Corresponding model name+version for response generator and retrieval model
  (`retrieval_model`, `generator_model`, `pipeline_version`) plus the
  `threshold` used, so any answer can be reproduced from its trace alone

Written to `observability/traces/traces.csv`, one row per call, append-only.
Adding a column would misalign earlier rows against the header, so a schema
change rotates the old log to `traces.<timestamp>.csv` rather than rewriting
or discarding it.


## Next Steps

1. Add a sql generation step
2. the evals all ignore "status" which would be incredibly important
3. Tighten the labels for relevance