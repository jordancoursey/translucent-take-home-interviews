import argparse, pandas as pd, pathlib, json, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = pathlib.Path(__file__).parent / "data" / "denials.csv"

def load_docs():
    df = pd.read_csv(DATA_PATH)
    docs = df.apply(lambda r: f"Department {r['department']} denied because {r['denial_reason']} on {r['service_date']}", axis=1).tolist()
    return docs, df

def answer(question: str) -> str:
    docs, df = load_docs()
    vect = TfidfVectorizer().fit(docs + [question])
    doc_vecs = vect.transform(docs)
    q_vec = vect.transform([question])
    sims = cosine_similarity(q_vec, doc_vecs).flatten()
    top_idx = sims.argsort()[-3:][::-1]
    rows = df.iloc[top_idx]
    reason_counts = rows['denial_reason'].value_counts().to_dict()
    department_counts = rows['department'].value_counts().to_dict()
    answer_parts = [f"{k}: {v}" for k,v in reason_counts.items()] + [f"{k}: {v}" for k,v in department_counts.items()]
    return " | ".join(answer_parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(answer(args.question))
