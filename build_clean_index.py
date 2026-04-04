"""
Build Index from data.json
Run: py build_clean_index.py
"""
import json, pickle, os
import numpy as np
from rag_system import load_json_data, prepare_text


def build_from_json(json_paths=('data.json', 'data2.json'), index_dir='./index'):
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = []
    for json_path in json_paths:
        if not os.path.exists(json_path):
            continue
        print(f'Loading {json_path}...')
        data.extend(load_json_data(json_path))

    if not data:
        raise FileNotFoundError('No data files found. Expected data.json and/or data2.json')

    texts = [prepare_text(d) for d in data]

    print('Building TF-IDF index...')
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=10000, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, f'{index_dir}/faiss.index')
    with open(f'{index_dir}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{index_dir}/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'Index built: {len(data)} entries, {matrix.shape[1]} dims')
    return index, vectorizer, data


if __name__ == '__main__':
    index, vectorizer, data = build_from_json()

    # Quick test
    from rag_system import retrieve
    print('\n=== Test ===')
    tests = [
        'كم ساعة معتمدة للتخرج',
        'شروط مرتبة الشرف',
        'مواد المستوى الأول الفصل الأول',
        'مواد المستوى الثاني الفصل الثاني',
        'درجة النجاح في المقرر',
        'متى يفصل الطالب',
        'الحد الأقصى لساعات التسجيل',
        'شروط التدريب الميداني',
    ]
    for q in tests:
        r = retrieve(q, index, vectorizer, data, top_k=1)
        if r:
            print(f'  [{r[0]["score"]:.3f}] {q}')
            print(f'         -> {r[0]["title"]}')
