# -*- coding: utf-8 -*-
"""
Business Entity Resolution Pipeline
=====================================
Scalable pipeline for ~2M S1 x ~10M S2/S3 records.

Approach:
  1. BLOCKING: Country-partitioned TF-IDF (char n-grams) with sparse matrix
     dot product for memory-efficient top-K retrieval.
  2. FEATURES: RapidFuzz string similarity + token overlap features.
  3. MODEL: LightGBM binary classifier.

Usage:  python entity_resolution.py
"""

import os, gc, re, sys, time, warnings, unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm
import lightgbm as lgb
from rapidfuzz import fuzz

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================= CONFIG =====================================
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN = os.path.join(BASE, "dataset", "train")
TEST  = os.path.join(BASE, "dataset", "test")
OUT   = os.path.join(BASE, "output")

TOPK          = 20       # candidates per S1 entity
COS_BATCH     = 2000     # S1 entities per sparse dot product batch
TFIDF_FEATS   = 80000    # max TF-IDF features
SAMPLE_FRAC   = 0.08     # fraction of train S1 for model training
THRESHOLD     = 0.45     # match probability threshold
N_BOOST       = 300      # LightGBM rounds

# ============================= TEXT NORM ==================================
_P = re.compile(r"[^\w\s]")
_S = re.compile(r"\s+")

_SUF = [(re.compile(p), r) for p, r in [
    (r'\bcorporation\b','corp'),(r'\bincorporated\b','inc'),(r'\blimited\b','ltd'),
    (r'\bcompany\b','co'),(r'\bprivate\b','pvt'),(r'\btechnolog(?:y|ies)\b','tech'),
    (r'\benterprises?\b','ent'),(r'\bindustries\b','ind'),(r'\binternational\b','intl'),
    (r'\bsolutions?\b','sol'),(r'\bconsult(?:ants?|ing)\b','consult'),
    (r'\bassociates\b','assoc'),(r'\bproperties\b','prop'),(r'\bmanagement\b','mgmt'),
    (r'\bmanufacturing\b','mfg'),(r'\bengineering\b','eng'),(r'\bdevelopment\b','dev'),
    (r'\bconstruction\b','const'),(r'\bfoundation\b','fdn'),(r'\bholdings?\b','hldg'),
    (r'\bpartners(?:hip)?\b','ptnr'),(r'\bventures?\b','vent'),
    (r'\bservices?\b','svc'),(r'\bgroup\b','grp'),
]]
_ADR = [(re.compile(p), r) for p, r in [
    (r'\broad\b','rd'),(r'\bstreet\b','st'),(r'\bavenue\b','ave'),
    (r'\bboulevard\b','blvd'),(r'\bdrive\b','dr'),(r'\bcourt\b','ct'),
    (r'\blane\b','ln'),(r'\bsuite\b','ste'),(r'\bapartment\b','apt'),
    (r'\bbuilding\b','bldg'),(r'\bfloor\b','fl'),(r'\bnorth\b','n'),
    (r'\bsouth\b','s'),(r'\beast\b','e'),(r'\bwest\b','w'),
    (r'\bhighway\b','hwy'),
]]
_NSTOP = frozenset("the of a an in for on at to by is it and or pvt ltd inc llc llp corp co svc ent ind intl sol consult assoc prop mgmt mfg eng dev const fdn hldg ptnr vent grp tech".split())
_ASTOP = frozenset("rd st ave blvd dr ct ln ste apt bldg fl n s e w hwy nr no null nan none".split())


def _n(t):
    if pd.isna(t): return ""
    t = str(t).lower().strip()
    if t in ("nan","none",""): return ""
    t = unicodedata.normalize("NFKD", t)
    return _S.sub(" ", _P.sub(" ", t)).strip()

def norm_name(t):
    t = _n(t)
    for p, r in _SUF: t = p.sub(r, t)
    return _S.sub(" ", t).strip()

def norm_addr(t):
    t = _n(t)
    for p, r in _ADR: t = p.sub(r, t)
    return _S.sub(" ", t).strip()

def ntoks(s):
    return [w for w in s.split() if w not in _NSTOP and len(w) > 1]

def atoks(s):
    return [w for w in s.split() if w not in _ASTOP and len(w) > 2]

def is_latin(s):
    if not s: return True
    return sum(1 for c in s if c.isascii()) / len(s) > 0.5


# ============================= DATA =======================================
def load_and_prep(path):
    """Load TSV and add normalized columns."""
    print(f"    Loading {os.path.basename(path)}...")
    df = pd.read_csv(path, sep="\t")
    df["business_name"]    = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"]          = df["country"].fillna("")
    print(f"    Normalizing {len(df):,} records...")
    df["nn"] = df["business_name"].apply(norm_name)
    df["na"] = df["business_address"].apply(norm_addr)
    df["nc"] = df["country"].apply(_n)
    # Combined text for TF-IDF: name (weighted by repeating) + address
    df["combo"] = df["nn"] + " " + df["nn"] + " " + df["na"]
    return df


# ============================= BLOCKING ===================================
def sparse_topk_per_row(sim_sparse, topk):
    """
    Given a sparse similarity matrix (n_queries x n_corpus), 
    extract top-K indices and scores per row efficiently.
    Returns list of lists of (index, score) tuples.
    """
    results = []
    # Convert to CSR for efficient row slicing
    sim_csr = sim_sparse.tocsr()
    for i in range(sim_csr.shape[0]):
        row_start = sim_csr.indptr[i]
        row_end   = sim_csr.indptr[i + 1]
        data      = sim_csr.data[row_start:row_end]
        indices   = sim_csr.indices[row_start:row_end]
        
        if len(data) == 0:
            results.append([])
            continue
        
        if len(data) <= topk:
            order = np.argsort(data)[::-1]
        else:
            order = np.argpartition(data, -topk)[-topk:]
            order = order[np.argsort(data[order])[::-1]]
        
        results.append([(int(indices[o]), float(data[o])) for o in order])
    return results


def tfidf_block(s1_df, s2s3_df, topk=TOPK, batch_sz=COS_BATCH):
    """
    Country-partitioned TF-IDF blocking using sparse matrix multiplication.
    For each country: fit TF-IDF on S2/S3, query S1 in batches using sparse
    dot product (A @ B.T stays sparse when both A and B are sparse CSR).
    """
    print("  TF-IDF Blocking...")
    candidates = {}
    
    countries = sorted(set(s1_df["nc"].unique()) | set(s2s3_df["nc"].unique()))
    
    for country in countries:
        s1c = s1_df[s1_df["nc"] == country]
        s23c = s2s3_df[s2s3_df["nc"] == country]
        
        if len(s1c) == 0:
            continue
        
        if len(s23c) == 0:
            for eid in s1c["entity_id"].values:
                candidates[eid] = []
            continue
        
        print(f"    Country '{country}': S1={len(s1c):,} S2/S3={len(s23c):,}")
        
        # Fit TF-IDF on S2/S3 texts
        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=TFIDF_FEATS,
            sublinear_tf=True,
            min_df=2,
            dtype=np.float32,
        )
        
        t0 = time.time()
        s23_vecs = tfidf.fit_transform(s23c["combo"].values)
        # L2-normalize so dot product = cosine similarity
        s23_vecs = sk_normalize(s23_vecs, norm='l2')
        print(f"      TF-IDF S2/S3: {time.time()-t0:.1f}s, shape={s23_vecs.shape}")
        
        s1_texts = s1c["combo"].values
        s1_eids  = s1c["entity_id"].values
        s23_eids = s23c["entity_id"].values
        
        # Query S1 in batches using sparse dot product
        t0 = time.time()
        n = len(s1_eids)
        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            batch_vecs = tfidf.transform(s1_texts[start:end])
            batch_vecs = sk_normalize(batch_vecs, norm='l2')
            
            # Sparse dot product: result stays sparse!
            sim_sparse = batch_vecs @ s23_vecs.T
            
            # Extract top-K from sparse result
            topk_results = sparse_topk_per_row(sim_sparse, topk)
            
            for i, row_topk in enumerate(topk_results):
                eid = s1_eids[start + i]
                candidates[eid] = [(s23_eids[idx], score) for idx, score in row_topk if score > 0.005]
            
            if (start // batch_sz) % 100 == 0:
                elapsed = time.time() - t0
                pct = end / n * 100
                print(f"      Blocked {end:,}/{n:,} ({pct:.0f}%) in {elapsed:.0f}s")
        
        elapsed = time.time() - t0
        print(f"      Done '{country}': {elapsed:.0f}s")
        
        del s23_vecs, tfidf
        gc.collect()
    
    # Ensure all S1 entities have entries
    for eid in s1_df["entity_id"].values:
        if eid not in candidates:
            candidates[eid] = []
    
    n_w = sum(1 for v in candidates.values() if v)
    tc  = sum(len(v) for v in candidates.values())
    print(f"    With cands: {n_w:,}/{len(candidates):,} | Total: {tc:,} | Avg: {tc/max(len(candidates),1):.1f}")
    return candidates


# ============================= FEATURES ===================================
def jac(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)

def feats(n1, a1, n2, a2, bs=0.0):
    """Compute pairwise features."""
    f = {}
    # Name
    f["n_rat"]  = fuzz.ratio(n1,n2)/100
    f["n_tsrt"] = fuzz.token_sort_ratio(n1,n2)/100
    f["n_tset"] = fuzz.token_set_ratio(n1,n2)/100
    f["n_part"] = fuzz.partial_ratio(n1,n2)/100
    f["n_wr"]   = fuzz.WRatio(n1,n2)/100
    
    t1, t2 = set(ntoks(n1)), set(ntoks(n2))
    f["n_jac"] = jac(t1,t2)
    if t1 and t2:
        c = len(t1&t2)
        f["n_omin"] = c/min(len(t1),len(t2))
        f["n_omax"] = c/max(len(t1),len(t2))
    else:
        f["n_omin"]=f["n_omax"]=0.0
    mx = max(len(n1),len(n2))
    f["n_lrat"] = min(len(n1),len(n2))/mx if mx else 0
    f["n_ldif"] = abs(len(n1)-len(n2))
    w1,w2 = n1.split(),n2.split()
    f["n_fw"] = 1.0 if w1 and w2 and w1[0]==w2[0] else 0.0
    f["xscript"] = 1.0 if is_latin(n1) != is_latin(n2) else 0.0
    
    # Address
    f["a_rat"]  = fuzz.ratio(a1,a2)/100
    f["a_tsrt"] = fuzz.token_sort_ratio(a1,a2)/100
    f["a_tset"] = fuzz.token_set_ratio(a1,a2)/100
    f["a_part"] = fuzz.partial_ratio(a1,a2)/100
    
    at1, at2 = set(atoks(a1)), set(atoks(a2))
    f["a_jac"] = jac(at1,at2)
    if at1 and at2:
        c = len(at1&at2)
        f["a_omin"] = c/min(len(at1),len(at2))
        f["a_omax"] = c/max(len(at1),len(at2))
    else:
        f["a_omin"]=f["a_omax"]=0.0
    mx = max(len(a1),len(a2))
    f["a_lrat"] = min(len(a1),len(a2))/mx if mx else 0
    f["a_miss"] = 1.0 if not a1 or not a2 else 0.0
    
    # Numbers
    nums1 = set(re.findall(r'\b\d+\b', n1+" "+a1))
    nums2 = set(re.findall(r'\b\d+\b', n2+" "+a2))
    f["nu_jac"] = jac(nums1,nums2)
    f["nu_ovl"] = len(nums1&nums2)/min(len(nums1),len(nums2)) if nums1 and nums2 else 0
    
    # Blocking score
    f["bscore"] = bs
    return f


# ============================= TRAIN ======================================
def make_train(s1, s2s3, gt, cands, frac=SAMPLE_FRAC):
    """Build training features and labels."""
    print("Preparing training data...")
    
    # GT lookup
    gtm = {}
    for _, r in gt.iterrows():
        m = r["matched_entity_ids"]
        gtm[r["source1_entity_id"]] = set(str(m).split(",")) if pd.notna(m) and m!="" else set()
    
    # Compact lookups: eid -> (nn, na)
    s1m = dict(zip(s1["entity_id"], zip(s1["nn"], s1["na"])))
    s2m = dict(zip(s2s3["entity_id"], zip(s2s3["nn"], s2s3["na"])))
    
    # Sample S1 entities
    ids = list(cands.keys())
    ns = max(1, int(len(ids)*frac))
    sample = set(np.random.choice(ids, ns, replace=False))
    print(f"  Sample: {ns:,}")
    
    rows, labels = [], []
    pf = pm = 0
    for sid in tqdm(sample, desc="  Features", mininterval=5):
        i1 = s1m.get(sid)
        if not i1: continue
        n1, a1 = i1
        truth = gtm.get(sid, set())
        cs = cands.get(sid, [])
        cset = {c[0] for c in cs}
        
        for cid, bs in cs:
            i2 = s2m.get(cid)
            if not i2: continue
            rows.append(feats(n1, a1, i2[0], i2[1], bs))
            lbl = 1 if cid in truth else 0
            labels.append(lbl)
            if lbl: pf += 1
        
        # Missed positives
        for mid in truth - cset:
            i2 = s2m.get(mid)
            if not i2: continue
            rows.append(feats(n1, a1, i2[0], i2[1], 0.0))
            labels.append(1)
            pm += 1
    
    X = pd.DataFrame(rows)
    y = np.array(labels)
    tot = pf + pm
    print(f"  Pairs: {len(X):,} | +:{y.sum():,} | -:{len(y)-y.sum():,}")
    if tot: print(f"  Block recall: {pf/tot:.4f}")
    return X, y, s1m, s2m


def train_lgb(X, y):
    """Train LightGBM."""
    print("Training LightGBM...")
    np_, nn_ = int(y.sum()), int(len(y) - y.sum())
    params = {
        "objective": "binary", "metric": "binary_logloss", "boosting": "gbdt",
        "num_leaves": 127, "learning_rate": 0.1, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 50,
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "scale_pos_weight": nn_ / max(np_, 1),
    }
    ds = lgb.Dataset(X, label=y)
    m = lgb.train(params, ds, num_boost_round=N_BOOST,
                  valid_sets=[ds], callbacks=[lgb.log_evaluation(100)])
    imp = m.feature_importance(importance_type="gain")
    for nm, v in sorted(zip(X.columns, imp), key=lambda x: -x[1])[:8]:
        print(f"    {nm}: {v:.0f}")
    return m


# ============================= PREDICT ====================================
def predict_all(model, cands, s1m, s2m, th=THRESHOLD):
    """Predict matches for all S1 entities."""
    print("Predicting...")
    res = {}
    for sid in tqdm(sorted(cands.keys()), desc="  Score", mininterval=10):
        cs = cands.get(sid, [])
        if not cs:
            res[sid] = []
            continue
        i1 = s1m.get(sid)
        if not i1:
            res[sid] = []
            continue
        n1, a1 = i1
        fr, ci = [], []
        for cid, bs in cs:
            i2 = s2m.get(cid)
            if not i2: continue
            fr.append(feats(n1, a1, i2[0], i2[1], bs))
            ci.append(cid)
        if not fr:
            res[sid] = []
            continue
        X = pd.DataFrame(fr)
        pr = model.predict(X)
        res[sid] = [c for c, p in zip(ci, pr) if p >= th]
    
    nm = sum(1 for v in res.values() if v)
    tm = sum(len(v) for v in res.values())
    print(f"  Matched: {nm:,}/{len(res):,} | Total: {tm:,}")
    return res


# ============================= OUTPUT =====================================
def write_tsv(data, path, k_col, v_col, keys):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{k_col}\t{v_col}\n")
        for k in keys:
            v = data.get(k, [])
            if v and isinstance(v[0], tuple): v = [x[0] for x in v]
            f.write(f"{k}\t{','.join(v)}\n")
    print(f"  -> {path}")


# ============================= MAIN =======================================
def main():
    t0 = time.time()
    print("=" * 70 + "\nBUSINESS ENTITY RESOLUTION\n" + "=" * 70)
    
    # --- Train ---
    print("\n[1] Load train...")
    s1  = load_and_prep(os.path.join(TRAIN, "train_source1.tsv"))
    s2  = load_and_prep(os.path.join(TRAIN, "train_source2.tsv"))
    s3  = load_and_prep(os.path.join(TRAIN, "train_source3.tsv"))
    gt  = pd.read_csv(os.path.join(TRAIN, "train_ground_truth.tsv"), sep="\t")
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
    del s2, s3; gc.collect()
    
    print("\n[2] Block (train)...")
    tc = tfidf_block(s1, s23)
    
    print("\n[3] Features (train)...")
    X, y, s1m, s2m = make_train(s1, s23, gt, tc)
    
    print("\n[4] Train model...")
    model = train_lgb(X, y)
    del X, y, s1, s23, tc, s1m, s2m, gt; gc.collect()
    
    # --- Test ---
    print("\n[5] Load test...")
    ts1 = load_and_prep(os.path.join(TEST, "test_source1.tsv"))
    ts2 = load_and_prep(os.path.join(TEST, "test_source2.tsv"))
    ts3 = load_and_prep(os.path.join(TEST, "test_source3.tsv"))
    ts23 = pd.concat([ts2, ts3], ignore_index=True)
    print(f"  S1={len(ts1):,} S2={len(ts2):,} S3={len(ts3):,}")
    del ts2, ts3; gc.collect()
    
    print("\n[6] Block (test)...")
    tc2 = tfidf_block(ts1, ts23)
    
    # Build lookups
    tm1 = dict(zip(ts1["entity_id"], zip(ts1["nn"], ts1["na"])))
    tm2 = dict(zip(ts23["entity_id"], zip(ts23["nn"], ts23["na"])))
    del ts23; gc.collect()
    
    print("\n[7] Predict (test)...")
    res = predict_all(model, tc2, tm1, tm2)
    
    # --- Output ---
    print("\n[8] Save...")
    keys = sorted(ts1["entity_id"].values)
    write_tsv(res, os.path.join(OUT, "matching_results.tsv"),
              "source1_entity_id", "matched_entity_ids", keys)
    c_out = {k: [x[0] for x in v] for k, v in tc2.items()}
    write_tsv(c_out, os.path.join(OUT, "candidate_pairs.tsv"),
              "source1_entity_id", "candidate_entity_ids", keys)
    
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
