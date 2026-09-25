# Business Entity Resolution

## Overview
This pipeline resolves business entities across three independent data sources using:
1. **Blocking**: Country-partitioned inverted index on name tokens, character trigrams, and address tokens
2. **Feature Engineering**: RapidFuzz string similarity features (ratio, token_sort, token_set, partial, WRatio) + token overlap features
3. **Matching**: LightGBM binary classifier

## Requirements
```
pip install -r requirements.txt
```

## How to Run
```bash
cd student_resource
python code/business_entity_resolution/src/entity_resolution.py
```

This will:
1. Load training data and ground truth
2. Build inverted index on S2/S3 training data
3. Generate blocking candidates for training S1 entities
4. Extract similarity features and train LightGBM
5. Load test data and repeat blocking + prediction
6. Output `matching_results.tsv` and `candidate_pairs.tsv` to `output/`

## Pipeline Architecture

### Blocking Strategy
- **Country Partitioning**: Only compare entities within the same country
- **Name Token Index**: Meaningful name tokens (stop words removed) → entity IDs
- **Character Trigram Index**: 3-char sliding windows on name → entity IDs  
- **Address Token Index**: Address tokens → entity IDs (catches cross-script/transliteration matches)
- **Weighted Scoring**: Name tokens (3x), Address tokens (2x), Trigrams (1x)
- **Top-K Selection**: Keep top 20 candidates per S1 entity

### Features
- Name similarity: ratio, token_sort_ratio, token_set_ratio, partial_ratio, WRatio
- Name token overlap: Jaccard, min-overlap, max-overlap
- Name length features: ratio, absolute difference
- First word match indicator
- Cross-script indicator
- Address similarity: ratio, token_sort_ratio, token_set_ratio, partial_ratio
- Address token overlap: Jaccard, min-overlap, max-overlap
- Number overlap: Jaccard on extracted digits
- Blocking score

### Model
- LightGBM with binary cross-entropy loss
- Scale_pos_weight for class imbalance
- 300 boosting rounds
