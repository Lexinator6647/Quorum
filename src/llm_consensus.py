# Install dependencies first:
# pip install sentence-transformers allennlp allennlp-models

from sentence_transformers import SentenceTransformer, util
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import numpy as np

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load Semantic Role Labeler
srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

#############################################
# Mock Query Functions
#############################################

def query_model_1(prompt):
    return "Albert Einstein proposed the theory of relativity in 1905."

def query_model_2(prompt):
    return "In 1905, Einstein introduced his relativity theory."

def query_model_3(prompt):
    return "Einstein's theory of relativity was published in the early 20th century."

def simulated_model(prompt):
    """
    Pretend this is a single model you can call repeatedly.
    """
    import random
    variants = [
        "Albert Einstein proposed the theory of relativity in 1905.",
        "In 1905, Einstein introduced his relativity theory.",
        "Einstein's relativity theory was published in 1905.",
        "Relativity was first proposed by Einstein in the early 20th century."
    ]
    return random.choice(variants)

#############################################
# Evaluation Utilities
#############################################

def compute_similarity_matrix(texts):
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    return similarity_matrix

# Semantic Role Labelling
def extract_srl_roles(sentence):
    srl_result = srl_predictor.predict(sentence=sentence)
    roles = []
    for verb in srl_result['verbs']:
        description = verb['description']
        roles.append(description)
    return roles

def srl_overlap(roles1, roles2):
    set1 = set(roles1)
    set2 = set(roles2)
    if not set1 or not set2:
        return 0.0
    # return Jaccard similarity coefficient (intersection over union) by comparing roles as a full string
    # How much do these objects agree relative to their full extent
    return len(set1 & set2) / len(set1 | set2)

def srl_cosine_similarity(roles1, roles2):
    # Compute cosine similarity and average it across all semantic roles 
    if not roles1 or not roles2:
        return 0.0

    embeddings1 = embedder.encode(roles1, convert_to_tensor=True)
    embeddings2 = embedder.encode(roles2, convert_to_tensor=True)

    # Compute cosine similarity matrix
    sim_matrix = util.cos_sim(embeddings1, embeddings2)

    # Take the best match for each role in roles1
    max_sim1 = sim_matrix.max(dim=1).values
    max_sim2 = sim_matrix.max(dim=0).values

    # Average them to get a symmetric score
    return (max_sim1.mean().item() + max_sim2.mean().item()) / 2

def evaluate_responses(responses):
    # Semantic similarity
    sim_matrix = compute_similarity_matrix(responses)

    # SRL roles
    srl_roles_list = [extract_srl_roles(r) for r in responses]

    # Pairwise SRL overlaps
    overlaps = []
    srl_cosines = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            overlap = srl_overlap(srl_roles_list[i], srl_roles_list[j])
            overlaps.append(((i, j), overlap))
            srl_cos = srl_cosine_similarity(srl_roles_list[i], srl_roles_list[j])
            srl_cosines.append(srl_cos)

    return sim_matrix, overlaps, srl_cosines

#############################################
# Sampling Utilities
#############################################

def sample_model_multiple_times(model_fn, prompt, n):
    return [model_fn(prompt) for _ in range(n)]

def sample_multiple_models(model_fns, prompt):
    return [fn(prompt) for fn in model_fns]

#############################################
# Main Routine
#############################################

def main():
    prompt = "When did Einstein propose the theory of relativity?"

    ######################################
    # CONSISTENCY SAMPLING (same model, n times)
    ######################################
    print("\n=== MODEL CONSISTENCY SAMPLING ===")
    responses_consistency = sample_model_multiple_times(simulated_model, prompt, n=5)
    for i, r in enumerate(responses_consistency):
        print(f"Sample {i+1}: {r}")

    sim_matrix_c, srl_overlaps_c, srl_cosine_c = evaluate_responses(responses_consistency)

    print("\nSemantic Similarity Matrix (Consistency):")
    print(sim_matrix_c)

    print(f"\nMean SRL Overlaps (Consistency):{np.mean(srl_overlaps_c)}")

    print(f"\nMean SRL Cosine (Consistency):{np.mean(srl_cosine_c)}")


    ######################################
    # CONSENSUS SAMPLING (different models)
    ######################################
    print("\n=== CONSENSUS SAMPLING ===")
    model_fns = [query_model_1, query_model_2, query_model_3]
    responses_consensus = sample_multiple_models(model_fns, prompt)
    for i, r in enumerate(responses_consensus):
        print(f"Model {i+1}: {r}")

    sim_matrix_s, srl_overlaps_s, srl_cosine_s = evaluate_responses(responses_consensus)
    
    print("\nSemantic Similarity Matrix (Consensus):")
    print(sim_matrix_s)

    print(f"\nMean SRL Overlaps (Consensus):{np.mean(srl_overlaps_s)}")

    print(f"\nMean SRL Cosine (Consensus):{np.mean(srl_cosine_s)}")


if __name__ == "__main__":
    main()
