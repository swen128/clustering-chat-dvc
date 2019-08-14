from pandas import DataFrame

from scripts.evaluate import aggregate_scores


def test_aggregate_scores():
    data = [
        dict(tokens=['テスト', 'トークン'], is_representative=False, clustering_label=1, document_vector=[0.1, 0.2, 1.3]),
        dict(tokens=['test', 'token'], is_representative=True, clustering_label=1, document_vector=[0.14, 0.23, 1.3]),
        dict(tokens=['ダミー', 'dummy'], is_representative=True, clustering_label=2, document_vector=[0.9, -32.0, 0.0])
    ]
    df = DataFrame(data)
    dfs = [df] * 5

    scores = aggregate_scores(dfs)

    assert scores == {
        'Calinski_Harabasz': 554722.4666666666,
        'Davies_Bouldin': 0.0007751781003320896,
        'Silhouette': 0.6656330958633799,
        'entropy_reduction': 0.41972078914818756,
        'vocabulary_coverage': 0.6666666666666666,
        'weighted_vocabulary_coverage': 0.6666666666666666
    }
