def distance_pow(vec: list[float]) -> float:
    """
    距離の二乗を取得する
    vec: 取得する対象のベクトル
    """
    return sum([v * v for v in vec])


def cosign_distance(vec1: list[float], vec2: list[float]):
    """
    コサイン距離を取得する(似ているほど-1に近い、似ていないほど大きくなる)
    """
    # ノルム同士をかけてL2を取る(0.5乗してルートを取る)
    l2_length = (distance_pow(vec1) ** 0.5) * (distance_pow(vec2) ** 0.5)
    if l2_length == 0.0:
        # L2が異常値になるなら0を返す
        return 0.0
    # 内積を取る
    dot_product = sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])
    # コサイン類似度を計算する（変化の方向をユークリッド距離に合わせたいので、1.0から引いてコサイン距離とする）
    return 1.0 - (dot_product / l2_length)


def euclid_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    ユークリッド距離の二乗を取得する(似ているほど0に近い、似ていないほど大きくなる)
    """
    # zipで複数の配列を1つの配列にまとめる
    return distance_pow([v1 - v2 for v1, v2 in zip(vec1, vec2)])


def calc_distance_list(
    base_point: list[float], reference_points: list[list], distance_function
):
    """
    基準点(base_point)を元に、データ点(reference_points)それぞれまでの距離を取得する
    """
    return [
        # 基準点から対象の点までの距離を求めて、距離リストに追加する
        distance_function(blue_point, base_point)
        for blue_point in reference_points
    ]


def nearest(
    base_point: list[float],
    reference_points: list[list],
    distance_function=cosign_distance,
    topk: int = 1,
):
    """
    基準点から見て、最も距離の近いデータ点を取得する
    """
    knn = sorted(
        # 距離のリストに、インデックスをつけてソートする
        enumerate(calc_distance_list(base_point, reference_points, distance_function)),
        # enumerateで、インデックスと値の配列になるので、x[1]で値を取得する
        key=lambda x: x[1],
        # 距離の昇順でソート、距離が小さいものを先頭に置く
        reverse=False,
    )
    return [{"index": k[0], "score": k[1]} for k in knn[:topk]]
