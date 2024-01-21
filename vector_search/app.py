import json
import pickle
from distance import nearest, euclid_distance, cosign_distance  # noqa
from text2vector import text_to_vector  # noqa
from time import perf_counter_ns
from decimal import Decimal


def main(prompt: str, top_k: int = 3):
    # 計測タイマー
    start = perf_counter_ns()
    end = start

    # あらかじめ取得したベクトルの辞書を読み込む
    # ※辞書はチャンクごとにベクトルデータがふられているので、入れ子になっている
    with open("vector.pkl", "rb") as fp:
        vecdata = pickle.load(fp)

    # 入れ子になった辞書を、フラットな検索用の配列に移し替える
    data_list = []
    index_list = []
    for content in vecdata:
        # 入れ子の中に入る
        for vector in content["vector"]:
            # ベクトルデータをフラットな配列にまとめる
            data_list.append(vector)
            # ベクトルデータに対応するインデックスを記録する
            index_list.append(content["index"])

    # 受け取った質問文をベクトル化する
    prompt_vector = text_to_vector(prompt)

    # 最も似ているベクトルの上位k件を検出する
    knn_index = nearest(prompt_vector, data_list, euclid_distance, top_k)
    for k in range(top_k):
        print(vecdata[index_list[knn_index[k]["index"]]]["data"])

    end = perf_counter_ns()
    print(f"処理時間: {Decimal(end - start) / Decimal(1_000_000_000)}秒")


def lambda_handler(event, context):
    main("DK1200をモデルにしたアイドルは誰ですか", top_k=3)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "hello world",
            }
        ),
    }


if __name__ == "__main__":
    import sys

    # ローカル環境で実行する
    if len(sys.argv) >= 2:
        main(sys.argv[1], top_k=1)
