import json
import pickle
from chalicelib.distance import nearest, euclid_distance, cosign_distance  # noqa
from pathlib import Path
import boto3
from os import environ
from chalice_a4ab import Chalice, AgentsForAmazonBedrockConfig
from chalice_spec.docs import Docs, Operation
from pydantic import BaseModel

AgentsForAmazonBedrockConfig(
    instructions=(
        "あなたはデータベースに登録されたデータをもとに、ユーザーからの質問に答えることができます。"
        "質問にはできるだけ丁寧に、根拠資料に沿った回答を返してください。"
    ),
).apply()

region = environ.get("BEDROCK_REGION", "us-east-1")
runtime = boto3.client("bedrock-runtime", region_name=region)
app = Chalice(app_name="vector-search")


class TalkInput(BaseModel):
    """
    ユーザーからの質問を受け取ります

    ユーザーからの質問を受け取ります。質問は加工せず、ユーザーから受けた原文のままにしてください。
    """

    question: str


class KnowledgeData(BaseModel):
    """
    根拠資料のデータを返します

    根拠資料のデータを返します
    """

    knowledge: str


def text_to_vector(prompt: str):
    """テキストをベクトル化する"""
    res = runtime.invoke_model(
        body=json.dumps({"inputText": prompt}).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
        modelId="amazon.titan-embed-text-v1",
    )
    return json.loads(res["body"].read())["embedding"]


def main(prompt: str, top_k: int = 3):
    # あらかじめ取得したベクトルの辞書を読み込む
    # ※辞書はチャンクごとにベクトルデータがふられているので、入れ子になっている
    with open(str(Path(__file__).parent / "chalicelib" / "vector.pkl"), "rb") as fp:
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
        return vecdata[index_list[knn_index[k]["index"]]]["data"]

    return "根拠資料がありませんでした"


@app.route(
    "/question",
    methods=["POST"],
    docs=Docs(
        post=Operation(
            request=TalkInput,
            response=KnowledgeData,
        )
    ),
)
def index():
    """
    ユーザーからの質問を元に、根拠資料を検索します

    ユーザーがした質問に対して、根拠資料を調べて、根拠資料の内容を返します。
    """
    input = TalkInput.model_validate(app.current_request.json_body)
    return KnowledgeData(knowledge=main(input.question, top_k=1)).model_dump_json()
