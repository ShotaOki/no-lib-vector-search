import boto3
import json
from os import environ

region = environ.get("BEDROCK_REGION", "us-east-1")
runtime = boto3.client("bedrock-runtime", region_name=region)


def text_to_vector(prompt: str):
    """テキストをベクトル化する"""
    res = runtime.invoke_model(
        body=json.dumps({"inputText": prompt}).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
        modelId="amazon.titan-embed-text-v1",
    )
    return json.loads(res["body"].read())["embedding"]


def text_to_multimodal_vector(prompt: str):
    """テキストをマルチモーダル向けにベクトル化する"""
    res = runtime.invoke_model(
        body=json.dumps(
            {
                "inputText": prompt,
                "embeddingConfig": {"outputEmbeddingLength": 384},
            }
        ).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
        modelId="amazon.titan-embed-image-v1",
    )
    return json.loads(res["body"].read())["embedding"]


def image_to_multimodal_vector(base64image: str):
    """画像をマルチモーダル向けにベクトル化する"""
    res = runtime.invoke_model(
        body=json.dumps(
            {
                "inputImage": base64image,
                "embeddingConfig": {"outputEmbeddingLength": 384},
            }
        ).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
        modelId="amazon.titan-embed-image-v1",
    )
    return json.loads(res["body"].read())["embedding"]
