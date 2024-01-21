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
