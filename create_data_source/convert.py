import pickle
from pathlib import Path
from sys import path

path.append(str(Path(__file__).parent.parent))

from vector_search.text2vector import text_to_vector  # noqa


EXPORT_DIRECTORY = "data_source"


def convert_text_to_pkl(
    resource_file_name: str, export_file_name: str, chunk_size: int = 300
):
    """
    テキストデータを埋め込みLLMで座標化、pickleファイルに格納する
    resource_file_name: 元データにするテキストファイル
    export_file_name: pickleファイルのファイル名、拡張子はpkl
    chunk_size: テキストデータを分割する文字数
    """
    # テキストデータを読み込む
    with open(resource_file_name, encoding="utf8") as fp:
        data = [line.strip() for line in fp.readlines()]

    # チャンクに分割して、テキストを格納する
    vecdata = []
    for index, line in enumerate(data):
        vecdata.append(
            {
                "index": index,
                "data": line,
                "chunk": [
                    line[l : l + chunk_size]  # noqa
                    for l in range(0, len(line), chunk_size)  # noqa
                ],
                "vector": [],
            }
        )

    # チャンクをベクトル化する
    for content in vecdata:
        for chunk in content["chunk"]:
            content["vector"].append(text_to_vector(chunk))
            print(chunk)

    # data_sourceディレクトリに、データをpickle形式で保存する
    with open(
        str(Path(__file__).parent.parent / EXPORT_DIRECTORY / export_file_name), "wb"
    ) as fp:
        pickle.dump(vecdata, fp)


convert_text_to_pkl("yes-no.txt", "yes-no.pkl")
