import csv

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from clustering_chat.elasticsearch import get_aws_es_from_environment


def main(es_client: Elasticsearch, es_index: str, out_path: str):
    response = scan(es_client, index=es_index)
    docs = (dic['_source'] for dic in response)

    with open(out_path, 'w') as output_file:
        keys = [
            'serial_number',
            'message',
            'video.url',
            'video.title',
            'video.started_at',
            'video.duration',
            'channel.url',
            'channel.name',
            'author.url',
            'author.name',
            'published_at',
            'superchat.value',
            'superchat.currency'
        ]
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(docs)


if __name__ == '__main__':
    es = get_aws_es_from_environment()
    main(es, es_index='livechat', out_path='resources/raw.csv')
