import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

load_dotenv()


def get_aws_es(endpoint: str, access_key: str, access_secret: str, region: str) -> Elasticsearch:
    awsauth = AWS4Auth(access_key, access_secret, region, 'es')

    return Elasticsearch(
        hosts=endpoint,
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )


def get_aws_es_from_environment() -> Elasticsearch:
    endpoint = os.environ['AWS_ES_ENDPOINT']
    access_key = os.environ['AWS_ACCESS_KEY']
    access_secret = os.environ['AWS_ACCESS_SECRET']
    region = 'us-east-2'

    return get_aws_es(endpoint, access_key, access_secret, region)
