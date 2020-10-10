import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
import os


class DB:
    def __init__(self):
        self.conn = Elasticsearch(os.environ["ELASTICSEARCH_HOST"], timeout=60)

    def query(self, method, index, query):
        method = method.lower()
        assert method in ["create", "read", "update", "delete"]

        if method == "create":
            return self.create(index, query)
        if method == "read":
            return self.read(index, query)
        if method == "update":
            return self.update(index, query)
        if method == "delete":
            return self.delete(index, query)

    def create(self, index, docs):
        response = bulk(self.conn, docs, index=index)
        return response

    def read(self, index, query):
        return [
            [doc["_id"], doc["_source"]]
            for doc in scan(self.conn, index=index, query=query)
        ]

    def update(self, index, query):
        response = self.conn.update_by_query(index=index, body=query)
        return response

    def delete(self, index, query):
        response = self.conn.delete_by_query(index=index, body=query)
        return response
