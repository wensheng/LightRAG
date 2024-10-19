import asyncio
import json
import os
from dataclasses import dataclass, field
import sqlite3

import numpy as np
import vectorlite_py

from .utils import logger
from .base import BaseKVStorage, BaseVectorStorage

MAX_ELEMENTS = 1_000_000


@dataclass
class SqliteKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.db")
        self._client = sqlite3.Connection(self._file_name, isolation_level=None)
        self._cur = self._client.cursor()
        self._cur.execute("CREATE TABLE IF NOT EXISTS kv (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self._data = {id: json.loads(data) for id, data in self._client.execute("SELECT * FROM kv")}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        items = [(id, json.dumps(data)) for id, data in self._data.items()]
        self._cur.executemany("INSERT OR REPLACE INTO kv (id, data) VALUES (?, ?)", items)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        new_data = {k: v for k, v in data.items() if k not in self._data}
        items = [(id, json.dumps(data)) for id, data in new_data.items()]
        self._cur.executemany("INSERT OR REPLACE INTO kv (id, data) VALUES (?, ?)", items)
        existing_data = {k: v for k, v in data.items() if k in self._data}
        # merge existing data
        for k, v in existing_data.items():
            existing_data[k].update(self._data[k])
        items = [(id, json.dumps(data)) for id, data in existing_data.items()]
        self._cur.executemany("INSERT OR REPLACE INTO kv (id, data) VALUES (?, ?)", items)

        # TODO: no need to return
        return new_data

    async def drop(self):
        self._cur.execute('drop table kv')
        self._data = {}


@dataclass
class VectorliteDBStorage(BaseVectorStorage):
    """
    Use Sqlite with Vectorlite extension to store vectors
    """
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.db"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )
        self._client = sqlite3.Connection(self._client_file_name, isolation_level=None)
        self._client.enable_load_extension(True)
        self._client.load_extension(vectorlite_py.vectorlite_path())
        self._cur = self._client.cursor()
        self._cur.execute('create table kv (id integer, k text, v text);')

        self._cur.execute(f"""
            create virtual table if not exists vec_kv
            using vectorlite(
                embedding float32[{self.embedding_func.embedding_dim}],
                hnsw(max_elements={MAX_ELEMENTS}), vdb_{self.namespace}.idx
            )
            """)

    async def upsert(self, data: dict[str, dict]) -> list:
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not data:
            logger.warning("You insert an empty data to vector DB")
            return []
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        list_data = [
            [k, json.dumps({k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields})]
            for k, v in data.items()
        ]
        db_dict = {k: id for id, k in self._cur.execute("SELECT k, id FROM kv").fetchall()}
        last_id = self._cur.execute("SELECT max(id) FROM kv").fetchone()[0] or 0

        old_data = []
        new_data = []
        old_embeddings = []
        new_embeddings = []
        for i, d in enumerate(list_data):
            if d[0] in db_dict:
                old_data.append([last_id + i + 1, *d])
                old_embeddings.append((last_id + i + 1, embeddings[i].astype(np.float32)))
            else:
                new_data.append([last_id + i + 1, *d])
                new_embeddings.append((last_id + i + 1, embeddings[i].astype(np.float32)))
        self._cur.executemany("INSERT INTO kv(id, k, v) VALUES(?, ?, ?)", new_data)
        self._cur.executemany("""
            INSERT INTO vec_kv(rowid, embedding)
            VALUES(?, ?)""",
            [(d[0], d[1].tobytes()) for d in new_embeddings]
        )
        # TODO: no need to return
        return []

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.close()
