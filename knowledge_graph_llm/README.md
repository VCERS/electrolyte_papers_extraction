# Introduction

this project is to provide a tool to extract knowledge graph from patent materials

# Usage

## install neo4j

```shell
docker pull neo4j:enterprise-bullseye
docker run -d --publish=7474:7474 --publish=7687:7687 \
           --volume=$HOME/neo4j/data:/data \
           --name neo4j-apoc \
           -e NEO4J_apoc_export_file_enabled=true \
           -e NEO4J_apoc_import_file_enabled=true \
           -e NEO4J_apoc_import_file_use__neo4j__config=true \
           -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
           --privileged --shm-size 12G -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes --cpus=32 --memory=128G neo4j 
```

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```
## Fix issues of langchain-experimental

edit line 551 of **<path/to/site-packages>/langchain_experimental/graph_transformers/llm.py** to change the code from

```python
except NotImplementedError:
```

to

```python
except:
```

edit line 595 of **<path/to/site-packages>/langchain_experimental/graph_transformers/llm.py** to change the code from

```python
parsed_json = self.json_repair.loads(raw_schema.content)
```

to

```python
parsed_json = self.json_repair.loads(raw_schema)
```

edit line 597 of **<path/to/site-packages>/langchain_experimental/graph_transformers/llm.py** to change the code from

```python
for rel in parsed_json:
    # Nodes need to be deduplicated using a set
    nodes_set.add((rel["head"], rel["head_type"]))
    nodes_set.add((rel["tail"], rel["tail_type"]))
```

to

```python
for rel in parsed_json:
    if type(rel) is not dict: continue
    elif "head" not in rel or "head_type" not in rel: continue
    elif "tail" not in rel or "tail_type" not in rel: continue
    elif rel["head"] is None or rel["head_type"] is None or rel['head_type'] == '': continue
    elif rel["tail"] is None or rel["tail_type"] is None or rel['tail_type'] == '': continue
    # Nodes need to be deduplicated using a set 
    nodes_set.add((rel["head"], rel["head_type"]))
    nodes_set.add((rel["tail"], rel["tail_type"]))
```

edit line 149 of **<path/to/site-package>/langchain_experimental/sql/base.py** to change the code from

```python
result = self.database.run(sql_cmd)
```

to
```python
import re
pattern = r"```sql(.*)```"
match = re.search(pattern, sql_cmd, re.DOTALL)
if match is None:
  pattern = r"```(.*)```"
  match = re.search(pattern, sql_cmd, re.DOTALL)
sql_cmd = match[1] if match is not None else sql_cmd
result = self.database.run(sql_cmd)
```

## Extract Knowledge Graph

```shell
python3 main.py --model (llama3|qwen2) --input_dir <path/to/directory/of/patents> [--host <host>] [--user <user>] [--password <password>] [--database <database>] [--locally]
```

## Export the database in cypher format

```shell
bash run_cypher.sh > output.cypher
```

the exported all.cypher file is under **/var/lib/neo4j/import/**
