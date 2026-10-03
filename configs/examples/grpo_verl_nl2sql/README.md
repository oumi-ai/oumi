# NL2SQL tool-agent GRPO

This example trains a verl tool agent on [Spider](https://github.com/taoyds/spider),
the cross-domain text-to-SQL benchmark of
[Yu et al. (2018)](https://arxiv.org/abs/1809.08887). Spider scores predictions
with exact-set match and with execution accuracy; this example optimizes
execution accuracy — run the predicted and the gold SQL against the same
database and compare the result sets (see the *Evaluation Metrics* discussion in
the paper and the reference implementation in
[`evaluation.py`](https://github.com/taoyds/spider/blob/master/evaluation.py)).
The `sql_execution_match` reward implements that comparison, treating row order
as significant only when the gold query has a top-level `ORDER BY`.

Download the JSON files and SQLite databases from the
[Spider site](https://yale-lily.github.io/spider) or the
[`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider) mirror, then
prepare the rows and launch training:

```bash
python scripts/datasets/build_spider_tool_agent.py \
  --spider-root /path/to/spider \
  --db-root /path/to/spider/database

oumi train -c configs/examples/grpo_verl_nl2sql/train.yaml
```

The Spider root must contain `train_spider.json` (7,000 examples) and `dev.json`
(1,034). The database root must contain `<db_id>/<db_id>.sqlite`.

A few Spider rows carry gold SQL that does not run against its own database. The
builder drops them from both splits — an unscoreable gold aborts the reward
mid-run — logging each one and a per-split kept/dropped count. Both splits
therefore come out smaller than the counts above, so validation accuracy here is
not directly comparable to a published Spider dev number without accounting for
the dropped rows.

## Prepared rows

The builder writes `data/grpo_verl_nl2sql/train.jsonl` and `val.jsonl` in Oumi
`Conversation` format — one row per surviving question, referencing the database
by path instead of copying its contents into the row:

- `messages[0]` — system prompt carrying the DDL of every table in that row's DB.
- `messages[1]` — the natural-language question.
- `metadata.agent_name` — the verl agent loop to run (`tool_agent`).
- `metadata.ground_truth` — the gold SQL, scored by `sql_execution_match`.
- `metadata.tools_kwargs.run_sql.create_kwargs.db_path` — absolute path to the
  SQLite file the `run_sql` tool opens for this rollout.

An example row, with the schema abridged to two tables:

````json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a SQL assistant for a SQLite database with this schema:\nCREATE TABLE singer (\nSinger_ID int,\nName text,\nCountry text,\nSong_Name text,\nSong_release_year text,\nAge int,\nIs_male bool,\nPRIMARY KEY (Singer_ID)\n)\n\nCREATE TABLE stadium (\nStadium_ID int,\nLocation text,\nName text,\nCapacity int,\nHighest int,\nLowest int,\nAverage int,\nPRIMARY KEY (Stadium_ID)\n)\nYou may call the tool run_sql(query) to execute a SQL query and inspect results. When you are done, give your final answer as a single SQL query inside a ```sql code block."
    },
    { "role": "user", "content": "How many singers do we have?" }
  ],
  "metadata": {
    "agent_name": "tool_agent",
    "ground_truth": "SELECT count(*) FROM singer",
    "tools_kwargs": {
      "run_sql": {
        "create_kwargs": {
          "db_path": "/path/to/spider/database/concert_singer/concert_singer.sqlite"
        }
      }
    }
  }
}
````
