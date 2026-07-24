# NL2SQL tool-agent GRPO

This example trains a verl tool agent on Spider execution accuracy. Prepare the
Spider JSON files and SQLite databases before launching training:

```bash
python scripts/datasets/build_spider_tool_agent.py \
  --spider-root /path/to/spider \
  --db-root /path/to/spider/database

oumi train -c configs/examples/grpo_verl_nl2sql/train.yaml
```

The Spider root must contain `train_spider.json` and `dev.json`. The database
root must contain `<db_id>/<db_id>.sqlite`. The builder writes the configured
`data/grpo_verl_nl2sql/train.jsonl` and `val.jsonl` files without copying the
database contents into each row.
