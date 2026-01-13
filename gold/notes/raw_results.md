# Results

## First test (num_train_epochs=7)
| Model                     | Correct | Accuracy | Missing Answer Tags | Eqn Not Balanced | Invalid Number Usage | LHS Parse/Eval Error | RHS Parse/Eval Error | Wrong Value |
|---------------------------|---------|----------|---------------------|------------------|----------------------|----------------------|----------------------|-------------|
| qwen2.5_7b_baseline       | 3336    | 0.3336   | 2408                | 2186             | 2013                 | 52                   | 2                    | 3           |
| qwen2.5_1.5b_baseline     | 1264    | 0.1264   | 3997                | 2295             | 2291                 | 112                  | 8                    | 33          |
| qwen2.5_1.5b_ckpt500      | 5394    | 0.5394   | 4329                | 91               | 156                  | 13                   | 0                    | 17          |
| qwen2.5_1.5b_ckpt1000     | 5807    | 0.5807   | 3924                | 129              | 112                  | 2                    | 0                    | 26          |
| qwen3_4b_baseline         | 7931    | 0.7931   | 1783                | 1                | 275                  | 6                    | 0                    | 4           |


qwen2.5_7b_baseline
```
Total: 10000
Correct: 3336
Accuracy: 0.333600

Failure breakdown:
  missing_answer_tags: 2408
  equation_not_balanced: 2186
  invalid_number_usage: 2013
  lhs_parse_or_eval_error: 52
  wrong_value: 3
  rhs_parse_or_eval_error: 2
```

qwen2.5_1.5b_baseline
```
Total: 10000
Correct: 1264
Accuracy: 0.126400

Failure breakdown:
  missing_answer_tags: 3997
  equation_not_balanced: 2295
  invalid_number_usage: 2291
  lhs_parse_or_eval_error: 112
  wrong_value: 33
  rhs_parse_or_eval_error: 8
```

qwen2.5_1.5b_ckpt500
```
Total: 10000
Correct: 5394
Accuracy: 0.539400

Failure breakdown:
  missing_answer_tags: 4329
  invalid_number_usage: 156
  equation_not_balanced: 91
  wrong_value: 17
  lhs_parse_or_eval_error: 13
```

qwen2.5_1.5b_ckpt1000
```
Total: 10000
Correct: 5807
Accuracy: 0.580700

Failure breakdown:
  missing_answer_tags: 3924
  equation_not_balanced: 129
  invalid_number_usage: 112
  wrong_value: 26
  lhs_parse_or_eval_error: 2
```

qwen3_4b_baseline
```
Total: 10000
Correct: 7931
Accuracy: 0.793100

Failure breakdown:
  missing_answer_tags: 1783
  invalid_number_usage: 275
  lhs_parse_or_eval_error: 6
  wrong_value: 4
  equation_not_balanced: 1
```


## k=1
| Checkpoint           | Total | Correct | Accuracy | missing_answer_tags | equation_not_balanced | invalid_number_usage | wrong_value | lhs_parse_or_eval_error | rhs_parse_or_eval_error |
|----------------------|------:|--------:|---------:|--------------------:|----------------------:|---------------------:|------------:|------------------------:|------------------------:|
| 500                  | 10000 | 5624    | 0.5624   | 3507                | 356                   | 292                  | 173         | 39                      | 9                       |
| 1000                 | 10000 | 6043    | 0.6043   | 3538                | 111                   | 241                  | 63          | 4                       | 0                       |
| 2000                 | 10000 | 2615    | 0.2615   | 27                  | 3213                  | 562                  | 1855        | 1680                    | 48                      |
| 3000                 | 10000 | 2636    | 0.2636   | 29                  | 3216                  | 542                  | 1854        | 1677                    | 46                      |
| qwen3_4b_baseline    | 10000 | 7931    | 0.7931   | 1783                | 1                     | 275                  | 4           | 6                       | 0                       |

19pts


500
```
Total: 10000
Correct: 5624
Accuracy: 0.562400

Failure breakdown:
  missing_answer_tags: 3507
  equation_not_balanced: 356
  invalid_number_usage: 292
  wrong_value: 173
  lhs_parse_or_eval_error: 39
  rhs_parse_or_eval_error: 9
```

1000
```
Total: 10000
Correct: 6043
Accuracy: 0.604300

Failure breakdown:
  missing_answer_tags: 3538
  invalid_number_usage: 241
  equation_not_balanced: 111
  wrong_value: 63
  lhs_parse_or_eval_error: 4
```

2000
```
Total: 10000
Correct: 2615
Accuracy: 0.261500

Failure breakdown:
  equation_not_balanced: 3213
  wrong_value: 1855
  lhs_parse_or_eval_error: 1680
  invalid_number_usage: 562
  rhs_parse_or_eval_error: 48
  missing_answer_tags: 27
```

3000
```
Total: 10000
Correct: 2636
Accuracy: 0.263600

Failure breakdown:
  equation_not_balanced: 3216
  wrong_value: 1854
  lhs_parse_or_eval_error: 1677
  invalid_number_usage: 542
  rhs_parse_or_eval_error: 46
  missing_answer_tags: 29
```


| Steps | Train Loss | Eval Loss | Grad Norm |
|-------|------------|-----------|-----------|
| 500   | 0.43       | 0.21      | 4.75      |
| 1000  | 0.38       | 0.19      | 3.00      |
| 2000  | 0.32       | 0.16      | 2.62      |
| 3000  | 0.34       | 0.15      | 3.28      |


## k=4

| N    | Total | Passed (pass@4) | Pass@4  | Missing Answer Tags | LHS Parse/Eval Error | RHS Parse/Eval Error | Equation Not Balanced | Invalid Number Usage | Wrong Value |
|------|-------|------------------|---------|---------------------|----------------------|----------------------|-----------------------|----------------------|-------------|
| 500  | 10000 | 7306             | 0.7306  | 1439                | 499                  | 4                    | 389                   | 237                  | 126         |
| 1000 | 10000 | 8022             | 0.8022  | 1371                | 112                  | 5                    | 188                   | 223                  | 79          |
| 1500 | 10000 | 8114             | 0.8114  | 1616                | 62                   | 6                    | 108                   | 76                   | 18          |
| 2000 | 10000 | 7789             | 0.7789  | 1366                | 310                  | 9                    | 209                   | 270                  | 47          |
| 3000 | 10000 | 7713             | 0.7713  | 1384                | 349                  | 4                    | 195                   | 288                  | 67          |

500
Total: 10000
Passed (pass@4): 7306
Pass@4: 0.730600

Failure breakdown (based on newest completion when row fails):
  missing_answer_tags: 1439
  lhs_parse_or_eval_error: 499
  equation_not_balanced: 389
  invalid_number_usage: 237
  wrong_value: 126
  rhs_parse_or_eval_error: 4

1000
Total: 10000
Passed (pass@4): 8022
Pass@4: 0.802200

Failure breakdown (based on newest completion when row fails):
  missing_answer_tags: 1371
  invalid_number_usage: 223
  equation_not_balanced: 188
  lhs_parse_or_eval_error: 112
  wrong_value: 79
  rhs_parse_or_eval_error: 5

1500
Total: 10000
Passed (pass@4): 8114
Pass@4: 0.811400

Failure breakdown (based on newest completion when row fails):
  missing_answer_tags: 1616
  equation_not_balanced: 108
  invalid_number_usage: 76
  lhs_parse_or_eval_error: 62
  wrong_value: 18
  rhs_parse_or_eval_error: 6

2000
Total: 10000
Passed (pass@4): 7789
Pass@4: 0.778900

Failure breakdown (based on newest completion when row fails):
  missing_answer_tags: 1366
  lhs_parse_or_eval_error: 310
  invalid_number_usage: 270
  equation_not_balanced: 209
  wrong_value: 47
  rhs_parse_or_eval_error: 9

3000
Total: 10000
Passed (pass@4): 7713
Pass@4: 0.771300

Failure breakdown (based on newest completion when row fails):
  missing_answer_tags: 1384
  lhs_parse_or_eval_error: 349
  invalid_number_usage: 288
  equation_not_balanced: 195
  wrong_value: 67
  rhs_parse_or_eval_error: 4




reported teacher score: 0.7145
reported trained model score (lambda=0.5, pass@4): 0.6345 (0.08 delta)

qwen2.5_1.5b_baseline
Total: 10000
Passed (pass@4): 1482
Pass@4: 0.148200

Failure breakdown (based on newest completion when row fails):
  invalid_number_usage: 3582
  equation_not_balanced: 2817
  missing_answer_tags: 1109
  lhs_parse_or_eval_error: 921
  wrong_value: 53
  rhs_parse_or_eval_error: 36

qwen2.5_7b_baseline
Total: 10000
Passed (pass@4): 6029
Pass@4: 0.602900

Failure breakdown (based on newest completion when row fails):
  equation_not_balanced: 2167
  invalid_number_usage: 1633
  missing_answer_tags: 89
  lhs_parse_or_eval_error: 77
  wrong_value: 3
  rhs_parse_or_eval_error: 2