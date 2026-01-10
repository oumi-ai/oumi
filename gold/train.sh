oumi train -c configs/countdown_v2.yaml | 2>&1 tee logs/train_countdown_v2.log
wait

# oumi train -c countdown.yaml | 2>&1 tee train_countdown.log
# wait

# python countdown_trl.py | 2>&1 tee train_countdown_hf.log
# wait

# oumi train -c debug_2.yaml | tee train_debug_2.log
# wait

# oumi train -c debug_3.yaml | tee train_debug_3.log
# wait

# oumi train -c debug_4.yaml | tee train_debug_4.log
# wait


