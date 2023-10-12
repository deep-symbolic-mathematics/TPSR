# python run.py --pmlb_data_type feynman --lam 0.1 --uct_alg var_p_uct --ucb_constant 4. --ucb_base 10. --run_id 4


# python run.py --pmlb_data_type feynman --lam 0.1 --uct_alg uct --ucb_constant 4. --ucb_base 10. --run_id 4


python run.py --eval_mcts_on_pmlb True \
                   --pmlb_data_type feynman \
                   --target_noise 0.0 \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True



python run.py --eval_mcts_on_pmlb True \
                   --pmlb_data_type \
                   --target_noise 0.0 \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True


python run.py --eval_mcts_on_pmlb True \
                   --pmlb_data_type blackbox \
                   --target_noise 0.0 \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True