import time
import json

import torch
import numpy as np
import sympy as sp
from parsers import get_parser

import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer
from model import Transformer, pred_for_sample_no_refine, respond_to_batch , pred_for_sample, refine_for_sample, pred_for_sample_test, refine_for_sample_test 
from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json
from rl_env import RLEnv
from default_pi import E2EHeuristic
from symbolicregression.metrics import compute_metrics


def evaluate_metrics(params, y_gt, tree_gt, y_pred, tree_pred):

    metrics = [] # 7 metrics and for all samples to evaluate
    results_fit = compute_metrics(
        {
            "true": [y_gt],
            "predicted": [y_pred],
            "tree": tree_gt,
            "predicted_tree": tree_pred,
        },
        metrics=params.validation_metrics,
    )
    for k, v in results_fit.items():
        print("metric {}: ".format(k), v)
        metrics.append(v[0])
    
    return metrics

def compute_nmse(y_gt , y_pred):
    eps = 1e-9 # For avoiding Nan or Inf
    return np.sqrt( np.mean((y_gt - y_pred)**2) / (np.mean((y_gt)**2)+eps) ) 

def compute_mse(y_gt , y_pred):
    return np.mean((y_gt - y_pred)**2)

def main(case, params, equation_env, samples):
    model = Transformer(params = params, env=equation_env, samples=samples)
    model.to(params.device) 
    generations_ref, gen_len_ref = respond_to_batch(model, max_target_length=200, top_p=1.0, sample_temperature=None) 
    sequence_ref = generations_ref[0][:gen_len_ref-1].tolist()
    
    rl_env = RLEnv(
        params = params,
        samples = samples,
        equation_env = equation_env,
        model = model
    )

    dp = E2EHeuristic(
            equation_env=equation_env,
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty = params.beam_length_penalty,
            train_value_mode=params.train_value,
            debug=params.debug
        )

    start = time.time()

    agent = UCT(
        action_space=[], # this will not be used as we have a default policy
        gamma=1., # no discounting
        ucb_constant=1.,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True,
        alg=params.uct_alg,
        ucb_base=params.ucb_base
    )

    agent.display()

    if params.sample_only:
        # a bit hacky, should set a large rollout number so all programs are saved in samples json file
        horizon = 1
    else:
        horizon = 200
        
    # try:
    done = False
    s = rl_env.state
    ret_all = []
    for t in range(horizon):
        if len(s) >= params.horizon:
            print(f'Cannot process programs longer than {params.horizon}. Stop here.')
            break

        if done:
            break

        act = agent.act(rl_env, done)
        s, r, done, _ = rl_env.step(act)
        
        if t ==0:
            real_root = agent.root

        if params.debug:
            # print the current tree
            print('tree:')
            # print_tree(agent.root, equation_env.equation_id2word)
            ret = convert_to_json(agent.root, rl_env, act)
            ret_all.append(ret)
            
            with open("tree.json", "w") as outfile:
                json.dump(ret_all, outfile)

            print('took action:')
            print(repr(equation_env.equation_id2word[act]))
            print('========== state (excluding prompt) ==========')
            print(s)

        update_root(agent, act, s)
        dp.update_cache(s)


    time_elapsed = time.time() - start

    y_gt = samples['y_to_fit'][0].reshape(-1)
    y_gt_test = samples['y_to_pred'][0].reshape(-1)

    y_ref , ref_str , ref_tree = pred_for_sample_no_refine(model, equation_env, sequence_ref ,samples['x_to_fit'])
    NMSE_ref = compute_nmse(y_gt, y_ref)
    ref_reward = rl_env.get_reward(sequence_ref, mode='test')
    
    y_ref_search , ref_str_search, ref_tree_search = pred_for_sample(model, equation_env,samples['x_to_fit'],samples['y_to_fit'], refine=False, beam_type='search', beam_size=100)
    NMSE_ref_search= compute_nmse(y_gt, y_ref_search)

    y_ref_sample , ref_str_sample, ref_tree_sample = pred_for_sample(model, equation_env,samples['x_to_fit'],samples['y_to_fit'], refine=False, beam_type='sampling', beam_size=100)
    NMSE_ref_sample = compute_nmse(y_gt, y_ref_sample)

    y_ref_refine , ref_str_refine, ref_tree_refine = pred_for_sample(model, equation_env,samples['x_to_fit'],samples['y_to_fit'], refine=True, beam_type='sampling', beam_size=100)
    NMSE_ref_refine = compute_nmse(y_gt, y_ref_refine)
    # MSE_ref_refine = compute_mse(y_gt, y_ref_refine)

    y_ref_refine_train ,y_ref_refine_test, _, _= pred_for_sample_test(model, equation_env,samples['x_to_fit'],samples['y_to_fit'],samples['x_to_pred'], refine=True, beam_type='sampling', beam_size=100)
    MSE_ref_refine = compute_mse(y_gt_test, y_ref_refine_test)

    y_mcts , mcts_str , mcts_tree = pred_for_sample_no_refine(model, equation_env, s ,samples['x_to_fit'])
    NMSE_mcts = compute_nmse(y_gt, y_mcts)
    final_reward = rl_env.get_reward(s, mode='test')

    y_mcts_refine , mcts_str_refine, mcts_tree_refine = refine_for_sample(params, model, equation_env, s,  samples['x_to_fit'], samples['y_to_fit'])
    NMSE_mcts_refine = compute_nmse(y_gt, y_mcts_refine)
    # MSE_mcts_refine = compute_mse(y_gt, y_mcts_refine)

    y_mcts_refine_train, y_mcts_refine_test , _, _ = refine_for_sample_test(model, equation_env, s,  samples['x_to_fit'], samples['y_to_fit'],samples['x_to_pred'])
    MSE_mcts_refine = compute_mse(y_gt_test, y_mcts_refine_test)

    print('#'*40)
    print('\nPre-trained Reference NMSE:', NMSE_ref)
    print('Pre-trained Reference NMSE after Refine:', NMSE_ref_refine)
    print('Pre-trained Reference NMSE (Beam Search):', NMSE_ref_search)
    print('Pre-trained Reference NMSE (Sampling):', NMSE_ref_sample)
    print('Pre-trained Reference MSE after Refine:', MSE_ref_refine)
    print('#'*20)
    print('TPSR NMSE:', NMSE_mcts)
    print('TPSR NMSE after Refine', NMSE_mcts_refine)
    print('TPSR MSE after Refine', MSE_mcts_refine)
    print('TPSR Time Elapsed:', time_elapsed)
    print('TPSR Sample Times (# of Explored Equation Candidates):', dp.sample_times)
    print('#'*40)

    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    for op,replace_op in replace_ops.items():
        ref_str = ref_str.replace(op,replace_op)
        ref_str_sample = ref_str_sample.replace(op,replace_op)
        ref_str_search = ref_str_search.replace(op,replace_op)
        mcts_str = mcts_str.replace(op,replace_op)

    mcts_eq = sp.parse_expr(mcts_str)
    mcts_eq_refine = sp.parse_expr(mcts_str_refine)
    ref_eq = sp.parse_expr(ref_str)
    ref_eq_search = sp.parse_expr(ref_str_search)
    ref_eq_sample = sp.parse_expr(ref_str_sample)
    ref_eq_refine = sp.parse_expr(ref_str_refine)

    print("Pre-trained Reference Equation:", ref_eq)
    print("\nPre-trained Reference Equation after Refine: ", ref_eq_refine)
    print("\nPre-trained Reference Equation (Beam Search): ", ref_eq_search)
    print("\nPre-trained Reference Equation (Sampling): ", ref_eq_sample)
    print('#'*20)
    print("\nTPSR Equation: ", mcts_eq)
    print("\nTPSR Equation after Refine: ", mcts_eq_refine)
    print('#'*40)




if __name__ == '__main__':

    case = 1
    parser = get_parser()
    params = parser.parse_args()
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    params.debug = True

    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    equation_env = build_env(params)
    modules = build_modules(equation_env, params)
    if not params.cpu:
        assert torch.cuda.is_available()
    symbolicregression.utils.CUDA = not params.cpu
    trainer = Trainer(modules, equation_env, params)
    
    
    #Example of Data:
    # x0 = np.random.uniform(-2,2, 200)
    x0 = np.linspace(-2,2, 200)
    # y = (x0 **2) * np.sin(x0)
    y= (x0**2 ) * np.sin(5*x0) + np.exp(-0.5*x0)
    data = np.concatenate((x0.reshape(-1,1),y.reshape(-1,1)), axis=1)

    samples = {'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
    samples['x_to_fit'] = [data[:,:1]]
    samples['y_to_fit'] = [data[:,1].reshape(-1,1)]
    samples['x_to_pred'] = [data[:,:1]]
    samples['y_to_pred'] = [data[:,1].reshape(-1,1)]
    
    main(case, params, equation_env, samples)
    