import cvxpy as cp
from math import floor
from math import ceil
import pandas as pd
import numpy as np
import pip

if not ('SCIP' in cp.installed_solvers()):
  pip.main(['install', 'cvxpy[SCIP]'])

def prepare_items(dataset, groups, target_prop, score_label, r):
  """
  Given a dataset, construct a weakly fair ranking of `r` elements with respect to the specified groups (`groups`)
  and constraints (`target_prop`).
  """
  center = pd.DataFrame(columns=dataset.columns)
  group_items = []
  group_labels = {}
  for gr_id, group in enumerate(groups):
    q = ' & '.join(
        [f'{s_i}=="{v_i}"' if isinstance(v_i, str) else f'{s_i}=={v_i}'
        for s_i, v_i in group.items()]
        )
    res = dataset.query(q)
    group_items.append(res)
    for i in res.index:
      group_labels[i] = gr_id
  group_counts = [0]*len(groups)
  # get minimum repr for every group
  for i in range(len(groups)):
    center = pd.concat([center, group_items[i].iloc[:floor(r*target_prop[i])]], axis=0)
    group_counts[i] += floor(r*target_prop[i])
  free_ind = dataset.index[~dataset.index.isin(center.index)]
  # fill up free spots until r or maximum repr reached for every group
  for i in free_ind:
    if len(center) == r:
      break
    cand_group = group_labels[i]
    if group_counts[cand_group] < ceil(r*target_prop[cand_group]):
        center.loc[i] = dataset.loc[i]
        group_counts[cand_group]+=1
  return center.sort_values(by=score_label, ascending=False)


def construct_lp(
    center: pd.DataFrame,
    groups: list[dict],
    target_prop: list,
    score_label: str,
    r: int=None,
    noise_rng: np.random.Generator=None,
    noise_mean: float=0,
    noise_var: float=0):
  """
    Given a central distribution, a list of groups, a target proportion of candidates from each group,
    and a noise generator (for the noisy data experiment),
    return a CVXPY formulation of the ILP.
  """
  scores = center[score_label].to_numpy()
  k = len(scores)
  if r is None:
     r = k
  s = scores
  c = 1/np.log2(1+np.arange(1,r+1))
  x = cp.Variable((k,r), boolean=True)
  group_masks = []
  # mask elements belonging to each group
  for group in groups:
    q = ' & '.join(
                    [f'{s_i}=="{v_i}"' if isinstance(v_i, str) else f'{s_i}=={v_i}'
                     for s_i, v_i in group.items()]
                     )
    group_masks.append(np.isin(center.index, (np.array(center.query(q).index))))
  group_idx = [[i for i, in_group in enumerate(mask) if in_group] for mask in group_masks]
  constr1 = [cp.sum(x[i]) <= 1 for i in range(k)]
  constr2 = [cp.sum(x[:,j]) == 1 for j in range(r)]
  constraints = constr1 + constr2
  for l in range(1, r+1):
    for p, group_ind in zip(target_prop, group_idx):
      fl = floor(p*l)
      cl = ceil(p*l)
      if not (noise_rng is None):
        fl -= np.abs(noise_rng.normal(noise_mean, noise_var))
        cl += np.abs(noise_rng.normal(noise_mean, noise_var))
      constraints.append(cp.sum(x[group_ind,:l]) >= fl)
      constraints.append(cp.sum(x[group_ind,:l]) <= cl)
  problem = cp.Problem(cp.Maximize(c@(s@x)), constraints=constraints)
  return problem

#### Linear programming algorithm
def linprog_alg(
    items:pd.DataFrame,
    groups: list[dict],
    target_prop: list,
    score_label: str,
    r: int,
    verbose: bool=False,
    return_problem: bool = False,
    solver: str='GUROBI',
    noise_rng: np.random.Generator=None,
    noise_mean: float=0,
    noise_var: float=0,
    is_wfair: bool = False):
  """
    Given a ranking, list of groups and target proportion of candidates from each group, solve the ILP problem of fair reranking. 
  """
  if not (len(target_prop) == len(groups)):
    raise ValueError('Must specify proportions for all groups!')
  ### make center
  if is_wfair:
    center = items
  else:
    center = prepare_items(items, groups, target_prop, 'Credit amount',r)
  ###
  problem = construct_lp(center, groups, target_prop, score_label, r,
                         noise_rng,noise_mean,noise_var)
  problem.solve(verbose=verbose, solver=solver)#, QCPDual=0)#,MIPGap=1e-1,MIPGapAbs=1e-2, QCPDual=0)
  perm_matrix = list(problem.solution.primal_vars.values())[0].astype(int)
  new_index = center.index @ perm_matrix
  rerank = center.loc[new_index]
  # for column in center.columns:
  #    rerank[column] = pd.Series(np.array(center[column]) @ perm_matrix)
  if return_problem:
     return rerank, problem
  else:
     return rerank