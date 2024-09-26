import numpy as np
import pandas as pd 

def infeasible_index(ranking: pd.DataFrame,
                     groups: list[dict] | list,
                     target_prop: list,
                     r=None,
                     control_upper = True,
                     verbose=False):
    ii = 0
    r_ = len(ranking)+1 if r is None else r
    viol = np.zeros(r_)
    # find elements from that group
    group_masks = []
    if len(groups) == len(ranking):
      group_codes = pd.Series(data=groups, index=ranking.index)
    else:
      group_codes = pd.Series(index=ranking.index, dtype=float)
    
    for i_gr, group in enumerate(groups):
        if len(groups) == len(ranking):
          group_codes = pd.Series(data=groups, index=ranking.index)
        else:
          q = ' & '.join(
                          [f'{s_i}=="{v_i}"' if isinstance(v_i, str) else f'{s_i}=={v_i}'
                          for s_i, v_i in group.items()]
                          )
          group_masks.append(np.isin(ranking.index, (np.array(ranking.query(q).index))))
          group_codes.loc[group_masks[-1]] = i_gr

        for k in range(1, r_):
          count_ai = np.sum(group_codes.iloc[:k] == i_gr)
          lower = np.floor(k*target_prop[i_gr])
          upper = np.ceil(k*target_prop[i_gr])
          if count_ai < lower or (count_ai > upper and control_upper):
            if not viol[k-1]:
              ii+=1
            if verbose and not viol[k-1]:
              print(f'Violation on k={k} by group {i_gr}: L: {lower} Act: {count_ai} U: {upper}')
            viol[k-1]=True
    return ii

def dcg(scores):
  # discounted cumulative gain
  scores = np.array(scores, dtype=float)
  logs = np.log2(np.arange(1, len(scores)+1)+1)
  z = np.sum(scores/logs)
  return z

def ndcg(scores, init_scores, sorted=False):
  if sorted:
    return dcg(scores) / dcg(init_scores)
  else:
    return dcg(scores) / dcg(np.sort(init_scores[:len(scores)])[::-1])

def KT(sigma, pi):
  """
  Compute the kendall tau distance of two permutations.
  """
  length = len(pi)
  kt = 0
  for i in range(length):
    for j in range(i+1, length):
      if (sigma[i] < sigma[j] and pi[i] > pi[j]) or \
       (sigma[i] > sigma[j] and pi[i] < pi[j]):
       kt += 1
  return kt