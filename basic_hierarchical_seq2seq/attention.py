# """ attention functions """
# from torch.nn import functional as F
# import torch

# def dot_attention_score(key, query, converage, projection):
#     if converage is None:
#         result = projection(key + query).view(-1, key.size()[1]).unsqueeze(-2)
#     else:
#         result = projection(key + query+ converage).view(-1, key.size()[1]).unsqueeze(-2)
#     return result

# def prob_normalize(score, mask):
#     """ [(...), T]
#     user should handle mask shape"""
#     score = score.masked_fill(mask == 0, -1e18)  #在mask中score为0的地发，填充为-1e18？
#     norm_score = F.softmax(score, dim=-1)
#     return norm_score

# def attention_aggregate(value, score):
#     """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
#     output = score.matmul(value)
#     return output


# def step_attention(query, key, value, converage, projection, attn_wc, mem_mask=None):
#     """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
#     #应该就是说原来那个矩阵里面存在为0的地方，也就是说没有target输出的地方，给他填一个数，再做softmax？可是为啥呢
#     if converage is not None:
#         converage_value = attn_wc(converage.unsqueeze(-1))
#     else:
#         converage_value = None
#     score = dot_attention_score(key, query.unsqueeze(-2), converage_value, projection) #34,1,81
#     if mem_mask is None:
#         norm_score = F.softmax(score, dim=-1)
#     else:
#         norm_score = prob_normalize(score, mem_mask)  #也就是给每个位置的词语一个weight
#     output = attention_aggregate(value, norm_score)
#     if converage is None:
#         min_attn = norm_score
#         min_attn = torch.sum(min_attn.squeeze(-2), 1)
#     else:
#         min_attn = torch.min(norm_score.squeeze(-2), converage)
#         min_attn = torch.sum(min_attn.squeeze(-2)).unsqueeze(-1)
#     #min_attn = torch.sum(min_attn.squeeze(-2), 1)

#     #计算converage 和score最小值加和的值，返回
#     return output.squeeze(-2), norm_score.squeeze(-2), min_attn


""" attention functions """
from torch.nn import functional as F


def dot_attention_score(key, query):
    """[B, Tk, D], [(Bs), B, Tq, D] -> [(Bs), B, Tq, Tk]"""
    return query.matmul(key.transpose(1, 2))

def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e18)  #在mask中score为0的地发，填充为-1e18？
    norm_score = F.softmax(score, dim=-1)
    return norm_score

def attention_aggregate(value, score):
    """[B, Tv, D], [(Bs), B, Tq, Tv] -> [(Bs), B, Tq, D]"""
    output = score.matmul(value)
    return output


def step_attention(query, key, value, mem_mask=None):
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    #应该就是说原来那个矩阵里面存在为0的地方，也就是说没有target输出的地方，给他填一个数，再做softmax？可是为啥呢
    score = dot_attention_score(key, query.unsqueeze(-2)) #34,1,81
    if mem_mask is None:
        norm_score = F.softmax(score, dim=-1)
    else:
        norm_score = prob_normalize(score, mem_mask)  #也就是给每个位置的词语一个weight
    output = attention_aggregate(value, norm_score)
    return output.squeeze(-2), norm_score.squeeze(-2)
