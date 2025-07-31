import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import pandas as pd
import math
import random
import hashlib
import logging
from amp.utils import basic_model_serializer
import amp.data_utils.sequence as du_sequence
from tqdm import tqdm
import multiprocessing as mp


bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/mic_classifier/')
mic_classifier_model = mic_classifier() 

# 假设氨基酸序列使用标准20氨基酸单字母代码
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

class State:
    def __init__(self, sequence, mutations=None, turn=0):
        self.sequence = sequence
        self.mutations = mutations if mutations is not None else set()
        self.turn = turn

    def next_states(self):
        # 生成当前序列的所有可能的单点突变状态
        # print("Current mutations:", self.mutations)
        L = len(self.sequence)
        next_states = []
        for i in range(L):
            if i not in self.mutations:  # 仅在未突变的位点进行突变
                original_aa = self.sequence[i]
                for aa in AMINO_ACIDS:
                    if aa != original_aa:
                        mutated_sequence = list(self.sequence)
                        mutated_sequence[i] = aa
                        mutated_sequence = ''.join(mutated_sequence)
                        new_mutations = self.mutations.union({i})
                        next_states.append(State(mutated_sequence, new_mutations, self.turn + 1))
        return next_states

    def terminal(self):
        return len(self.mutations) == len(self.sequence)  # 所有位点都突变过

    def reward(self):
        seq = []
        seq.append(self.sequence)
        pad_seq = du_sequence.pad(du_sequence.to_one_hot(seq))
        pred_amp = amp_classifier_model.predict(pad_seq)
        pred_mic = mic_classifier_model.predict(pad_seq)
        return pred_mic[0][0] + pred_amp[0][0]

    def __hash__(self):
        return hash((self.sequence, tuple(self.mutations)))

    def __eq__(self, other):
        return (self.sequence, self.mutations) == (other.sequence, other.mutations)

    def __repr__(self):
        return f"Sequence: {self.sequence}; Mutations: {self.mutations}"

class Node:
    total_nodes = 0
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        Node.total_nodes += 1

    def add_child(self, state):
        child = Node(state, self)
        self.children.append(child)
        return child

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        L = len(self.state.sequence)
        # print(L - len(self.state.mutations))
        return len(self.children) >= 19 * (L - len(self.state.mutations))

    def best_child(self, scalar=2/math.sqrt(2)):
        best_score = -float('inf')
        best_children = []
        for child in self.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(2.0 * math.log(self.visits) / child.visits)
            score = exploit + scalar * explore
            if score > best_score:
                best_children = [child]
                best_score = score
            elif score == best_score:
                best_children.append(child)
        return random.choice(best_children) if best_children else None

    def __repr__(self):
        return f"Node({self.state}; visits: {self.visits}; reward: {self.reward})"

def tree_policy(node):
    # 选择或扩展
    if not node.fully_expanded():
        return node
    while node.fully_expanded():
        node = node.best_child()
    return node

def expand(node):
    next_states = node.state.next_states()
    next_state = random.choice(next_states)  # 随机选择一个未尝试的突变
    return node.add_child(next_state)

def default_policy(state):
    l = len(state.sequence)
    if l <= 5:
        k = 3
    elif l <=10:
        k = 5
    else:
        k = 10
    mutations = []
    for i in range(k):
        sequence = list(state.sequence)
        new_amino_acid = random.choice(AMINO_ACIDS)
        mutated_sequence = sequence[:]
        mutated_sequence[i] = new_amino_acid
        mutations.append(''.join(mutated_sequence))
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(mutations))
    pred_amp = amp_classifier_model.predict(pad_seq)
    pred_mic = mic_classifier_model.predict(pad_seq)
    tol = 0
    for i in range(k):
        tol += pred_amp[i][0] + pred_mic[i][0]
    rs = tol / k
    return rs

def uct_search(root, iterations):
    for _ in range(iterations):
        leaf = tree_policy(root)
        leaf = expand(leaf)
        reward = default_policy(leaf.state)
        backup(leaf, reward)
    return root.best_child(0)

def backup(node, reward):
    while node:
        node.visits += 1
        node.reward += reward
        node = node.parent

def best_child_by_average_reward(node):
    best_reward = -float('inf')
    best_child = None
    for child in node.children:
        average_reward = child.reward / child.visits
        if average_reward > best_reward:
            best_reward = average_reward
            best_child = child
    return best_child

def find_best_node_by_average_reward(node):
    if not node.children:  # 如果没有子节点，返回当前节点
        return node
    
    best_node = node
    best_average_reward = node.reward / node.visits if node.visits > 0 else -float('inf')

    # 递归检查所有子节点
    for child in node.children:
        candidate_node = find_best_node_by_average_reward(child)
        candidate_average_reward = candidate_node.reward / candidate_node.visits if candidate_node.visits > 0 else -float('inf')

        # 更新最优节点
        if candidate_average_reward > best_average_reward:
            best_average_reward = candidate_average_reward
            best_node = candidate_node

    return best_node

def collect_all_nodes_scores(node):
    scores = [(node.state.sequence, node.state.reward())]
    for child in node.children:
        scores.extend(collect_all_nodes_scores(child))
    return scores
def find_highest_scored_sequence(node):
    all_scores = collect_all_nodes_scores(node)
    # 根据得分排序并选择得分最高的序列
    best_sequence = max(all_scores, key=lambda x: x[1])
    return best_sequence

def process_sequence(s):
    root = Node(State(s))
    best_node = uct_search(root, 1000)
    best_sequence = find_highest_scored_sequence(root)
    return best_sequence[0]


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    data = pd.read_csv('/geniusland/home/wanglijuan/sci_proj/GA_opt/srcdata/case1.csv')
    seqs = data['Sequence']
   
    with mp.Pool(processes=3) as pool:
        rs = list(tqdm(pool.imap(process_sequence, seqs), total=len(seqs)))
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(rs))
    pred_amp = amp_classifier_model.predict(pad_seq)
    pred_mic = mic_classifier_model.predict(pad_seq)
    amp = pred_amp.flatten()
    mic = pred_mic.flatten()
    case1 = pd.read_csv('/geniusland/home/wanglijuan/sci_proj/GA_opt/srcdata/case1.csv')
    c1_amp = case1['amp'].tolist()
    c1_mic = case1['mic'].tolist()
    data = {
        'Case1_amp': c1_amp,
        'Case1_mic': c1_mic,
        'MCTS_amp': amp,
        'MCTS_mic': mic,
    }
    df0 = pd.DataFrame(rs, columns=['Sequence'])
    df = pd.DataFrame(data)
    df.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case1/MCTS/pred_1000.csv', index=False)
    df0.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case1/MCTS/seq_1000.csv', index=False)

