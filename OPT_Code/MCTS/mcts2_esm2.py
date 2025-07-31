from ast import arg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import math
import pandas as pd
import random
import torch
import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
from tqdm import tqdm
from amp.utils import basic_model_serializer
import amp.data_utils.sequence as du_sequence
from collections import deque
from collections import defaultdict
import torch.nn.functional as F
import multiprocessing as mp
import heapq

# 加载ESM2预测模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("/geniusland/home/wanglijuan/sci_proj/facebookesm2_t33_650M_UR50D")
model = EsmForMaskedLM.from_pretrained("/geniusland/home/wanglijuan/sci_proj/facebookesm2_t33_650M_UR50D")
model = model.cuda()
model.eval()

# 加载打分器
bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/mic_classifier/')
mic_classifier_model = mic_classifier()

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']



class Node:
    def __init__(self, sequence, probability_matrix=None, prob=0, mutation_site=None, parent=None, grop_id=-1):# 突变点就是grop_id
        self.sequence = sequence
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.total_value = 0.0
        self.mutation_site = mutation_site# 这里记录的是关于它的父亲节点发生突变的位置
        self.prob = prob
        self.group_id = grop_id
        self.mutated_sites = set() if parent is None else parent.mutated_sites.copy()
        self.groups = None 
        if mutation_site is not None:
            self.mutated_sites.add(mutation_site)
        self.group_score = float('inf') if parent is None else parent.group_score
        self.probability_matrix = probability_matrix if probability_matrix is not None else None
        # self.groups = defaultdict(lambda: {'score': float('inf'), 'nodes': []})
        # self.groups[self.root.group_id]['nodes'].append(self.root)
    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):# 这个函数不符合规律  
        self.visits += 1
        # self.value += (reward - self.value) / self.visits# 反向传播形成新的分数
        self.total_value += reward
        # if self.parent:
        #     self.parent.update_group_score()

    def simulate(self):
        l = len(self.sequence)
        if l <= 5:
            k = 3
        elif l <= 10:
            k = 5
        else:
            k = 10
        mutations = []
        for i in range(k):# 循环进行多少次突变
            sequence = list(self.sequence)
            new_amino_acid = random.choice(AMINO_ACIDS)
            mutated_sequence = sequence[:]# 的sequence和mutated_sequence之间不共享同一个数据
            mutated_sequence[i] = new_amino_acid
            mutations.append(''.join(mutated_sequence))# 使用原始的序列进行k次突变，每一次都使用原始序列突变一个位点(假设是深拷贝的话)
        pad_seq = du_sequence.pad(du_sequence.to_one_hot(mutations))
        pred_amp = amp_classifier_model.predict(pad_seq)
        pred_mic = mic_classifier_model.predict(pad_seq)
        tol = 0
        for i in range(k):
            tol += pred_amp[i][0] + pred_mic[i][0]
        rs = tol / k
        return rs
    
    # 计算概率矩阵
    def calculate_probability_matrix(self):
        l = len(self.sequence)
        seqs = []
        for i in range(l):# 在序列的每个位置上都放上掩码，然后再放进seqs中
            x = list(self.sequence)
            x[i] = tokenizer.mask_token
            seqs.append(''.join(x))

        # 对序列进行批量处理
        inputs = tokenizer(seqs, return_tensors="pt", padding=True)
        inputs = inputs.to('cuda')
        rs = []
        with torch.no_grad():
            logits = model(**inputs).logits  # 获取模型的输出
            logits = logits.detach().cpu()# detach用于释放梯度数据，节省内存

            # 计算每个mask位置的概率
            for i, seq in enumerate(seqs):# 对于每一个序列
                # 找到当前序列中mask的索引位置
                # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                mask_token_index = (inputs.input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                # print(mask_token_index)
                # 对每个mask位置计算softmax概率
                for index in mask_token_index:#只有一个mask位置，所以只计算一次
                    predicted_token_logits = logits[i, index]
                    # print(logits.shape)
                    mutation_probs = F.softmax(predicted_token_logits[4:24]).cpu().numpy() 
                    # print(mutation_probs)
                rs.append(mutation_probs)                   
        return rs
    
        # 扩展步骤
    def expand(self):
        if self.probability_matrix is None:
            self.probability_matrix = self.calculate_probability_matrix()# 获取点位的突变概率  这个矩阵的第一列表示不同的位点，第二列表示不同的氨基酸
        L = len(self.sequence)
        self.groups = MCTS()
        for idx in range(L):
            if idx in self.mutated_sites:# 在已经突变过的位点上
                continue
            current_aa = self.sequence[idx]# 第一个未突变的位点
            for aa_idx, mutation in enumerate(vocab):# 
                if mutation == current_aa:# 使用其他19个氨基酸进行突变
                    continue
                s = list(self.sequence)
                s[idx] = mutation# 突变
                mutated_sequence = ''.join(s)
                new_node = Node(mutated_sequence, None, self.probability_matrix[idx][aa_idx], idx, self, idx)# 父亲为node    #porb是突变为第idx个位点，第aa_idx个氨基酸的概率
                self.add_child(new_node)
                self.groups.groups[idx]['nodes'].append(new_node)# idx位点的突变
        # # 随机选择一个组
        # random_group_id = random.choice(list(self.groups.keys()))
        # # 在该组内找到概率值最大的子节点
        # best_node = max(self.groups[random_group_id]['nodes'], key=lambda x: x.prob, default=None)# 在单个组内部prob最大的突变氨基酸
        # return best_node


    
    def backpropagate(self, score):
        node = self
        while node is not None:
            # if node.visits == 0:

            #     node.update(score)
            #     node = node.parent
            # else:
            #     node.update(score)
            #     node.groups.update_group_score()
            #     node = node.parent

            node.update(score)# 更新本身
            mutation_site = node.mutation_site# 得到mutation点位快速定位数组 
            node = node.parent# 找到父亲节点，以更新group
            if node:# 如果到了根节点了，就不做这个事
                node.groups.update_group_score(mutation_site)# 更新信息

                
            

    def get_best_sequence(self):
        # 初始化最佳节点为根节点
        best_node = self
        max_value = self.value

        # 使用队列来遍历所有节点
        node_queue = [self]# 这里使用节点进行操作
        while node_queue:# 从所有的节点中找出分数最高的作为我们的突变序列    这里从树中找节点的方法还不是很确定，需要我们看一遍树的样子之后再来检验
            current_node = node_queue.pop(0)
            if current_node.value > max_value:
                max_value = current_node.value
                best_node = current_node
            node_queue.extend(current_node.children)

        return best_node.sequence
    

    def reward(self):
        seq = []
        seq.append(self.sequence)
        pad_seq = du_sequence.pad(du_sequence.to_one_hot(seq))
        pred_amp = amp_classifier_model.predict(pad_seq)
        pred_mic = mic_classifier_model.predict(pad_seq)
        return pred_mic[0][0] + pred_amp[0][0]
    


class MCTS:# 看看能不能解决所有的组都共用同一个groups的问题。
    def __init__(self):
        self.groups = defaultdict(lambda: {'score': float('inf'), 'nodes': []})
        # self.groups[self.root.group_id]['nodes'].append(self.root)# 根节点放进组-1中 默认的分数是无穷


    def select_best_child_group(self):
        # 选择具有最高得分的组
        best_group_id = max(self.groups, key=lambda x: self.groups[x]['score'])
        return self.groups[best_group_id]
    
    def select_best_child_node(self, group):
        # 在指定组内选择UCB值最高的子节点
        count = 1# 已经被访问过的节点+1
        length = len(group['nodes'][0].sequence)
        for node in group['nodes']:
            if node.visits!=0:#如果说该分组的节点被访问过，那我们就跳过这个节点   假设有2个节点被访问过，我们就count+=2，count变成3，我们找第三大
                count+=1
        if count <= length:# 如果该组中还有节点没有被访问过的，那我们使用esm2的prob对所有的节点进行排序，返回当前最大概率的节点     当其为length+1，代表所有节点都被访问过
            

            best_child = heapq.nlargest(count, group['nodes'], key=lambda x : x.prob)[-1]
        else:
            best_child = heapq.nlargest(1, group['nodes'], key=lambda x: self.ucb_value(x))[-1]
        return best_child
    


# 看整个过程
    def ucb_value(self, node, c_param=np.sqrt(2)):
        if node.visits == 0:
            return float('inf')
        if node.parent is None:
            parent_visits = 1  # 根节点没有父节点 将其visits值设置为1 其实如果不设置，ucb_value在上一个if语句中会自动返回无穷
        else:
            parent_visits = node.parent.visits
        # print(node.prob)
        # print("++++++++++++")
        # print((node.total_value / node.visits))
        # print(c_param * np.sqrt((2 * np.log(parent_visits) / node.visits)))

        return (node.total_value / node.visits) + c_param * np.sqrt((2 * np.log(parent_visits) / node.visits))# 这里没有使用总探索次数，而是使用了parent_visits作为替代

    # def run(self, iterations):
        
    #     for _ in range(iterations):
    #         root = self.root
    #         while root.children:# 使用group进行寻找，快速找best_child            
    #             best_group = self.select_best_group()
    #             root = self.select_best_child_node(best_group)
    #         node = self.expand(root)# 扩展了已经探索过的节点之后，我们在所有生成的节点中随机选取一个组，然后计算其最有可能的氨基酸突变
    #         score = self.simulate(node)# 计算这个节点的分数
    #         self.backpropagate(node, score)
    def update_group_score(self, mutation_site):
        # for group_id, group_info in self.groups.items():# 对于其每一个子节点进行更新
        #     group_info['score'] = sum(self.ucb_value(node) for node in group_info['nodes'] if node.visits > 0)
        self.groups[mutation_site]['score'] = max(self.ucb_value(node) for node in self.groups[mutation_site]['nodes'] if node.visits > 0)# 用最高分进行更新

    
    
    # def get_best_sequence(self):
    #     best_node = max((node for node in self.root.children if node.visits > 0), key=lambda n: n.value, default=self.root)
    #     return best_node.sequence
def process_sequence(sequence):
    root_node = Node(sequence, )
    root_groups = MCTS()# 这个是最起始时候的group id为-1
    root_groups.groups[-1]['nodes'].append(root_node)
    for _ in range(500):
        node = root_node
        while node.groups:# 不断循环 可以找到最佳分数的叶子节点
            group = node.groups.select_best_child_group()# 我们得到这个root的最佳分组
            node = node.groups.select_best_child_node(group)# group中的最佳分数节点
        
        # 在这里，我想通过减枝的方法，不但使用前五高的分数的节点进行计算，来代表这个分组的数值，  分组第一次更新分数的时候很重要，涉及到以后会不会被选择
        # ，如果说不小心选了一个错误的氨基酸，会导致这个位点始终不被突变更新   因此我们对分组内多个节点进行simulate来计算分组的分数
        # 这里找到了叶子节点
        # 对第一个节点更新
        best_node = node
        if best_node.visits == 0:# 如果说一个节点的值为0，可以说明这个组中还有其他的节点还没有被使用
            score = best_node.simulate()
            node_score = best_node.reward()
            best_node.value = node_score
            best_node.backpropagate(score)
        if best_node.visits != 0:
            best_node.expand()
    best_sequence = root_node.get_best_sequence()
    return best_sequence
if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    data = pd.read_csv('/geniusland/home/wanglijuan/sci_proj/GA_opt/srcdata/case1.csv')
    seqs = data['Sequence']

    with mp.Pool(processes=4) as pool:
        rs = list(tqdm(pool.imap(process_sequence, seqs), total=len(seqs)))
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(rs))
    pred_amp = amp_classifier_model.predict(pad_seq)
    pred_mic = mic_classifier_model.predict(pad_seq)
    amp = pred_amp.flatten()
    mic = pred_mic.flatten()

    case1 = pd.read_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/src_datasets/case1.csv')
    c1_amp = case1['amp'].tolist()
    c1_mic = case1['mic'].tolist()
    data = {
        'Case1_amp': c1_amp,
        'MCTS_ESM2_amp': amp,
        'Case1_mic': c1_mic,
        'MCTS_ESM2_mic': mic,
    }
    df0 = pd.DataFrame(rs, columns=['Sequence'])
    df = pd.DataFrame(data)
    df.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case1/MCTS_esm2/noprob_v1_pred_500.csv', index=False)
    df0.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case1/MCTS_esm2/noprob_v1_seq_500.csv', index=False)


