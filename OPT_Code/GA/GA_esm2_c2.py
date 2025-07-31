import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
from amp.utils import basic_model_serializer
import amp.data_utils.sequence as du_sequence
from decimal import Decimal
import Levenshtein
import matplotlib.pyplot as plt

# 加载预测模型
tokenizer = AutoTokenizer.from_pretrained("/geniusland/home/wanglijuan/sci_proj/facebookesm2_t33_650M_UR50D")
model = EsmForMaskedLM.from_pretrained("/geniusland/home/wanglijuan/sci_proj/facebookesm2_t33_650M_UR50D")
model = model.cuda()
model.eval()

# 加载打分模型 
bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/mic_classifier/')
mic_classifier_model = mic_classifier() 

# 从CSV文件读取抗菌肽数据集
def load_antimicrobial_peptides_from_csv(file_path):
    df = pd.read_csv(file_path)  # 读取CSV文件
    peptides = df['Sequence'].tolist()  # 将"Sequence"列的值存为列表
    return peptides

# 使用示例
antimicrobial_peptides = load_antimicrobial_peptides_from_csv('/geniusland/home/wanglijuan/sci_proj/GA_opt/srcdata/case3.csv')
# print(antimicrobial_peptides)
# 定义适应度和个体类
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化适应度
creator.create("Individual", list, fitness=creator.FitnessMax)

# 使用抗菌肽数据集生成个体
def generate_individual_from_dataset(dataset):
    return creator.Individual(list(random.choice(dataset)))

# 适应度函数：返回抗菌活性减去毒性作为适应度
def evaluate_sequence(sequence):
    # print('1111111')         
    # print(sequence)
    seq = ''.join(sequence)
    tmp = []
    tmp.append(seq)
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(tmp))
    pred_amp = amp_classifier_model.predict(pad_seq)
    pred_mic = mic_classifier_model.predict(pad_seq)
    # amp = pred_amp[0][0]
    mic = pred_mic[0][0] + pred_amp[0][0]
    
    return (mic , )  # 适应度：抗菌活性 - 0.5 * 毒性

# 自定义变异函数，将整数变异替换为字符替换
def mutate_individual(individual):
    for i in range(len(individual)):
        if random.random() < 0.5:  # 0.1 是变异概率
            seq = ''.join(individual)
            src_token = tokenizer(seq, return_tensors="pt")
            testt = src_token['input_ids']
            testt[0][i+1] = 32
            src_token['input_ids'] = testt

            inputs = src_token
            inputs = inputs.to('cuda')
            with torch.no_grad():
                logits = model(**inputs).logits  #[batch seq_len,33]
                logits = logits.detach().cpu()
            mask_token_index = (inputs.to('cpu').input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]  # 找到序列中mask的id
            predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
            individual[i] = predicted_token[0]
    return individual,

# def mutate_individual(individual):
#     for i in range(len(individual)):
#         if random.random() < 0.1:  # 0.1 是变异概率
#             individual[i] = random.choice('ACDEFGHIKLMNPQRSTVWY')
#     return individual,





# 注册遗传算法的主要工具
toolbox = base.Toolbox()
toolbox.register("individual", generate_individual_from_dataset, antimicrobial_peptides)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)  # 双点交叉
# toolbox.register("mate", tools.cxMessyOnePoint) #混乱单点交叉
# toolbox.register("mate", tools.cxUniform) # 均匀交叉
# toolbox.register("mate", tools.cxPartialyMatched) # 部分匹配交叉
# toolbox.register("mate", tools.cxBlend) # 混合交叉
# toolbox.register("mutate", tools.mutUniformInt, low=0, up=19, indpb=0.1)  # 变异
# 注册自定义的变异函数
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择
# toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate_sequence)


# 初始化种群，使用自己的抗菌肽数据集
population = toolbox.population(n=476)  # 种群大小为300
# 用于存储每一代两个目标的适应度信息
max_score1_values = []
avg_score1_values = []
max_score2_values = []
avg_score2_values = []
# 定义遗传算法的参数
NGEN = 20  # 迭代次数
CXPB = 0.9
MUTPB = 0.1

for gen in range(NGEN):
    print(f"-- Generation {gen} --")
    # if gen <= 5:
    #     CXPB = 0.5
    #     MUTPB = 0.7
    # else:
    #     CXPB = 0.4
    #     MUTPB = 0.5
    # 选择下一代个体
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    
    # 应用交叉操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    # 应用变异操作
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring
    
    
rs = []
for s in population:
    rs.append(''.join(s))

pad_seq = du_sequence.pad(du_sequence.to_one_hot(rs))
pred_amp = amp_classifier_model.predict(pad_seq)
pred_mic = mic_classifier_model.predict(pad_seq)
amp = pred_amp.flatten()
mic = pred_mic.flatten()
data = pd.read_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/src_datasets/case3.csv')
c_amp = data['amp'].tolist()
c_mic = data['mic'].tolist()   

data = {
    'Case3_amp': c_amp,
    'GA_ESM2_amp': amp,
    'Case3_mic': c_mic,
    'GA_ESM2_mic': mic
}
df = pd.DataFrame(data)
df0 = pd.DataFrame(rs, columns=['Sequence'])
df.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case3/GA_esm2/case3_pred_u.csv', index=False)
df0.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case3/GA_esm2/case3_seq_u.csv', index=False)
