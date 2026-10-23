import numpy as np
import torch
import json
import random
import operator
from pandas import DataFrame
from deap.gp import Primitive
from inspect import isclass
from torch.utils.data import TensorDataset, Dataset, DataLoader
import scipy.special
from pycd.train.trainer import  EarlyStopping  #新增早停
import os
import operator
from deap import base, creator, tools, gp, algorithms
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def dot(x, y):
    if type(x) is np.ndarray and type(y) is np.ndarray:
        return np.sum(x * y, dtype=np.float64)
    else:
        return (x * y).sum(dim=1).unsqueeze(1)


def sigmoid(x):
    if type(x) is np.ndarray or type(x) is np.float64 or type(x) is np.float32:
        # to avoid overflow
        return scipy.special.expit(x)
    else:
        return torch.sigmoid(x)


def tanh(x):
    if type(x) is np.ndarray or type(x) is np.float64 or type(x) is np.float32:
        return np.tanh(x)
    else:
        return torch.tanh(x)


class StudentDataSet(Dataset):
    def __init__(self, loaded_data):
        """
        This class is designed for transforming loaded_data from np.ndarray to Dataset.
        """
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def exam(test_set, proficiency, difficulty, discrimination, interaction_func):
    """
    Simulate the interaction between students and questions.

    :param test_set: test set excluding the training data
    :param proficiency: proficiency level of each student
    :param difficulty: difficulty of each knowledge attributes in each questions
    :param discrimination: discrimination of questions
    :param interaction_func: compiled interaction function from genetic programming
    :return: prediction of response `y_pred` and true labels `y_true`
    """
    y_pred, y_true = [], []
    for batch_data in test_set:
        student_id_batch, question_batch, q_matrix_batch, y = list(map(np.array, batch_data))
        for student_id, question, q_matrix in zip(student_id_batch, question_batch, q_matrix_batch):
            # print(student_id, question, q_matrix)
            p = sigmoid(proficiency[student_id])
            dk = sigmoid(difficulty[question])
            de = sigmoid(discrimination[question])
            pred = sigmoid(interaction_func(de, p - dk, q_matrix)).item()
            y_pred.append(pred)
        y_true.extend(y.tolist())
    y_pred = np.array(y_pred)
    y_pred = y_pred.tolist()
    return y_pred, y_true


def transform(student_id, question, y, q_matrix=None):
    """
    Transform data to match the input of parameter optimization

    :return: torch.DataLoader(batch_size=32)
    """
    if q_matrix is None:
        dataset = TensorDataset(torch.tensor(student_id, dtype=torch.int64) - 1,
                                torch.tensor(question, dtype=torch.int64) - 1,
                                torch.tensor(y, dtype=torch.float32))
    else:
        q_matrix_line = q_matrix[question - 1]
        dataset = TensorDataset(torch.tensor(student_id, dtype=torch.int64) - 1,
                                torch.tensor(question, dtype=torch.int64) - 1,
                                q_matrix_line,
                                torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=32)


def mut_uniform_with_pruning(individual, pset, pruning=0.5):
    rand = np.random.uniform(0, 1)
    if rand < pruning:
        # pruning tree
        # We don't want to "shrink" the tree too much
        if len(individual) < 15 or individual.height <= 5:
            return individual,

        iprims = []
        for i, node in enumerate(individual[1:], 1):
            if isinstance(node, Primitive) and node.ret in node.args:
                iprims.append((i, node))

        if len(iprims) != 0:
            index, prim = random.choice(iprims)
            arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
            rindex = index + 1
            for _ in range(arg_idx + 1):
                rslice = individual.searchSubtree(rindex)
                subtree = individual[rslice]
                rindex += len(subtree)

            slice_ = individual.searchSubtree(index)
            individual[slice_] = subtree
    else:
        index = random.randrange(len(individual))
        node = individual[index]
        slice_ = individual.searchSubtree(index)
        choice = random.choice

        # As we want to keep the current node as children of the new one,
        # it must accept the return value of the current node
        primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

        if len(primitives) == 0:
            return individual,

        new_node = choice(primitives)
        new_subtree = [None] * len(new_node.args)
        position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

        for i, arg_type in enumerate(new_node.args):
            if i != position:
                term = choice(pset.terminals[arg_type])
                if isclass(term):
                    term = term()
                new_subtree[i] = term

        new_subtree[position:position + 1] = individual[slice_]
        new_subtree.insert(0, new_node)
        individual[slice_] = new_subtree

    return individual,


def sel_random(individuals, k):
    candidates = individuals
    return [random.choice(candidates) for i in range(k)]


def sel_tournament(individuals, k, tournament_size, fit_attr="fitness"):
    chosen = []
    for i in range(k):
        aspirants = sel_random(individuals, tournament_size)
        chosen.append(max(aspirants, key=operator.attrgetter(fit_attr)))
    return chosen


def init_interaction_function(discrimination, proficiency, q_matrix_line):
    if type(proficiency) is np.ndarray:
        return discrimination * np.sum(proficiency * q_matrix_line)
    else:
        return discrimination * (proficiency * q_matrix_line).sum(dim=1).unsqueeze(1)



# from .interaction import GeneticInteractionFunc

# interaction文件部分



# import SCDM.operators as operators  # not build-in type!
# from .eval import accuracy, area_under_curve, f1_score
# from .utility import mut_uniform_with_pruning, sel_tournament, exam, init_interaction_function

def log_metrics(metrics, wandb_instance):
    """记录指标到wandb（如果启用）"""
    if wandb_instance:
            wandb_instance.log(metrics)


class InteractionFunc:
    def __init__(self, train_set):
        self.train_set = train_set
        self.proficiency = None
        self.difficulty = None
        self.discrimination = None

        # f(discrimination, proficiency - difficulty, q_matrix_line)
        self.input_type = [np.float64, np.ndarray, np.ndarray]
        self.output_type = np.float64

        # construction set
        self.primitive_set = gp.PrimitiveSetTyped("main", self.input_type, self.output_type)
        self.primitive_set_init()

        # gp toolbox configuration
        self.toolbox = base.Toolbox()
        self.toolbox_init()

        # gp multi statistics configuration
        self.multi_statistics = tools.MultiStatistics(
            AUC=tools.Statistics(lambda ind: ind.fitness.values[0]),
            accuracy=tools.Statistics(lambda ind: ind.fitness.values[1]),
        )

        self.multi_statistics.register("min", np.min)
        self.multi_statistics.register("max", np.max)

        # other settings
        self.population = None
        self.hof = None

    def primitive_set_init(self):
        # including all necessary base functions (meet monotonicity assumption)
        self.primitive_set.addPrimitive(add, [np.ndarray, np.ndarray], np.ndarray)
        self.primitive_set.addPrimitive(add, [np.ndarray, np.float64], np.ndarray)
        self.primitive_set.addPrimitive(add, [np.float64, np.float64], np.float64)
        self.primitive_set.addPrimitive(mul, [np.ndarray, np.ndarray], np.ndarray)
        self.primitive_set.addPrimitive(mul, [np.ndarray, np.float64], np.ndarray)
        self.primitive_set.addPrimitive(mul, [np.float64, np.float64], np.float64)
        self.primitive_set.addPrimitive(dot, [np.ndarray, np.ndarray], np.float64)
        self.primitive_set.addPrimitive(tanh, [np.ndarray], np.ndarray)
        # rename arguments
        # De: discrimination
        # PDk: proficiency_level - difficulty
        # Q: Q-matrix
        self.primitive_set.renameArguments(ARG0="De", ARG1="PDk", ARG2="Q")

    def toolbox_init(self):
        # register all genetic operations
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.primitive_set, min_=5, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.primitive_set)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", sel_tournament, tournament_size=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", mut_uniform_with_pruning, pset=self.primitive_set)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def evaluate(self, individual):
        currentInteractionFunc = self.toolbox.compile(expr=individual)
        y_pred, y_true = exam(self.train_set,
                              self.proficiency,
                              self.difficulty,
                              self.discrimination,
                              currentInteractionFunc,)

        acc = accuracy(y_pred, y_true)
        auc = area_under_curve(y_pred, y_true)
        return auc, acc,

    def train(self, population_size=200, ngen=10, cxpb=0.5, mutpb=0.1):

        self.population = self.toolbox.population(n=population_size)  #定义群体
        # create hall of fame to record best individual
        if self.hof is None:
            self.hof = tools.HallOfFame(maxsize=1)
        self.population, _ = algorithms.eaSimple(self.population, self.toolbox,
                                                 cxpb, mutpb, ngen,
                                                 stats=self.multi_statistics,
                                                 halloffame=self.hof, verbose=False) # 更新群体

    def unpack(self, is_compiled=False):
        if self.hof:
            if is_compiled:
                return self.toolbox.compile(expr=self.hof.items[0])
            else:
                return self.hof.items[0]
        else:
            return init_interaction_function

    def update(self, proficiency, difficulty, discrimination):
        self.proficiency = proficiency.copy()
        self.difficulty = difficulty.copy()
        self.discrimination = discrimination.copy()


class GeneticInteractionFunc:
    def __init__(self, train_set, train_size):
        # genetic programming and algorithm init
        creator.create("fitness_if", base.Fitness, weights=(1.0, 1.0))
        creator.create("individual", gp.PrimitiveTree, fitness=creator.fitness_if)

        self.train_size = train_size

        self.proficiency = None
        self.difficulty = None
        self.discrimination = None

        self.train_set = train_set
        self.interaction = InteractionFunc(train_set)

        self.interaction_funcs = []
        self.interaction_funcs_string = []

    def __str__(self):
        if len(self.interaction_funcs) != 0:
            return self.interaction_funcs_string[0]
        else:
            return "default"

    def evaluation(self, test_data, function) -> tuple:

        if function == None :
            current_interaction_func = self.function()
        else:
            current_interaction_func = function
        # print(current_interaction_func)
        prediction, truth = exam(test_data,
                                 self.proficiency,
                                 self.difficulty,
                                 self.discrimination,
                                 current_interaction_func,)

        acc = accuracy(prediction, truth)
        auc = area_under_curve(prediction, truth)
        f1 = f1_score(prediction, truth)
        rmse = metrics.mean_squared_error(truth, prediction, squared=False)


        return acc, auc, f1, rmse, current_interaction_func
 
    def train(self, population_size, ngen, cxpb, mutpb):
        print("Genetic programming search")
        interaction_funcs = []
        interaction_funcs_string = []
        self.interaction.train(population_size, ngen, cxpb, mutpb)
        interaction_funcs.append(self.interaction.unpack(is_compiled=True))
        interaction_funcs_string.append(str(self.interaction.unpack()))
        self.interaction_funcs = interaction_funcs
        self.interaction_funcs_string = interaction_funcs_string
        fun_str = str(self) #保存交互函数字符串
        print("Final Function:", str(self))
        return fun_str

    def function(self):
        if len(self.interaction_funcs) != 0:
            def final_function(discrimination, proficiency_level, q_matrix):
                return self.interaction_funcs[0](discrimination, proficiency_level, q_matrix)
            return final_function
        else:
            return init_interaction_function

    def update(self, proficiency, difficulty, discrimination):
        self.proficiency = proficiency.copy()
        self.difficulty = difficulty.copy()
        self.discrimination = discrimination.copy()
        self.interaction.update(proficiency, difficulty, discrimination)


# from .parameter import Parameter
# parameter文件部分



# from .eval import accuracy, area_under_curve, f1_score
# from .utility import init_interaction_function


class ComputeIF(nn.Module):  # 这里进行向量嵌入
    def __init__(self,
                 student_number,
                 question_number,
                 knowledge_number):
        super(ComputeIF, self).__init__()
        self.student_emb = nn.Embedding(student_number, knowledge_number)
        self.difficulty = nn.Embedding(question_number, knowledge_number)
        self.discrimination = nn.Embedding(question_number, 1)

        # initialize
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_id, question, q_matrix_line, interaction_func):
        proficiency_level = torch.sigmoid(self.student_emb(student_id))
        difficulty = torch.sigmoid(self.difficulty(question))
        discrimination = torch.sigmoid(self.discrimination(question))

        input_x = interaction_func(discrimination, proficiency_level - difficulty, q_matrix_line)
        output = torch.sigmoid(input_x)

        return output.view(-1)


class Parameter:
    def __init__(self,
                 student_number: int,
                 question_number: int,
                 knowledge_number: int,):
        self.net = ComputeIF(student_number, question_number, knowledge_number)
        self.student_number = student_number
        self.question_number = question_number
        self.knowledge_number = knowledge_number
        self.interaction_function = init_interaction_function
        self.interaction_function_string = "initial interaction function"

    def train(self, train_set, test_set, lr, epochs, wandb_instance, device="cuda", init=True):
        # initialize
        if init:
            for name, param in self.net.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
        # 参数初始化
        self.net = self.net.to(device)
        self.net.train()
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        #学习率调度器定义
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        #早停设置
        early_stop = EarlyStopping(patience=5, mode='max')

        # 数据训练
        for epoch_i in range(1, epochs+1):
            epoch_losses = []
            for batch_data in tqdm(train_set, "Train"):
                student_id, question, q_matrix_line, y = batch_data
                student_id: torch.Tensor = student_id.to(device)
                question: torch.Tensor = question.to(device)
                q_matrix_line: torch.Tensor = q_matrix_line.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.net(student_id,
                                                question,
                                                q_matrix_line,
                                                self.interaction_function)  #实际用的网络结构是self.interaction_function 即为init_interaction_function，这个函数就是向量之间作了点乘
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())
            acc, auc, f1 = self.evaluate(test_set, self.interaction_function)
            loss = sum(epoch_losses) / len(epoch_losses)
            if scheduler:
                try:
                    scheduler.step(loss)
                except TypeError:
                    scheduler.step()

            if wandb_instance:
                log_metrics({
                    'parameter_train_epoch': epoch_i
                }, wandb_instance)
            
            print(f"Parameter train: [Epoch {epoch_i}] Train Loss: {loss:.4f}, Val Metric: {auc}")

            # 早停判断
            if early_stop is not None:
                if early_stop.step(auc):
                    print(f"  → Early stopping at epoch {epoch_i}")
                    break

    # 测试
    def evaluate(self, test_set, interaction_func, device="cuda"):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_set, "Eval"):
            student_id, question, q_matrix_line, y = batch_data
            student_id: torch.Tensor = student_id.to(device)
            question: torch.Tensor = question.to(device)
            q_matrix_line: torch.Tensor = q_matrix_line.to(device)
            pred: torch.Tensor = self.net(student_id,
                                          question,
                                          q_matrix_line,
                                          interaction_func)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        acc = accuracy(y_pred, y_true)
        auc = area_under_curve(y_pred, y_true)
        f1 = f1_score(y_pred, y_true)

        return acc, auc, f1,

    def unpack(self):
        proficiency_level = self.net.student_emb(torch.arange(0, self.student_number).to().to(device)).detach().cpu().numpy()
        difficulty = self.net.difficulty(torch.arange(0, self.question_number).to().to(device)).detach().cpu().numpy()
        discrimination = self.net.discrimination(torch.arange(0, self.question_number).to().to(device)).detach().cpu().numpy()
        return proficiency_level, difficulty, discrimination,

    def update(self, interaction_func, interaction_func_str):
        self.interaction_function = interaction_func
        self.interaction_function_string = interaction_func_str

# from .eval import degree_of_agreement
#eval文件部分


def accuracy(y_pred, y_true, threshold=0.5, weights=None):
    pred = np.array(y_pred)
    true = np.array(y_true)
    result = np.where(pred > threshold, 1, 0)
    if weights is not None:
        correct = np.sum((true == result) * weights)
        total = np.sum(weights)
        return correct / total
    else:
        return metrics.accuracy_score(true, result)


def area_under_curve(y_pred, y_true):
    pred = np.array(y_pred)
    true = np.array(y_true)
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    return metrics.auc(fpr, tpr)


def f1_score(y_pred, y_true, threshold=0.5):
    pred = np.array(y_pred)
    true = np.array(y_true)
    result = np.where(pred >= threshold, 1, 0)
    return metrics.f1_score(true, result)


def loss(y_pred, y_true):
    pred = np.array(y_pred)
    true = np.array(y_true)
    losses = np.abs(pred - true)
    losses /= np.max(losses)
    return losses


def degree_of_agreement(q_matrix, proficiency, dataset):
    problem_number, knowledge_number = q_matrix.shape
    student_number = proficiency.shape[0]
    r_matrix = np.full((student_number, problem_number), -1)
    for lines in dataset:
        student_id_batch, question_batch, _, y_batch = lines
        for student_id, question, y in zip(student_id_batch, question_batch, y_batch ):
            r_matrix[student_id][question] = y
    doaList = []
    for k in range(knowledge_number):
        numerator = 0.0
        denominator = 0.0
        delta_matrix = proficiency[:, k].reshape(-1, 1) > proficiency[:, k].reshape(1, -1)
        question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
        for j in question_hask:
            # avoid blank logs
            row_vec = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
            column_vec = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vec * column_vec
            delta_response_logs = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
            i_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
            numerator += np.sum(delta_matrix * np.logical_and(mask, delta_response_logs))
            denominator += np.sum(delta_matrix * np.logical_and(mask, i_matrix))
        doaList.append(numerator / denominator)

    return np.mean(doaList)


# 模型入口部分

class SymbolicCDM:
    def __init__(self,
                 q_matrix: np.ndarray,
                 student_number: int,
                 question_number: int,
                 knowledge_number: int,
                 train_size,
                 train_set,
                 valid_set,
                 test_set,
                 device="cuda"):

        self.train_set = train_set
        self.train_size = train_size
        self.valid_set = valid_set
        self.test_set = test_set

        self.interaction = GeneticInteractionFunc(self.train_set, train_size)

        self.parameter = Parameter(student_number,
                                   question_number,
                                   knowledge_number)

        self.q_matrix = q_matrix
        self.logs = dict({})
        self.device = device
        self.saved_function = None # 用于保存最好交互函数

    def train(self, epochs, nn_epochs, dir, lr, population_size, ngen, cxpb, mutpb, wandb_instance=None):
        # for logs
        wandb_instance = wandb_instance
        lr = lr
        exp_dir = dir # 获取实验目录
        ckpt_path = os.path.join(exp_dir, 'best_interaction_fun.json')
        # 这里的ckpt_path是交互函数的名称保存，不是网络参数
        print(f"model save to: {ckpt_path}")

        best_metric = None
        best_epoch = None
        current_function = None
        fun_early_stop = EarlyStopping(patience=5, mode='max')



        for epoch in range(1, epochs+1):

            self.parameter.train(self.train_set, self.valid_set, epochs=nn_epochs, lr=lr, wandb_instance=wandb_instance, device=self.device, init=(epoch == 1)) #训练

            print(f"The {epoch}-th epoch parameters optimization complete")
            # update arguments
            arguments = self.parameter.unpack()  # 只是将tensor转换成np数组

            # 下面是一个传值的过程，从parameter(类：Parameter) -> interaction(类：GeneticInteractionFunc) -> 类InteractionFunc 
            # 传递的是三个np数组 分别对应s_id，q_id，q_id((bacth,1))三个进行emb+sigmoid的结果
            self.interaction.update(*arguments)  

            # calculate degree of agreement
            # doa = degree_of_agreement(self.q_matrix, arguments[0], self.valid_set)  # DOA时间有点久，验证的时候暂时不计算
            # print(f"DOA in this epoch: {doa}")   
            # evaluate argument on valid set

            # self.interaction.evaluation() 是同样的三个形状的向量进行预测，只是数据是测试集
            # Training interaction function  训练互动功能
            # print("Training interaction function...")


            fun_str = self.interaction.train(population_size, ngen, cxpb, mutpb)  #训练 这一部分相当于用了遗传算法进行优化 self.interaction.train()这个train()里面会用一个其他类的train()进行参数优化

            print(f"The {epoch}-th epoch interaction function complete")
            # Update interaction function
            if wandb_instance:
                log_metrics({
                    'interaction_train_epoch': epoch
                }, wandb_instance)
            self.parameter.update(self.interaction.function(), str(self.interaction))  #这里传递交互函数,把通过遗传算法优化好的函数进行传递，

            val_acc, val_auc, val_f1, val_rmse, current_function = self.interaction.evaluation(self.valid_set, self.saved_function)   # 应该是经过遗传算法优化以后再进行一个验证


            # 这里用来选择最优的交互函数
            if best_metric is None or val_auc > best_metric:
                best_metric = val_auc
                best_epoch = epoch
                self.saved_function = current_function
                with open(ckpt_path, 'w') as f:
                    json.dump({"Function": fun_str}, f, indent=4)
                print(f"  → Saved best model to {ckpt_path}")

            # 交互优化早停判断
            if fun_early_stop is not None:
                if fun_early_stop.step(val_auc):
                    print(f"  → Early stopping at epoch {epoch}")
                    break

            print(f"Interaction Fun : Val Metric: {val_auc:.4f}")

        #加载最优模型测试
        if ckpt_path is not None:
            test_acc, test_auc, test_f1, test_rmse, _= self.interaction.evaluation(self.test_set, self.saved_function)
            # doa = None
            # doa = degree_of_agreement(self.q_matrix, arguments[0], self.test_set)
            return test_acc, test_auc, test_rmse, ckpt_path, best_metric, best_epoch
        else:
            return None

