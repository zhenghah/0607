import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
import math
import time


class BasicData:
    def __init__(self, which_data='mine'):  

        self.dataPath = '/tf/haha/all_data/'

        self.which_data = which_data

        self.similarWord = {
            'fruity': ['fruity', 'fruit', 'Fruity', 'fruits', 'fruit-like', 'fruitiness'],
            'green': ['green', 'Green'],
            'sweet': ['sweet', 'Sweet'],
            'floral': ['floral', 'Floral', 'flower'],
            'woody': ['woody', 'wood', 'Woody'],
            'herbal': ['herbal', 'herbaceous', 'herb', 'herbs'],
            'fatty': ['fatty', 'Fatty', 'fat'],
        }

    def readOD(self, od_choice, od_selected=None, filelist=None):
        # od_choice是只保留大于出现od_choice次的od; od_selected是列表表示保留哪些od

        # 从文件中按顺序读取od
        if self.which_data == 'mine':
            filelist = ['summary_odorant.txt']

        all_od = []  # all_od中的每个元素是一个物质的od列表

        for file in filelist:
            f = open(self.dataPath+file)
            all_line = f.read().split('\n')
            f.close()
            all_line.pop(-1)
            for line in all_line:
                line = line.split()
                line.pop(0)
                line.pop(0)
                for each in range(len(line)):
                    line[each] = line[each].strip(',')
                all_od.append(line)

        # 数每种od的阳性样本数量
        od_count = {}
        for each in range(len(all_od)):
            odlist = all_od[each]
            for ele in odlist:
                if ele not in od_count:
                    od_count[ele] = 1
                else:
                    od_count[ele] += 1
        print('number of all type odor descriptors = %d' % len(od_count))

        od_count_sort = sorted(
            od_count.items(), key=lambda x: x[1], reverse=True)  
        # print('od_count_sort: ', end='') #打印所有出现的od及其次数
        # print(od_count_sort)

        high_od = []
        not_od = ['and', 'with', 'like', 'a', 'nuance', 'nuances',
                  'clean', 'mild', 'ethereal', 'ripe', 'slight', 'warm']
        for each in od_count_sort:
            if each[1] >= od_choice:
                if each[0] not in not_od:
                    high_od.append(each[0])
            else:
                break

        
        if self.which_data == 'mine':
            if 'fruit' in high_od:
                high_od.remove('fruit')

        print('appear times >= %d odor descriptors have %d' %
              (od_choice, len(high_od)))
        print('they are')
        print(high_od)

        if od_selected is None:
            self.od_name = high_od
            self.feature_od = len(high_od)
        else:
            self.od_name = od_selected
            self.feature_od = len(self.od_name)

         # one-hot制作
        self.od_mat_ori = np.zeros((len(all_od), self.feature_od))
        self.od_times = {}  # 用到的od与出现次数对应的字典
        for each in self.od_name:
            self.od_times[each] = 0
        if self.which_data == 'mine' or self.which_data == 'keller': #!!!!!!
        # else:
            for row in range(len(all_od)):
                for each in all_od[row]:
                    if each == 'fruit':
                        each = 'fruity'
                    for col in range(self.feature_od):
                        if each == self.od_name[col] and self.od_mat_ori[row][col] != 1:
                            self.od_mat_ori[row][col] = 1
                            self.od_times[self.od_name[col]] += 1
                            break
        print('final od_times with all samples:', end=' ')
        print(self.od_times)
        

    def readSmiles(self, filelist=None):
        all_smile = []

        if self.which_data == 'mine':
            filelist = ['summary_odorant.txt']

        for file in filelist:
            f = open(self.dataPath+file)
            all_line = f.read().split('\n')
            f.close()
            all_line.pop(-1)
            for line in all_line:
                line = line.split()
                smile = line[1]
                all_smile.append(smile)

        return all_smile


class Transformer2OD:
    def __init__(self, train_test=(5, 1)):
        self.train_test = train_test
        print('transformer2od')


    def atomFeature(self, all_mol):
        atom_type = {'C': 0, 'O': 1, 'S': 2, 'N': 3, 'Cl': 4, 'Na': 5, 'P': 6, 'F': 7, 'Mg': 8, 'I': 9, 'Br': 10, 'Zn': 11, 'Fe': 12,
                     'As': 13, 'Ca': 14, 'B': 15, 'Si': 16, 'K': 17, 'Co': 18, 'Cr': 19, 'H': 20, 'Al': 21, 'other': 22
                     }
        heavy_neigh = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 'other':5}
        num_H = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 'other':5}
        # aromatic = {True:0, False:0}
        # ring = {True:0, False:0}
        hybrid = {'S': 0, 'SP': 1, 'SP2': 2, 'SP3': 3,
                  'SP3D': 4, 'SP3D2': 5, 'UNSPECIFIED': 6, 'other': 7
                  }
        chiral = {'CHI_TETRAHEDRAL_CCW': 0,
                  'CHI_TETRAHEDRAL_CW': 1, 'CHI_UNSPECIFIED': 2, 'other': 3}
        formal_charge = {0: 0, -1: 1, 1: 2, 2: 3, -2: 4, 3: 5, 4: 6, 'other': 7}
        explicit_valence = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 'other': 8}
        implicit_valence = {0: 0, 1: 1, 2: 2, 3: 3, 'other': 4}


        self.len_atom_feat = len(atom_type) + len(heavy_neigh) + len(num_H) + 1 + 1 + len(hybrid) + len(chiral)
        self.len_atom_feat += (len(formal_charge) + len(explicit_valence) + len(implicit_valence))

        all_feat = np.zeros((self.sample, self.max_mol, self.len_atom_feat))
        for i in range(len(all_mol)):
            mol = all_mol[i]
            atoms = mol.GetAtoms()

            for j in range(len(atoms)):
                index_count = 0
                atom = atoms[j]

                a_type = atom.GetSymbol()
                if a_type not in atom_type:
                    a_type = 'other'
                temp_index = atom_type[a_type]
                all_feat[i][j][temp_index] = 1
                index_count += len(atom_type)

                a_heavy = atom.GetDegree()
                if a_heavy not in heavy_neigh:
                    a_heavy = 'other'
                temp_index = heavy_neigh[a_heavy] + index_count
                all_feat[i][j][temp_index] = 1
                index_count += len(heavy_neigh)

                a_H = atom.GetTotalNumHs()
                if a_H not in num_H:
                    a_H = 'other'
                temp_index = num_H[a_H] + index_count
                all_feat[i][j][temp_index] = 1
                index_count += len(num_H)

                a_aromatic = atom.GetIsAromatic()
                temp_index = 0 + index_count
                all_feat[i][j][temp_index] = int(a_aromatic)
                index_count += 1

                a_ring = atom.IsInRing()
                temp_index = 0 + index_count
                all_feat[i][j][temp_index] = int(a_ring)
                index_count += 1

                a_hybrid = atom.GetHybridization().name
                if a_hybrid not in hybrid:
                    a_hybrid = 'other'
                temp_index = hybrid[a_hybrid] + index_count
                all_feat[i][j][temp_index] = 1
                index_count += len(hybrid)

                a_chiral = atom.GetChiralTag().name
                if a_chiral not in chiral:
                    a_chiral = 'other'
                temp_index = chiral[a_chiral] + index_count
                all_feat[i][j][temp_index] = 1
                index_count += len(chiral)

                a_charge = atom.GetFormalCharge()
                if a_charge not in formal_charge:
                    a_charge = 'other'
                temp_index = formal_charge[a_charge] + index_count
                all_feat[i][j][temp_index] = 1
                index_count += len(formal_charge)

                a_exValence = atom.GetExplicitValence()
                if a_exValence not in explicit_valence:
                    a_exValence = 'other'
                temp_index = explicit_valence[a_exValence] + index_count
                all_feat[i][j][temp_index] = 1
                index_count += len(explicit_valence)

                a_imValence = atom.GetImplicitValence()
                if a_imValence not in implicit_valence:
                    a_imValence = 'other'
                temp_index = implicit_valence[a_imValence] + index_count
                all_feat[i][j][temp_index] = 1
                index_count += len(implicit_valence)

        return all_feat


    def molData_atomicF(self, smiles_list, rseed=1, dist_value='original', atomH=False, failed_mol=None, bondInfo=False):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if atomH:
            max_num_atom = 140
        else:
            max_num_atom = 60

        # 1. 将smiles转为mol对象
        all_mol = []
        if failed_mol is not None:
            self.failed_mol = failed_mol
        else:
            self.failed_mol = []

        for each in range(len(smiles_list)):
            if each in self.failed_mol:
                continue
        
            mol = Chem.MolFromSmiles(smiles_list[each])

            if atomH:
                try:
                    mol = Chem.AddHs(mol)
                except:
                    self.failed_mol.append(each)
                    continue

            try:
                flag = AllChem.EmbedMolecule(mol)
            except:
                flag = -1
            if flag == -1:
                self.failed_mol.append(each)
                continue

            if mol.GetNumAtoms() > max_num_atom:
                self.failed_mol.append(each)
                continue

            all_mol.append(mol)

        print('%d mol cannot compute distance, they are:' %
              len(self.failed_mol), end=' ')
        print(self.failed_mol)

        self.sample = len(all_mol)
        print('num_sample : %d' % self.sample)

        # 2. shuffle_index
        tf.random.set_seed(rseed)
        self.shuffle_index = tf.random.shuffle(
            range(self.sample))  # 如果这里加上seed反倒结果每次都不一样

        # 3. MASK
        self.max_mol = 60 
        num_atom_list = []  # 每个分子的原子个数的列表
        for i in range(len(all_mol)):
            temp = all_mol[i].GetNumAtoms()
            num_atom_list.append(temp)
            if self.max_mol < temp:
                self.max_mol = temp

        mask_mat = []
        for i in range(self.sample):
            num_atom = num_atom_list[i]
            temp = tf.zeros([1, num_atom])
            temp = tf.pad(
                temp, [[0, 0], [0, self.max_mol-num_atom]], constant_values=1)
            mask_mat.append(temp)
        mask_mat = tf.concat(mask_mat, axis=0)
        # (batch_size, 1, 1, max_mol)
        mask_mat = mask_mat[:, tf.newaxis, tf.newaxis, :]

        mask_mat = tf.gather(mask_mat, self.shuffle_index)

        print('mask_mat: ', end='')
        print(mask_mat.shape)

        # 4. atomic feature
        mol_atom_feat = self.atomFeature(all_mol)
        mol_atom_feat = tf.convert_to_tensor(mol_atom_feat)
        mol_atom_feat = tf.gather(mol_atom_feat, self.shuffle_index)
        mol_atom_feat = tf.cast(mol_atom_feat, dtype=tf.float32)

        print('mol_atom_feat.shape: ', end='')
        print(mol_atom_feat.shape)

        # 5. 邻接矩阵和距离矩阵
        all_adj = np.zeros((self.sample, self.max_mol, self.max_mol))
        if dist_value == 'original':
            all_dist = np.zeros((self.sample, self.max_mol, self.max_mol))
        else:
            all_dist = np.ones((self.sample, self.max_mol, self.max_mol))
            all_dist = all_dist * 1000
        for each in range(len(all_mol)):
            a = all_mol[each].GetNumAtoms()
            all_adj[each][:a, :a] = AllChem.GetAdjacencyMatrix(all_mol[each])
            all_dist[each][:a, :a] = AllChem.Get3DDistanceMatrix(all_mol[each])
        all_adj = tf.convert_to_tensor(all_adj)
        all_dist = tf.convert_to_tensor(all_dist)
        all_adj = tf.cast(all_adj, dtype=tf.float32)
        all_dist = tf.cast(all_dist, dtype=tf.float32)

        all_adj = tf.gather(all_adj, self.shuffle_index)
        all_dist = tf.gather(all_dist, self.shuffle_index)
        if dist_value != 'original':
            temp = tf.constant([math.e])
            all_dist = tf.cast(all_dist, tf.float32)
            all_dist = tf.pow(temp, -all_dist)
        self.dist_value = dist_value
        print('distance value type is: %s' % self.dist_value)
        print('all_adj.shape = :', end=' ')
        print(all_adj.shape)
        print('all_dist.shape = :', end=' ')
        print(all_dist.shape)

        # 6. 分训练和测试集
        self.num_train = (
            self.sample // (self.train_test[0]+self.train_test[1])) * self.train_test[0]
        self.num_test = self.sample - self.num_train

        self.train_input = (tf.ones([self.num_train, 1, 1]), mol_atom_feat[:self.num_train],
                            all_adj[:self.num_train], all_dist[:self.num_train], mask_mat[:self.num_train])
        self.test_input = (tf.ones([self.num_test, 1, 1]), mol_atom_feat[self.num_train:],
                           all_adj[self.num_train:], all_dist[self.num_train:], mask_mat[self.num_train:])

        if bondInfo:
            self.bondInfo(all_mol)

        return all_mol


    
    def odData(self, od_mat_ori, od_name):
        self.feature_od = od_mat_ori.shape[1]
        self.od_name = od_name
        od_mat_ori = tf.convert_to_tensor(od_mat_ori, dtype=tf.float32)
        if self.sample + len(self.failed_mol) - od_mat_ori.shape[0] != 0:
            temp = tf.zeros(
                [self.sample + len(self.failed_mol), self.feature_od])  # 0补多了但无影响
            od_mat_ori = tf.concat([od_mat_ori, temp], axis=0)

        # 去除掉fail_mol中的分子
        save_mol = list(range(self.sample + len(self.failed_mol)))
        for each in self.failed_mol:
            save_mol.remove(each)
        save_mol = tf.constant(save_mol)

        od_mat_ori = tf.gather(od_mat_ori, save_mol)

        # shuffle
        od_shuffled = tf.gather(od_mat_ori, self.shuffle_index)
        print('od_shuffled.shape = ', end=' ')
        print(od_shuffled.shape)

        self.od_train = od_shuffled[: self.num_train]
        self.od_test = od_shuffled[self.num_train:]

        # 阳性样本训练和测试集上的分布
        positive = tf.reduce_sum(self.od_test, axis=0)
        positive_dict = {}
        for each in range(self.feature_od):
            positive_dict[self.od_name[each]] = positive[each].numpy()
        print('positive sample distribution in test set:')
        print(positive_dict)

        positive = tf.reduce_sum(self.od_train, axis=0)
        positive_dict = {}
        for each in range(self.feature_od):
            positive_dict[self.od_name[each]] = positive[each].numpy()
        print('positive sample distribution in test set:')
        print(positive_dict)
        self.positive_dict = positive_dict

    def Mask(self, mol_mat_atomID):
        mol_mat_atomID = tf.convert_to_tensor(mol_mat_atomID)
        mask_mat = tf.cast(tf.math.equal(mol_mat_atomID, 0), tf.float32)

        # (batch_size, 1, 1, max_mol)
        return mask_mat[:, tf.newaxis, tf.newaxis, :]

    

    def hpTableCreat(self, path):  # 用来创建hp表格的抬头
        metric_record_temp = ['avg_F1', 'avg_pcs', 'avg_rc', 'avg_F1_train', 'avg_pcs_train', 'avg_rc_train',
                              'max_F1']
        for each in self.od_name:
            metric_record_temp.append('F1_'+each)
            metric_record_temp.append('max_F1_'+each)

        metric_record = []
        for each in metric_record_temp:
            metric_record.append(hp.Metric(each, display_name=each))

        with tf.summary.create_file_writer(path).as_default():
            temp = []
            for each in self.hparams_dict:
                temp.append(self.hparams_dict[each])
            hp.hparams_config(
                hparams=temp,
                metrics=metric_record
            )

        self.path = path

    def HPset(self, hparams_dict):
        '''
        暂时让feedfoward_dim = encoder_dim !!!!!!!!!!!

        hparams_dict = {
            'num_heads': [3, 4, 5],
            'single_attn_dim': [10,20,30],
            'num_encoderLayer': [3, 4, 5],
            'num_decoderLayer': [1, 2, 3], # 若不需要decoderLayer, 输入[None]
            'learning_rate': [0.0005, 0.0001],
            'temperature': [0.7, 1.0]
        }
        '''

        self.hparams_dict = {}

        self.hparams_dict['num_heads'] = hp.HParam('num_heads',
                                                   hp.Discrete(hparams_dict['num_heads']))
        self.hparams_dict['single_attn_dim'] = hp.HParam('single_attn_dim',
                                                         hp.Discrete(hparams_dict['single_attn_dim']))
        self.hparams_dict['num_encoderLayer'] = hp.HParam('num_encoderLayer',
                                                          hp.Discrete(hparams_dict['num_encoderLayer']))
        self.hparams_dict['num_decoderLayer'] = hp.HParam('num_decoderLayer',
                                                          hp.Discrete(hparams_dict['num_decoderLayer']))
        self.hparams_dict['learning_rate'] = hp.HParam('learning_rate',
                                                       hp.Discrete(hparams_dict['learning_rate']))
        self.hparams_dict['temperature'] = hp.HParam('temperature',
                                                     hp.Discrete(hparams_dict['temperature']))


    def calFscore(self, evaluate_dict):
        # 顺便计算一下pcs和rc的均值
        fscore_dict = {'avg': 0, 'avg_pcs': 0, 'avg_rc': 0}
        for each in self.od_name:
            fscore_dict[each] = 0
            pcs = evaluate_dict[each+'_'+each+'_'+'precision']
            fscore_dict['avg_pcs'] += pcs
            rc = evaluate_dict[each+'_'+each+'_'+'recall']
            fscore_dict['avg_rc'] += rc
            temp1 = pcs+rc
            if temp1 != 0:
                temp = 2*pcs*rc / temp1
                fscore_dict[each] = temp
                fscore_dict['avg'] += temp
        fscore_dict['avg'] /= self.feature_od
        fscore_dict['avg_pcs'] /= self.feature_od
        fscore_dict['avg_rc'] /= self.feature_od

        return fscore_dict

    
   
    def attnExtract(self, model, num_cls=None, trainORtest='test', dc_layer=False):  # 获得测试集的attn

        num_cls = self.feature_od

        model_getattn = tf.keras.Model(
            inputs=model.inputs, outputs=model.get_layer('transformer').output)
        if trainORtest == 'test':
            # (samples, num_heads, numcls+max_mol, numcls+max_mol)
            attn = model_getattn.predict(self.test_input)[1]
        else:
            attn = model_getattn.predict(self.train_input)[1]

        if dc_layer:
            pass
        else:
            # dc_layer设置为true时， 代码还没改！！！
            # (samples, num_heads, numcls, max_mol)
            attn = attn[:, :, :num_cls, num_cls:]

        return attn

    def drawAttn(self, attn, smiles_list, pred, savepath, max_draw=50, per_head=True, trainORtest='test', attn_c=4, posneg='positive'):
        # attn_c 指attention的值放大的倍数
        # attn要求输入为列表

        from rdkit import rdBase, Chem
        from rdkit.Chem import AllChem, Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        from IPython.display import SVG
        from rdkit.Chem.Draw import IPythonConsole

        smiles_list1 = smiles_list.copy()  
        temp = self.failed_mol.copy()
        temp.sort(reverse=True)
        for each in temp:
            smiles_list1.pop(each)

        num_dclayers = len(attn)

        if per_head:
            # for i in range(num_dclayers):
            # attn[i] = attn[i].numpy()
            num_heads = attn[0].shape[1]
            img_per_row = num_heads
        else:
            for i in range(num_dclayers):
                attn[i] = tf.reduce_sum(attn[i], axis=1)
                attn[i] = tf.expand_dims(attn[i], axis=1)
                attn[i] = attn[i].numpy()
            num_heads = 1
            if num_dclayers == 1:
                img_per_row = 3
            else:
                img_per_row = num_dclayers

        if trainORtest == 'test':
            sample = self.num_test
            shuffle_index = self.shuffle_index[self.num_train:]
            y = self.od_test
        else:
            sample = self.num_train
            shuffle_index = self.shuffle_index
            y = self.od_train

        for which_cls in range(self.feature_od):

            # 把positive或negative sample挑出来
            index_test = []
            for i in range(sample):
                if posneg == 'positive':
                    if tf.math.equal(y[i, which_cls], 1) and pred[which_cls][i] >= 0.5:
                        index_test.append(i)
                else:
                    if tf.math.equal(y[i, which_cls], 0) and pred[which_cls][i] < 0.5:
                        index_test.append(i)

            if len(index_test) == 0:
                continue

            smiles_index = []
            for i in index_test:
                smiles_index.append(shuffle_index[i])

            mol_od = []
            colorDict_list = []
            atom_list = []

            if len(smiles_index) > max_draw:
                num_mol_draw = max_draw
            else:
                num_mol_draw = len(smiles_index)

            for i in range(num_mol_draw):
                s_index = smiles_index[i]
                mol = Chem.MolFromSmiles(smiles_list1[s_index])
                num_atoms = mol.GetNumAtoms()

                for dc in range(num_dclayers):

                    for head in range(num_heads):
                        atom_list.append(list(range(num_atoms)))
                        mol_od.append(mol)
                        color_dict = {}

                        for atom in range(num_atoms):
                            score = attn[dc][index_test[i]
                                             ][head][which_cls][atom]
                            score *= attn_c
                            if score > 0.8:
                                score = 0.8
                            color_dict[atom] = (1-score, 1-score, 1)
                        colorDict_list.append(color_dict)

            bond_lists = [None] * len(mol_od)
            # print(len(bond_lists))
            img = Draw.MolsToGridImage(mol_od, highlightAtomLists=atom_list,
                                       highlightAtomColors=colorDict_list,
                                       molsPerRow=img_per_row,
                                       highlightBondLists=bond_lists,
                                       subImgSize=(500, 500),
                                       returnPNG=False)

            img.save(savepath+self.od_name[which_cls]+'.png')

    
    def TPfind(self, smiles_list, pred, trainORtest='test'):
        # num_sample 指要保留多少个TF样本， trainORtest指TF由哪个数据集预测

        smiles_list1 = smiles_list.copy()  # 函数传入的列表不是深拷贝
        temp = self.failed_mol.copy()
        temp.sort(reverse=True)
        for each in temp:
            smiles_list1.pop(each)

        if trainORtest == 'test':
            sample = self.num_test
            shuffle_index = self.shuffle_index[self.num_train:].numpy()
            y = self.od_test
        else:
            sample = self.num_train
            shuffle_index = self.shuffle_index.numpy()
            y = self.od_train

        result = {}  # 返回值是在smiles_list1中的编号
        for which_cls in range(self.feature_od):

            # 把positive sample挑出来
            index_test = []
            for i in range(sample):
                if tf.math.equal(y[i, which_cls], 1) and pred[which_cls][i] > 0.5:
                    index_test.append(i)

            if len(index_test) == 0:
                result[self.od_name[which_cls]] = [{}, {}]
                continue

            smiles_index = []
            for i in index_test:
                smiles_index.append(shuffle_index[i])

            result[self.od_name[which_cls]] = [smiles_index, index_test]

        # result = {'od1': [[tf sample index], [testset_index]], ...}
        return result, smiles_list1

    
    def attnProve(self, model, weight_path_list, draw_num, smiles_list, savepath, topk=2, atom_appear_times=2):

        from rdkit import rdBase, Chem
        from rdkit.Chem import AllChem, Draw

        # 加载模型， 预测test集， 保留预测正确的样本编号，取交集, topk
        od_tp_index = {}
        attn_topk_index = []
        for path in weight_path_list:

            model.load_weights(path)
            print('loading %s' % path)

            pred = model(self.test_input)

            model_getattn = tf.keras.Model(
                inputs=model.inputs, outputs=model.get_layer('transformer').output)
            # (num_test, num_heads, numcls, max_mol)
            attn = model_getattn(self.test_input)[1][-1]
            attn = tf.reduce_sum(attn, axis=1)  # (num_test, numcls, max_mol)
            print(attn.shape)
            # attn topk
            # (num_test, num_cls, topk)
            temp = tf.math.top_k(attn, k=topk).indices
            attn_topk_index.append(temp)

            del model_getattn

            # 挑预测正确的positive sample
            temp_od_tp_index, smiles_list1 = self.TPfind(
                smiles_list, pred, trainORtest='test')
            # 取交集
            if len(od_tp_index) == 0:
                od_tp_index = temp_od_tp_index
            else:
                for od in self.od_name:
                    od_tp_index[od][0] = [
                        j for j in temp_od_tp_index[od][0] if j in od_tp_index[od][0]]
                    od_tp_index[od][1] = [
                        j for j in temp_od_tp_index[od][1] if j in od_tp_index[od][1]]

        print(attn_topk_index)

        # 对于每个od, 列出tp样本所对应的highlight atom
        od_highlight_dict = {}  # {od1: [{high_sample1}, {high_sample2}, ...]}
        for od in range(self.feature_od):

            od_highlight_dict[self.od_name[od]] = []

            testset_tp_index = od_tp_index[self.od_name[od]][1]

            for i in testset_tp_index:
                high_atomlist = set([])
                atom_times = {}
                for m in range(len(weight_path_list)):
                    temp = attn_topk_index[m][i][od]  # shape = (topk)
                    for k in range(topk):
                        if self.test_input[4][i, 0, 0, k] == 1:
                            break

                        if int(temp[k]) in atom_times:
                            atom_times[int(temp[k])] += 1
                        else:
                            atom_times[int(temp[k])] = 1
                    for atom in atom_times:
                        if atom_times[atom] >= atom_appear_times:
                            high_atomlist.add(atom)

                od_highlight_dict[self.od_name[od]].append(high_atomlist)

        # 画highlight atom的图
        for od in range(self.feature_od):

            smiles_tp_index = od_tp_index[self.od_name[od]][0]

            if len(smiles_tp_index) == 0:
                continue
            highlist = []
            mollist = []

            for i, s_index in enumerate(smiles_tp_index):

                mol = Chem.MolFromSmiles(smiles_list1[s_index])
                print(mol.GetNumAtoms())
                mollist.append(mol)
                highlist.append(list(od_highlight_dict[self.od_name[od]][i]))

                if len(mollist) >= draw_num:
                    break

            print(highlist)

            img = Draw.MolsToGridImage(
                mollist, highlightAtomLists=highlist, subImgSize=(500, 500), returnPNG=False)
            img.save(savepath+self.od_name[od]+'.png')

        




class Transformer2OD_tada(Transformer2OD):
    def __init__(self): 
        print('Transformer2OD_tada')
        super(Transformer2OD_tada, self).__init__()
        self.use_adj_dist = 'both'
        self.fold = 5
    
    
    def modelBuild_ec(self, batch_size, num_heads=3, single_attn_dim=30, feedforward_dim=120, 
                    num_encoderLayer=2, num_decoderLayer=2, dropout_rate=0.1, lr=0.001, compile=True, 
                    normal_init=False, temperature=1):

        encoder_dim = num_heads * single_attn_dim

        print('modelBuild_ec-temperature %f' % temperature)
        import Transformer_mol3 as my_trans

        # !!!!!!!!!!!!!!!!! 只在EConly时有用
        units_dict = {
            1: [100], 2:[100,50], 3:[120,60,30], 4:[120, 90, 60, 30] 
        }
        print('units ', end ='')
        print(units_dict[num_decoderLayer])

        transformer = my_trans.EConly( # !!!!!!!!!!!!!!!!!!!!!!
            num_ec_layers=num_encoderLayer,
            units=units_dict[num_decoderLayer], # !!!!!!
            # num_dc_layers=num_decoderLayer, # !!!!!!!!!!!!!!!!
            d_model=encoder_dim,
            num_heads=num_heads,
            dff=feedforward_dim,
            num_cls=self.feature_od,
            rate=0.1,
            dist_value=self.dist_value,
            temperature=temperature
        )

        input_cls = tf.keras.Input(shape=(1, 1), batch_size=batch_size)
        input_mol = tf.keras.Input(
            shape=(self.max_mol, self.len_atom_feat), batch_size=batch_size)
        input_adj = tf.keras.Input(
            shape=(self.max_mol, self.max_mol), batch_size=batch_size)
        input_dist = tf.keras.Input(
            shape=(self.max_mol, self.max_mol), batch_size=batch_size)
        input_mask = tf.keras.Input(
            shape=(1, 1, self.max_mol), batch_size=batch_size)
        input_label = tf.keras.Input(
            shape=(self.feature_od), batch_size=batch_size)

        inputs = [input_cls, input_mol, input_adj,
                  input_dist, input_mask, input_label]
        dc_output, attn, ec, ec_attn= transformer(inputs)
        outputs = []
        for i in range(self.feature_od):
            outputs.append(tf.keras.layers.Dense(
                1, activation='sigmoid', name=self.od_name[i])(dc_output[:, i, :]))

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.input_items = len(inputs)

        if compile:
            # loss list 和 metrics list
            loss_list = []
            metrics_list = []

            for each in range(self.feature_od):
                temp_loss = tf.keras.losses.BinaryCrossentropy()
                temp_metrics = [
                    tf.keras.metrics.Precision(
                        name=self.od_name[each]+'_precision'),
                    tf.keras.metrics.Recall(name=self.od_name[each]+'_recall')
                ]
                loss_list.append(temp_loss)
                metrics_list.append(temp_metrics)

            learning_rate = lr
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            model.compile(optimizer=optimizer,
                          loss=loss_list, metrics=metrics_list)

        if normal_init:
            weights = model.get_weights()
            normal_weights = []
            for each in weights:
                normal_weights.append(np.random.normal(
                    loc=0.0, scale=0.01, size=each.shape))
            model.set_weights(normal_weights)

        return model
    
    
    def modelBuild2(self, batch_size, num_heads=3, single_attn_dim=30, feedforward_dim=120, 
                    num_encoderLayer=2, num_decoderLayer=2, dropout_rate=0.1, lr=0.001, compile=True, 
                    normal_init=False, temperature=1):
        if self.use_adj_dist == 'both':
            import Transformer_mol3 as my_trans # !!!!!!!
        elif self.use_adj_dist == 'adj_only':
            import Transformer_noDist as my_trans # !!!!!!!
        elif self.use_adj_dist == 'dist_only':
            import Transformer_noAdj as my_trans # !!!!!!!
        elif self.use_adj_dist == 'MAT': # MAT论文的的attention但不是整个模型
            import Transformer_MAT as my_trans

        encoder_dim = num_heads * single_attn_dim

        print('modelBuild2-temperature %f' % temperature)

        # # !!!!!!!!!!!!!!!!! 只在EConly时有用
        # units_dict = {
        #     1: [100], 2:[100,50], 3:[120,60,30], 4:[120, 90, 60, 30] 
        # }
        # print('units ', end ='')
        # print(units_dict[num_decoderLayer])

        transformer = my_trans.Transformer4( # !!!!!!!!!!!!!!!!!!!!!!
            num_ec_layers=num_encoderLayer,
            # units=units_dict[num_decoderLayer], # !!!!!!
            num_dc_layers=num_decoderLayer, # !!!!!!!!!!!!!!!!
            d_model=encoder_dim,
            num_heads=num_heads,
            dff=feedforward_dim,
            num_cls=self.feature_od,
            rate=0.1,
            dist_value=self.dist_value,
            temperature=temperature
        )

        input_cls = tf.keras.Input(shape=(1, 1), batch_size=batch_size)
        input_mol = tf.keras.Input(
            shape=(self.max_mol, self.len_atom_feat), batch_size=batch_size)
        input_adj = tf.keras.Input(
            shape=(self.max_mol, self.max_mol), batch_size=batch_size)
        input_dist = tf.keras.Input(
            shape=(self.max_mol, self.max_mol), batch_size=batch_size)
        input_mask = tf.keras.Input(
            shape=(1, 1, self.max_mol), batch_size=batch_size)
        input_label = tf.keras.Input(
            shape=(self.feature_od), batch_size=batch_size)

        inputs = [input_cls, input_mol, input_adj,
                  input_dist, input_mask, input_label]
        dc_output, attn, ec, ec_attn= transformer(inputs)
        outputs = []
        for i in range(self.feature_od):
            outputs.append(tf.keras.layers.Dense(
                1, activation='sigmoid', name=self.od_name[i])(dc_output[:, i, :]))

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.input_items = len(inputs)

        if compile:
            # loss list 和 metrics list
            loss_list = []
            metrics_list = []

            for each in range(self.feature_od):
                temp_loss = tf.keras.losses.BinaryCrossentropy()
                temp_metrics = [
                    tf.keras.metrics.Precision(
                        name=self.od_name[each]+'_precision'),
                    tf.keras.metrics.Recall(name=self.od_name[each]+'_recall')
                ]
                loss_list.append(temp_loss)
                metrics_list.append(temp_metrics)

            learning_rate = lr
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            model.compile(optimizer=optimizer,
                          loss=loss_list, metrics=metrics_list)

        if normal_init:
            weights = model.get_weights()
            normal_weights = []
            for each in weights:
                normal_weights.append(np.random.normal(
                    loc=0.0, scale=0.01, size=each.shape))
            model.set_weights(normal_weights)

        return model

    

    
    def dataSetMaker(self, batch_size, which_fold=None):
        if which_fold is not None:
            fold_size = self.num_train // self.fold
            print('%dth fode size = %d' % (which_fold, fold_size))

            # test
            output_test = self.od_train[which_fold*fold_size: (which_fold+1)*fold_size]
            input_test = []
            for i in range(len(self.train_input)):
                temp = self.train_input[i][which_fold*fold_size: (which_fold+1)*fold_size]
                input_test.append(temp)
            input_test = tuple(input_test)


            # train
            fold_index = list(range(self.fold))
            fold_index.pop(which_fold)
            input_train = []
            
            output_train = self.od_train[fold_index[0]*fold_size: (fold_index[0]+1)*fold_size]
            for each in range(1, len(fold_index)):
                temp = self.od_train[fold_index[each]*fold_size: (fold_index[each]+1)*fold_size]
                output_train = tf.concat([output_train, temp], axis=0)

            input_train = []
            for i in range(len(self.train_input)):
                temp = self.train_input[i][fold_index[0]*fold_size: (fold_index[0]+1)*fold_size]
                input_train.append(temp)
            for each in range(1, len(fold_index)):
                for i in range(len(self.train_input)):
                    temp = self.train_input[i][fold_index[each]*fold_size: (fold_index[each]+1)*fold_size]
                    input_train[i] = tf.concat([input_train[i], temp], axis=0)
            input_train = tuple(input_train)
            
            print([output_test.shape, output_train.shape])

        else:
            output_train = self.od_train
            output_test = self.od_test
            input_train = self.train_input
            input_test = self.test_input

        # make output
        out_train = []
        out_test = []
        for i in range(self.feature_od):
            out_train.append(output_train[:, i])
            out_test.append(output_test[:, i])
        out_train = tuple(out_train)
        out_test = tuple(out_test)   

        # 向self.train_input, self.test_input中添加label
        if len(input_train) == self.input_items-1:
            input_train = list(input_train)
            input_train.append(output_train)
            input_train = tuple(input_train)
        if len(input_test) == self.input_items-1:
            input_test = list(input_test)
            input_test.append(output_test)
            input_test = tuple(input_test)

        # 输入输出转为dataset形式
        for each in input_train:
            print(each.shape)


        inp_train = tf.data.Dataset.from_tensor_slices(input_train)
        outp_train = tf.data.Dataset.from_tensor_slices(out_train)
        train_dataset = tf.data.Dataset.zip((inp_train, outp_train))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        inp_test = tf.data.Dataset.from_tensor_slices(input_test)
        outp_test = tf.data.Dataset.from_tensor_slices(out_test)
        test_dataset = tf.data.Dataset.zip((inp_test, outp_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return (train_dataset, test_dataset)


    
    def foldTrain(self, hpara_dict, max_epochs, record_dir, save_path='None'):
        avg_fscore = {}
        for f in range(self.fold):
            record_dir1 = record_dir+'_'+str(f)
            _, fscore_dict = self.modelTrain2(hpara_dict, max_epochs, record_dir1, which_fold=f, save_threshold=None, save_path=save_path)
            for k in fscore_dict:
                if k not in avg_fscore:
                    avg_fscore[k] = fscore_dict[k]
                else:
                    avg_fscore[k] += fscore_dict[k]

        for k in avg_fscore:
            avg_fscore[k] /= self.fold

        return avg_fscore


        
    
    def modelTrain2(self, hpara, epochs, record_dir, which_fold=None, batch_size=32, record_hp=None, save_threshold=None, save_path=None, model_cls=None,
                    normal_init=False):
        # 搭配modelBuild2
        '''hpara = {
            'num_heads': 3,
            'single_attn_dim': 50,
            'num_encoderLayer': 2,
            'num_decoderLayer': 3, 
            'learning_rate': 0.0001
        }

        record_dir用来记录画图的训练结果
        record_hp用来记录hp的表头 
        例“ record_hp = '/tf/haha/code/logs/0123/', record_dir='/tf/haha/code/logs/0123/0'
        '''

        if record_hp is not None:
            self.hpTableCreat(record_hp)

        print(record_dir)

        if model_cls is None:
            model = self.modelBuild2( #!!!!!!!
                batch_size=batch_size,
                num_heads=hpara['num_heads'],
                single_attn_dim=hpara['single_attn_dim'],
                feedforward_dim=hpara['num_heads']*hpara['single_attn_dim'],
                num_encoderLayer=hpara['num_encoderLayer'],
                num_decoderLayer=hpara['num_decoderLayer'],
                dropout_rate=0.1,
                lr=hpara['learning_rate'],
                compile=True,
                normal_init=normal_init,
                temperature=hpara['temperature']
            )
        else:
            model = model_cls

        train_dataset, test_dataset = self.dataSetMaker(batch_size=batch_size, which_fold=which_fold)

        # metrices, loss, optimizer
        loss_list = []
        metrics_dict = {
            'precision': [],
            'recall': [],
            'loss': []
        }

        for each in range(self.feature_od):
            temp_loss = tf.keras.losses.BinaryCrossentropy()
            loss_list.append(temp_loss)

            metrics_dict['precision'].append(
                tf.keras.metrics.Precision(name=self.od_name[each]+'_precision'))
            metrics_dict['recall'].append(
                tf.keras.metrics.Recall(name=self.od_name[each]+'_recall'))
            metrics_dict['loss'].append(
                tf.keras.metrics.BinaryCrossentropy(name=self.od_name[each]+'_loss'))

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hpara['learning_rate'])

        # 开始表演

        @tf.function
        def train_step(input_batch, output_batch):
            with tf.GradientTape() as tape:
                preds = model(input_batch, training=True)
                if self.feature_od == 1:
                    preds = [preds]
                loss = model.losses
                # loss = []
                for i in range(self.feature_od):
                    # loss.append(loss_list[i](
                    #     output_batch[i], preds[i])+model.losses[i])
                    loss.append(loss_list[i](
                        output_batch[i], preds[i]))
                
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            # loss平均和metrics计算
            for i in range(self.feature_od):
                metrics_dict['precision'][i].update_state(
                    output_batch[i], preds[i])
                metrics_dict['recall'][i].update_state(
                    output_batch[i], preds[i])
                metrics_dict['loss'][i].update_state(output_batch[i], preds[i])

            return loss

        @tf.function
        def test_step(input_batch, output_batch):
            preds = model(input_batch, training=False)
            if self.feature_od == 1:
                    preds = [preds]
            for i in range(self.feature_od):
                metrics_dict['precision'][i].update_state(
                    output_batch[i], preds[i])
                metrics_dict['recall'][i].update_state(
                    output_batch[i], preds[i])
                metrics_dict['loss'][i].update_state(output_batch[i], preds[i])

        max_F1 = 0
        for epo in range(epochs):
            print(epo, end=' ')
            # reset metrices
            for i in range(self.feature_od):
                metrics_dict['precision'][i].reset_states()
                metrics_dict['recall'][i].reset_states()
                metrics_dict['loss'][i].reset_states()

            # train
            for (batch, (inp, outp)) in enumerate(train_dataset):

                train_step(input_batch=inp, output_batch=outp)

            # metrics计算
            evaluate_dict = {}
            for i in range(self.feature_od):
                od = self.od_name[i]
                pcs = metrics_dict['precision'][i].result()
                evaluate_dict[od+'_'+od+'_'+'precision'] = float(pcs)

                rc = metrics_dict['recall'][i].result()
                evaluate_dict[od+'_'+od+'_'+'recall'] = float(rc)

                loss = metrics_dict['loss'][i].result()
                evaluate_dict[od+'_'+od+'_'+'loss'] = float(loss)

            fscore_dict = self.calFscore(evaluate_dict)

            # tensorboard
            with tf.summary.create_file_writer(record_dir+'/train/').as_default():
                if epo == epochs-1 and record_hp is not None:
                    hp.hparams(hpara)

                # 记录fscore
                tf.summary.scalar('avg_F1_train', fscore_dict['avg'], step=epo)
                tf.summary.scalar(
                    'avg_pcs_train', fscore_dict['avg_pcs'], step=epo)
                tf.summary.scalar(
                    'avg_rc_train', fscore_dict['avg_rc'], step=epo)

                for each in self.od_name:
                    tf.summary.scalar('F1_train_'+each,
                                      fscore_dict[each], step=epo)

                # 记录loss （pcs rc暂时先不记录吧）
                temp_loss = 0
                for each in self.od_name:
                    temp_loss += evaluate_dict[od+'_'+od+'_'+'loss']
                    tf.summary.scalar(
                        'loss_train_'+each, evaluate_dict[od+'_'+od+'_'+'loss'], step=epo)
                tf.summary.scalar('avg_loss_train_',
                                  temp_loss/self.feature_od, step=epo)

            # reset metrices
            for i in range(self.feature_od):
                metrics_dict['precision'][i].reset_states()
                metrics_dict['recall'][i].reset_states()
                metrics_dict['loss'][i].reset_states()

            # test
            for (batch, (inp, outp)) in enumerate(test_dataset):

                test_step(inp, outp)

            # metrics计算
            evaluate_dict = {}
            for i in range(self.feature_od):
                od = self.od_name[i]
                pcs = metrics_dict['precision'][i].result()
                evaluate_dict[od+'_'+od+'_'+'precision'] = float(pcs)

                rc = metrics_dict['recall'][i].result()
                evaluate_dict[od+'_'+od+'_'+'recall'] = float(rc)

                loss = metrics_dict['loss'][i].result()
                evaluate_dict[od+'_'+od+'_'+'loss'] = float(loss)

            fscore_dict = self.calFscore(evaluate_dict)

            # tensorboard
            with tf.summary.create_file_writer(record_dir+'/test/').as_default():
                if epo == epochs-1 and record_hp is not None:
                    hp.hparams(hpara)

                # 记录fscore
                tf.summary.scalar('avg_F1', fscore_dict['avg'], step=epo)
                tf.summary.scalar('avg_pcs', fscore_dict['avg_pcs'], step=epo)
                tf.summary.scalar('avg_rc', fscore_dict['avg_rc'], step=epo)

                for each in self.od_name:
                    tf.summary.scalar('F1_'+each, fscore_dict[each], step=epo)

                if fscore_dict['avg'] > max_F1:
                    max_F1 = fscore_dict['avg']
                    fscore_dict_max = fscore_dict
                    max_epoF1 = epo
                    # 保存weight
                    if save_threshold is not None and max_F1 >= save_threshold:
                        print('weight保存开始：', end='')
                        print(time.ctime(), end=' ')
                        model.save_weights(save_path)
                        print('weight保存完成', end=' ')
                        print(time.ctime())

                # 记录loss （pcs rc暂时先不记录吧）
                temp_loss = 0
                for each in self.od_name:
                    temp_loss += evaluate_dict[od+'_'+od+'_'+'loss']
                    tf.summary.scalar(
                        'loss_test_'+each, evaluate_dict[od+'_'+od+'_'+'loss'], step=epo)
                tf.summary.scalar('avg_loss_test_', temp_loss /
                                  self.feature_od, step=epo)

        # 记录最大F1
        with tf.summary.create_file_writer(record_dir+'/test/').as_default():
            hp.hparams(hpara)

            tf.summary.scalar('max_F1', fscore_dict_max['avg'], step=max_epoF1)

            for each in self.od_name:
                tf.summary.scalar(
                    'max_F1_'+each, fscore_dict_max[each], step=max_epoF1)

        print(fscore_dict_max)

        return model, fscore_dict_max

    
    
    
    def modelTrain_hp1(self, run_count_begin=0, max_epochs=200, save_threshold=None, save_path=None, foldfile=None):

        count_run = 0
        for num_heads in self.hparams_dict['num_heads'].domain.values:
            for single_attn_dim in self.hparams_dict['single_attn_dim'].domain.values:
                for num_encoderLayer in self.hparams_dict['num_encoderLayer'].domain.values:
                    for num_decoderLayer in self.hparams_dict['num_decoderLayer'].domain.values:
                        for learning_rate in self.hparams_dict['learning_rate'].domain.values:
                            for temperature in self.hparams_dict['temperature'].domain.values:
                                hpara_dict = {
                                    'num_heads': num_heads, 'single_attn_dim': single_attn_dim,
                                    'num_encoderLayer': num_encoderLayer, 'num_decoderLayer': num_decoderLayer,
                                    'learning_rate': learning_rate, 'temperature': temperature
                                }

                                record_dir = self.path + \
                                    str(count_run+run_count_begin)

                                temp = '%d_%d_%.1f_%d%d/' % (
                                        num_heads, single_attn_dim, temperature, num_encoderLayer, num_decoderLayer)

                                if foldfile is None:
                                    self.modelTrain2(
                                        hpara_dict, max_epochs, record_dir, save_threshold=save_threshold, save_path=save_path+temp)
                                    count_run += 1

                                else:
                                    avg_fscore = self.foldTrain(hpara_dict, max_epochs, record_dir)
                                    file = open(foldfile, 'a')
                                    file.write(record_dir+'\n')
                                    file.write(temp+'\n')
                                    for k in avg_fscore:
                                        file.write(k+': '+str(avg_fscore[k])+'\t')
                                    file.write('\n')
                                    file.close()

                                    count_run += 1
    
    
    
    def attnExtract(self, model, num_cls=None, trainORtest='test', ecORdc='dc', dc_layer=True):
        # 可以抽取dc_attn, ec_attn, ec_output

        model_getattn = tf.keras.Model(
            inputs=model.inputs, outputs=model.get_layer('transformer').output)
        if trainORtest == 'test':
            trans_output = model_getattn(self.test_input)
            if ecORdc == 'dc':
                attn = trans_output[1]
            elif ecORdc == 'ec':
                attn = trans_output[3]
            elif ecORdc == 'ec_output':
                attn = trans_output[2]
        else:
            trans_output = model_getattn(self.train_input)
            if ecORdc == 'dc':
                attn = trans_output[1]
            elif ecORdc == 'ec':
                attn = trans_output[3]
            elif ecORdc == 'ec_output':
                attn = trans_output[2]

        return attn

    

class Transformer2Sub(Transformer2OD_tada):
    def __init__(self):
        print('transformer2sub')
        super(Transformer2Sub, self).__init__()

    def odData(self, smiles_list, sub_list=None, od_name=None, otherpairs=[]):
        # otherpairs = [[8,9], [6,7]] # 8或9（8，9皆有&&）
        if sub_list is None:
            sub_list = ['C(=O)O', 'C=C-C=C', 'C1CCCC1', 'CCCCCC', 'c1ccccc1', 
                        'C(=O)O&&c1ccccc1', 'C(=O)O~~c1ccccc1', 'C=C-C=C~~C1CCCC1']

        from rdkit import Chem

        if od_name is None:
            od_name = sub_list
        self.od_name = od_name

        all_mol = []
        for i in range(len(smiles_list)):
            if i not in self.failed_mol:
                mol = Chem.MolFromSmiles(smiles_list[i])
                all_mol.append(mol)
        if len(all_mol) != self.sample:
            raise('wrong in odData')

        self.feature_od = len(sub_list) + len(otherpairs)

        od_mat_ori = np.zeros((self.sample, self.feature_od))
        for j in range(len(sub_list)):
            if '&&' in sub_list[j]:
                sub_split = sub_list[j].split('&&')

                sub_strc = []
                for each_sub in sub_split:
                    sub_strc.append(Chem.MolFromSmarts(each_sub))

                for i in range(self.sample):
                    if all_mol[i].HasSubstructMatch(sub_strc[0]) and all_mol[i].HasSubstructMatch(sub_strc[1]):
                        od_mat_ori[i][j] = 1

            elif '~~' in sub_list[j]:
                sub_split = sub_list[j].split('~~')

                sub_strc = []
                for each_sub in sub_split:
                    sub_strc.append(Chem.MolFromSmarts(each_sub))

                for i in range(self.sample):
                    if all_mol[i].HasSubstructMatch(sub_strc[0]) or all_mol[i].HasSubstructMatch(sub_strc[1]):
                        od_mat_ori[i][j] = 1

            else:
                for i in range(self.sample):
                    sub_strc = Chem.MolFromSmarts(sub_list[j])
                    if all_mol[i].HasSubstructMatch(sub_strc):
                        od_mat_ori[i][j] = 1

        # otherpairs处理
        for i in range(len(otherpairs)):
            pairs = otherpairs[i]
            for s in range(self.sample):
                if od_mat_ori[s][pairs[0]] == 1 or od_mat_ori[s][pairs[1]] == 1:
                    od_mat_ori[s][len(sub_list)+i] = 1
        

        od_mat_ori = tf.convert_to_tensor(od_mat_ori, dtype=tf.float32)

        # shuffle
        od_shuffled = tf.gather(od_mat_ori, self.shuffle_index)
        print('od_shuffled.shape = ', end=' ')
        print(od_shuffled.shape)

        self.od_train = od_shuffled[: self.num_train]
        self.od_test = od_shuffled[self.num_train:]

        # 阳性样本训练和测试集上的分布
        positive = tf.reduce_sum(self.od_test, axis=0)
        positive_dict = {}
        for each in range(self.feature_od):
            positive_dict[self.od_name[each]] = positive[each].numpy()
        print('positive sample distribution in test set:')
        print(positive_dict)

        positive = tf.reduce_sum(self.od_train, axis=0)
        positive_dict = {}
        for each in range(self.feature_od):
            positive_dict[self.od_name[each]] = positive[each].numpy()
        print('positive sample distribution in test set:')
        print(positive_dict)
        self.positive_dict = positive_dict


