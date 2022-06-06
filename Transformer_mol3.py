# Transformer模型返回ec_output和ec_attn

import numpy as np
import tensorflow as tf

def scaled_dot_product_attention_AD(q, k, v, mask, adj=None, dist=None, dist_value='original'): # distance和邻接矩阵！！！！！
    '''
    q.shape = (batch, num_heads, num_q, attention_dim) (注：attention_dim是指对于每个attention的feature的个数. 
                                                        encoder时，num_q = max_mol_real, decoder时，num_q = num_cls)
    k.shape = (batch, num_heads, max_mol_real, attention_dim)
    v.shape = (batch, num_heads, max_mol_real, attention_dim)
    mask.shape = (batch, 1, 1, max_mol_real)
    adj_shape = (batch, max_mol_real, max_mol_real)
    dist.shape = (batch, max_mol_real, max_mol_real)
    '''

    mask = mask * -1e9
 
    # 普通的attention QK^T
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch, num_heads, num_q, max_mol_real)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    scaled_attention_logits += mask
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch, num_heads, num_q, max_mol_real)

    # 邻接矩阵和距离矩阵
    if adj is not None:
        adj = tf.expand_dims(adj, axis=1) # (batch, 1,  max_mol_real, max_mol_real)
        attention_weights = attention_weights * adj # (batch, num_heads, max_mol_real, max_mol_real)

    if dist is not None:
        dist = tf.expand_dims(dist, axis=1) # (batch, 1,  max_mol_real, max_mol_real)
        if dist_value == 'original':
            dist += mask
            dist = tf.nn.softmax(dist, axis=-1) # (batch, 1, max_mol_real, max_mol_real)
        attention_weights = attention_weights * dist # (batch, num_heads, max_mol_real, max_mol_real)


    output = tf.matmul(attention_weights, v) # (batch, num_heads, num_q, attention_dim)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads): # d_model=attention_dim * num_heads, d_model在classTransformer中是encoder_dim
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads #depth是attention_dim

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, adj=None, dist=None, dist_value='original'): # (注意！这里输入的顺序时v, k, q， 而scaled_dot_product_attention_AD函数的输入顺序是q, k, v)
        '''
        q.shape = (batch, num_q, attention_dim*num_heads)
        k.shape = (batch, max_mol_real, attention_dim*num_heads)
        v.shape = (batch, max_mol_real, attention_dim*num_heads)
        mask.shape = (batch, 1, 1, max_mol_real)
        '''
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, num_q, d_model)
        k = self.wk(k)  # (batch_size, max_mol_real, d_model)
        v = self.wv(v)  # (batch_size, max_mol_real, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, num_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, max_mol_real, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, max_mol_real, depth)

        # scaled_attention.shape == (batch_size, num_heads, max_mol, depth)
        # attention_weights.shape == (batch_size, num_heads, max_mol, max_mol)
        # scaled_attention, attention_weights = scaled_dot_product_attention(
        #     q, k, v, mask)
        scaled_attention, attention_weights = scaled_dot_product_attention_AD(
            q, k, v, mask, adj, dist, dist_value) #!!!!!!!!!!!

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, num_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, num_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, num_q, d_model) #就感觉这个dense这么多余

        return output, attention_weights



class MultiHeadAttention_encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads): # d_model=attention_dim * num_heads, d_model在classTransformer中是encoder_dim
        super(MultiHeadAttention_encoder, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads #depth是attention_dim

        self.wq_dist = tf.keras.layers.Dense(d_model/2)
        self.wk_dist = tf.keras.layers.Dense(d_model/2)
        self.wv_dist = tf.keras.layers.Dense(d_model/2)

        self.wq_adj = tf.keras.layers.Dense(d_model/2)
        self.wk_adj = tf.keras.layers.Dense(d_model/2)
        self.wv_adj = tf.keras.layers.Dense(d_model/2)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size, heads):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v_ori, k_ori, q_ori, mask, adj=None, dist=None, dist_value='original'): # (注意！这里输入的顺序时v, k, q， 而scaled_dot_product_attention_AD函数的输入顺序是q, k, v)
        '''
        q.shape = (batch, num_q, attention_dim*num_heads)
        k.shape = (batch, max_mol_real, attention_dim*num_heads)
        v.shape = (batch, max_mol_real, attention_dim*num_heads)
        mask.shape = (batch, 1, 1, max_mol_real)
        '''
        batch_size = tf.shape(q_ori)[0]

        # dist
        q = self.wq_dist(q_ori)  # (batch_size, num_q, d_model/2)
        k = self.wk_dist(k_ori)  # (batch_size, max_mol_real, d_model/2)
        v = self.wv_dist(v_ori)  # (batch_size, max_mol_real, d_model/2)

        q = self.split_heads(q, batch_size, self.num_heads//2)  # (batch_size, num_heads, num_q, depth)
        k = self.split_heads(k, batch_size, self.num_heads//2)  # (batch_size, num_heads, max_mol_real, depth)
        v = self.split_heads(v, batch_size, self.num_heads//2)  # (batch_size, num_heads, max_mol_real, depth)

        # scaled_attention.shape == (batch_size, num_heads, max_mol, depth)
        # attention_weights.shape == (batch_size, num_heads, max_mol, max_mol)
        scaled_attention, attention_weights_dist = scaled_dot_product_attention_AD(
            q, k, v, mask, adj=None, dist=dist, dist_value=dist_value) #!!!!!!!!!!!

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, num_q, num_heads, depth)

        concat_attention_dist = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model//2))  # (batch_size, num_q, d_model/2)

        # adj
        q = self.wq_adj(q_ori)  # (batch_size, num_q, d_model/2)
        k = self.wk_adj(k_ori)  # (batch_size, max_mol_real, d_model/2)
        v = self.wv_adj(v_ori)  # (batch_size, max_mol_real, d_model/2)

        q = self.split_heads(q, batch_size, self.num_heads//2)  # (batch_size, num_heads, num_q, depth)
        k = self.split_heads(k, batch_size, self.num_heads//2)  # (batch_size, num_heads, max_mol_real, depth)
        v = self.split_heads(v, batch_size, self.num_heads//2)  # (batch_size, num_heads, max_mol_real, depth)

        scaled_attention, attention_weights_adj = scaled_dot_product_attention_AD(
            q, k, v, mask, adj=adj, dist=None, dist_value=dist_value)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, num_q, num_heads, depth)

        concat_attention_adj = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model//2))  # (batch_size, num_q, d_model/2)

        # 合并
        concat_attention = tf.concat([concat_attention_dist, concat_attention_adj], axis=-1) # (batch_size, num_q, d_model)
        attention_weights = tf.concat([attention_weights_dist, attention_weights_adj], axis=1) # (batch, num_heads, max_mol, max_mol)

        output = self.dense(concat_attention)  # (batch_size, num_q, d_model) #就感觉这个dense这么多余

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff): # dff是feedforward_dim
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, num_q, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, num_q, encoder_dim)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention_encoder(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, adj, dist, dist_value): # 此处应有training参数
        '''
        x.shape: (batch, max_mol_real, attention_dim * num_heads)
        mask.shape: (batch, 1, 1, max_mol_real)
        '''

        attn_output, attns = self.mha(x, x, x, mask, adj, dist, dist_value)  # (batch_size, max_mol_real, d_model), (batch, heads, max_mol_real, max_mol_real)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, max_mol_real, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, max_mol_real, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, max_mol_real, d_model)

        return (out2, attns) # encoder的输出attns


class Encoder(tf.keras.layers.Layer): # num_layers是num_encoderLayer
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding,
        #                                         self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, adj, dist, dist_value): # 此处应有training参数
        '''
        x.shape: (batch, max_mol_real, attention_dim * num_heads)
        mask.shape: (batch, 1, 1, max_mol_real)
        '''

        # seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]

        # x = self.dropout(x, training=training)

        attns = []
        for i in range(self.num_layers):
            x, attn = self.enc_layers[i](x, mask, adj, dist, dist_value)
            attns.append(attn)

        return (x, attns)  # (batch_size, max_mol_real, d_model), [(batch, num_heads, max_mol_real, max_mol_real),  * num_eclayers]


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, mask): # 此处应有training参数
        '''
        x.shape = (batch, num_cls, attention_dim * num_heads)
        enc_output.shape = (batch_size, max_mol_real, d_model)
        mask.shape: (batch, 1, 1, max_mol_real)
        '''
        attn_output, attns = self.mha(enc_output, enc_output, x, mask) # (batch_size, num_cls, d_model), (batch, num_heads, num_cls, max_mol_real)
        attn_output = self.dropout1(attn_output)     
        out1 = self.layernorm1(attn_output + x) # (batch, num_cls, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, num_cls, d_model) 
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, num_cls, d_model)

        return out2, attns


class Decoder(tf.keras.layers.Layer):# num_layers是num_decoderLayer
    def __init__(self, num_layers, d_model, num_heads, dff, num_cls, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_cls = num_cls

        self.cls_embedding = [tf.keras.layers.Dense(d_model) for i in range(num_cls)]

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, input, enc_output, mask): # 此处应有training参数
        '''
        input.shape = (batch, 1, 1)
        enc_output.shape = (batch_size, max_mol_real, d_model)
        mask.shape: (batch, 1, 1, max_mol_real)
        '''

        cls_embed = [self.cls_embedding[i](input) for i in range(self.num_cls)] # [(batch, 1, d_model)]
        cls_embed = tf.concat(cls_embed, axis=-2) # (batch, num_cls, d_model)

        x = self.dropout(cls_embed)

        attns = []
        for i in range(self.num_layers):
            x, attn = self.dec_layers[i](x, enc_output, mask) # (batch, num_cls, d_model), [(batch, num_heads, num_cls, max_mol_real)]
            attns.append(attn)

        return x, attns



# 加入了contrastive loss
class Transformer4(tf.keras.layers.Layer):  # 这模型应该是0.561的那个模型
    def __init__(self, num_ec_layers, num_dc_layers, d_model, num_heads, dff, num_cls, 
                    rate=0.1, dist_value='original', temperature=None):
        # attention_dim = d_model / num_heads
        super().__init__(name='transformer')

        self.num_cls = num_cls
        self.num_dc_layers = num_dc_layers
        self.dist_value = dist_value

        self.raw_mol_embedding = tf.keras.layers.Dense(d_model)

        self.encoder = Encoder(num_ec_layers, d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate) # embedding atomic feature时的dropout
        
        self.decoder = [Decoder(num_dc_layers, d_model, num_heads, dff, 1, rate) for i in range(num_cls)]

        if temperature is not None:
            self.contrastive_list = [Contrastive(temperature=temperature) for i in range(num_cls)]

    def call(self, inputs): # 此处应有training参数， 因为使用model.fit所以不需要
        '''
        max_mol_real是真实的分子最大长度， max_mol是加上了cls输入后的长度
        inputs = [
            [1], # shape=(batch, 1, 1)  注：无论多少cls, 输入都只有一个1，多cls通过多个cls_embedding实现
            mol_atom_feature, # shape=(batch, max_mol_real, atom_feat_dim)
            all_adj, # shape=(batch, max_mol_real, max_mol_real)
            all_distance, # shape = (batch, max_mol_real, max_mol_real)
            mask # shape = (batch, 1, 1, max_mol_real)
            label_cls # shape = (batch, num_cls)
        ]
        '''

        mol_embed = self.raw_mol_embedding(inputs[1]) # (batch, max_mol_real, d_model)
        mol_embed = self.dropout(mol_embed)
        
        ec_output, ec_attns = self.encoder(mol_embed, inputs[4], inputs[2], inputs[3], self.dist_value) #(batch_size, max_mol_real, d_model)

        dc_output = []
        attns = [[] for i in range(self.num_dc_layers)]
        for i in range(self.num_cls):
            dc, attn = self.decoder[i](inputs[0], ec_output, inputs[4]) # (batch, 1, d_model), [(batch, num_heads, 1, max_mol_real)]
            dc_output.append(dc)
            for j in range(self.num_dc_layers):
                attns[j].append(attn[j])
        
        dc_output = tf.concat(dc_output, axis=-2) # (batch, num_cls, d_model)

        for i in range(self.num_dc_layers):
            attns[i] = tf.concat(attns[i], axis=-2) # [(batch, num_heads, num_cls, max_mol_real)]

        l2_output = []
        for i in range(self.num_cls):
            temp = self.contrastive_list[i](dc_output[:, i, :], inputs[-1][:, i]) # (batch, d_model)
            temp = tf.expand_dims(temp, axis=1) # (batch, 1, d_model)
            l2_output.append(temp)
        l2_output = tf.concat(l2_output, axis=-2) # (batch, num_cls, d_model)

        return (l2_output, attns, ec_output, ec_attns)




class Transformer_sub(tf.keras.layers.Layer):  
    def __init__(self, num_dc_layers, d_model, num_heads, dff, num_cls, 
                    rate=0.1, temperature=None):
        # attention_dim = d_model / num_heads
        super().__init__(name='transformer')

        self.num_cls = num_cls
        self.num_dc_layers = num_dc_layers
        
        self.decoder = [Decoder(num_dc_layers, d_model, num_heads, dff, 1, rate) for i in range(num_cls)]

        if temperature is not None:
            self.contrastive_list = [Contrastive(temperature=temperature) for i in range(num_cls)]

    def call(self, inputs): # 此处应有training参数， 因为使用model.fit所以不需要
        '''
        inputs = [
            [1], # shape=(batch, 1, 1)  注：无论多少cls, 输入都只有一个1，多cls通过多个cls_embedding实现
            mol_atom_feature, # shape=(batch, max_mol_real, d_model)
            mask # shape = (batch, 1, 1, max_mol_real)
            label_cls # shape = (batch, num_cls)
        ]
        '''
        dc_output = []
        attns = [[] for i in range(self.num_dc_layers)]
        for i in range(self.num_cls):
            dc, attn = self.decoder[i](inputs[0], inputs[1], inputs[2]) # (batch, 1, d_model), [(batch, num_heads, 1, max_mol_real)]
            dc_output.append(dc)
            for j in range(self.num_dc_layers):
                attns[j].append(attn[j])

        dc_output = tf.concat(dc_output, axis=-2) # (batch, num_cls, d_model)

        for i in range(self.num_dc_layers):
            attns[i] = tf.concat(attns[i], axis=-2) # [(batch, num_heads, num_cls, max_mol_real)]

        l2_output = []
        for i in range(self.num_cls):
            temp = self.contrastive_list[i](dc_output[:, i, :], inputs[-1][:, i]) # (batch, d_model)
            temp = tf.expand_dims(temp, axis=1) # (batch, 1, d_model)
            l2_output.append(temp)
        l2_output = tf.concat(l2_output, axis=-2) # (batch, num_cls, d_model)

        return (l2_output, attns)


class EC2OD(tf.keras.layers.Layer): # 用训练好的encoder直接推测od
    def __init__(self, units, num_cls, temperature, rate=0.1):
        # 是不是用relu作为激活函数，使用dropout是不合理的？
        super().__init__()

        self.units = units
        # self.rate = rate

        self.hiddenLayers = [self.hiddenMake() for i in range(num_cls)]
        self.contra_list = [Contrastive(temperature=temperature) for i in range(num_cls)]

    def hiddenMake(self):
        layer_list = []
        for i in range(len(self.units)):
            layer_list.append(tf.keras.layers.Dense(self.units[i], activation='relu'))
        return tf.keras.models.Sequential(layer_list)

    def call(self, inputs):
        '''
        [1], # shape=(batch, 1, 1) (注: 没有必要存在, 懒得重写模型输入的代码)
        ec_output, # shape = (batch, max_mol, d_model)
        mask, # shape = (batch, 1, 1, max_mol)
        label, # shape = (batch, num_cls)
        '''
        num_cls = inputs[-1].shape[1]

        mask = tf.squeeze(inputs[-2]) # (batch, max_mol)
        mask = tf.expand_dims(mask, axis=-1) # (batch, max_mol, 1)
        mask = tf.equal(mask, 0) # (batch, max_mol, 1)
        mask = tf.cast(mask, dtype=tf.float32)

        ec_output = inputs[-3]
        ec_output *= mask # (batch, max_mol, d_model)
        ec_output = tf.reduce_sum(ec_output, axis=1) # (batch, d_model)

        output = []
        for i in range(num_cls):
            x = self.hiddenLayers[i](ec_output) # (batch, units[-1])
            x = self.contra_list[i](x, label=inputs[-1][:, i]) # (batch, d_model)
            output.append(x)

        return output
        


class EConly(tf.keras.layers.Layer):
    def __init__(self, num_ec_layers, units, d_model, num_heads, dff, num_cls, 
                    rate=0.1, dist_value='original', temperature=None):
        # attention_dim = d_model / num_heads
        super().__init__(name='transformer')

        self.num_cls = num_cls
        self.dist_value = dist_value
        self.units = units

        self.raw_mol_embedding = tf.keras.layers.Dense(d_model)

        self.encoder = Encoder(num_ec_layers, d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate) # embedding atomic feature时的dropout
        
        self.hiddenLayers = [self.hiddenMake() for i in range(num_cls)]
        # self.decoder = [Decoder(num_dc_layers, d_model, num_heads, dff, 1, rate) for i in range(num_cls)]

        if temperature is not None:
            self.contra_list = [Contrastive(temperature=temperature) for i in range(num_cls)]

    def hiddenMake(self):
        layer_list = []
        for i in range(len(self.units)):
            layer_list.append(tf.keras.layers.Dense(self.units[i], activation='relu'))
        return tf.keras.models.Sequential(layer_list)


    def call(self, inputs): # 此处应有training参数， 因为使用model.fit所以不需要
        '''
        max_mol_real是真实的分子最大长度， max_mol是加上了cls输入后的长度
        inputs = [
            [1], # shape=(batch, 1, 1)  注：无论多少cls, 输入都只有一个1，多cls通过多个cls_embedding实现
            mol_atom_feature, # shape=(batch, max_mol_real, atom_feat_dim)
            all_adj, # shape=(batch, max_mol_real, max_mol_real)
            all_distance, # shape = (batch, max_mol_real, max_mol_real)
            mask # shape = (batch, 1, 1, max_mol_real)
            label_cls # shape = (batch, num_cls)
        ]
        '''

        mol_embed = self.raw_mol_embedding(inputs[1]) # (batch, max_mol_real, d_model)
        mol_embed = self.dropout(mol_embed)
        
        ec_output, ec_attns = self.encoder(mol_embed, inputs[4], inputs[2], inputs[3], self.dist_value) #(batch_size, max_mol_real, d_model)

        mask = tf.squeeze(inputs[-2]) # (batch, max_mol)
        mask = tf.expand_dims(mask, axis=-1) # (batch, max_mol, 1)
        mask = tf.equal(mask, 0) # (batch, max_mol, 1)
        mask = tf.cast(mask, dtype=tf.float32)

        ec_output *= mask # (batch, max_mol, d_model)
        ec_output = tf.reduce_sum(ec_output, axis=1) # (batch, d_model)

        output = []
        for i in range(self.num_cls):
            x = self.hiddenLayers[i](ec_output) # (batch, units[-1])
            x = self.contra_list[i](x, label=inputs[-1][:, i]) # (batch, d_model)
            x = tf.expand_dims(x, axis=1) # (batch, 1, d_model)
            output.append(x)
        output = tf.concat(output, axis=1) # (batch, num_cls, d_model)

        return output, None, None, None


class Contrastive(tf.keras.layers.Layer):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def mask_label(self, label):
        # mask1: 对positive sample标记为1
        mask1 = tf.cast(tf.math.equal(label, 1), tf.float32)
        # mask0: 对neg sample标记为1
        mask0 = tf.cast(tf.math.equal(label, 0), tf.float32)
        return mask1, mask0

    def call(self, inputs, label, training=None):
        '''
        inputs.shape = (batch, d_model)
        label.shape = (batch, )
        '''
        batch = inputs.shape[0]
        # l2正则化inputs ############注意！！！！！！！！！！！！！在这一步中将改变inputs
        inputs = tf.math.l2_normalize(inputs, axis=1) # (batch, d_model)

        if training:
            mask1, mask0 = self.mask_label(label) # (batch, ), (batch, )
            num_p = tf.reduce_sum(mask1) # scalar
            num_n = tf.reduce_sum(mask0) # scalar
            # num_label_vector是|p(i)|组成的向量
            num_label_vector = num_p * mask1 + num_n * mask0 # (batch, )


            # corr矩阵
            inner_product = tf.matmul(inputs, inputs, transpose_b=True) # (batch, batch)
            mask_diag = tf.ones([batch, batch], dtype=tf.float32)
            mask_diag -= tf.eye(batch)
            inner_product = inner_product * mask_diag # (batch, batch)
            inner_product /= self.temperature
            inner_product_exp = tf.exp(inner_product)

            # 分母
            denominator = tf.reduce_sum(inner_product_exp, axis=1) # (batch, )
            denominator = tf.math.log(denominator)
            denominator *= num_label_vector # (batch, )

            # 分子 # (未检查是否正确)
            numerator = mask1 * tf.reduce_sum(mask1*inner_product, axis=1) + mask0 * tf.reduce_sum(mask0*inner_product, axis=1) # (batch, )

            # 整合
            numerator_deno = numerator - denominator # (batch, )
            # loss_vec = tf.math.divide_no_nan(numerator_deno, num_label_vector)
            loss_vec = numerator_deno / num_label_vector # 可能需要改成tf.math.divide_no_nan
            loss = tf.reduce_sum(loss_vec) # scalar
            contrastive_loss = -loss

            self.add_loss(contrastive_loss)

        return inputs # (batch, d_model) #注意，此时返回值经过过l2-normalize


