from prelude import *


class DCNNMLD:
    def __init__(self, rho: float, sir_db: int, is_improved: bool):
        self.is_improved = is_improved
        self.rho = rho
        self.sir_db = sir_db
        self.total_layer = 4
        self.feature_maps = [32, 16, 8, 1]
        self.feature_sizes = [36, 3, 3, 36]

        if self.is_improved:
            self.unique_name = "improved_ant{}_rho{:.1f}_sir{}".format(NUM_ANT, rho, sir_db)
        else:
            self.unique_name = "baseline_ant{}_rho{:.1f}_sir{}".format(NUM_ANT, rho, sir_db)

        self.__make_graph()

        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

    def __make_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 定义占位符
            self.y = tf.placeholder(tf.float32, [TRANSMIT_TIMES_PER_BATCH, 2 * NUM_ANT, 1], "y")
            self.h = tf.placeholder(tf.float32, [TRANSMIT_TIMES_PER_BATCH, 2 * NUM_ANT, 2 * NUM_ANT], "h")
            self.one_hot = tf.placeholder(tf.float32, [TRANSMIT_TIMES_PER_BATCH, QPSK_CANDIDATE_SIZE], "one_hot")
            self.w = tf.placeholder(tf.float32, [TRANSMIT_TIMES_PER_BATCH, 2 * NUM_ANT, 1], "w")
            self.hat_w = tf.placeholder(tf.float32, [TRANSMIT_TIMES_PER_BATCH, 2 * NUM_ANT, 1], "w_mld")

            # 定义卷积层
            layer_input = {}
            layer_output = {}
            in_channels = {}
            out_channels = {}
            for layer_id in range(self.total_layer):
                with tf.variable_scope("conv_layer_{}".format(layer_id)):
                    if layer_id == 0:
                        # 将其实部和虚部排列为两个通道进行卷积, 这里需要使用Fortan-like形式的reshape来转换，
                        # 由于tensorflow不支持直接Fortan-like的reshape，我们还得绕点弯路用C-like的reshape来实现
                        w_in = tf.reshape(self.hat_w, [PACKETS_PER_BATCH, TRANSMIT_TIMES_PER_PACKET, 2, NUM_ANT])
                        w_in = tf.transpose(w_in, perm=[0, 1, 3, 2])
                        w_in = tf.reshape(w_in, [PACKETS_PER_BATCH, PACKET_SIZE, 1, 1])

                        layer_input[layer_id] = w_in
                        in_channels[layer_id] = 1
                    else:
                        layer_input[layer_id] = layer_output[layer_id - 1]
                        in_channels[layer_id] = self.feature_maps[layer_id - 1]

                    out_channels[layer_id] = self.feature_maps[layer_id]

                    kernel = tf.get_variable(name="weights",
                                             shape=[self.feature_sizes[layer_id],
                                                    1,
                                                    in_channels[layer_id],
                                                    out_channels[layer_id]],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())

                    conv = tf.nn.conv2d(input=layer_input[layer_id], filter=kernel, strides=[1, 1, 1, 1],
                                        padding="SAME")

                    bias = tf.get_variable(name="bias", shape=[out_channels[layer_id]], dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())

                    if layer_id == self.total_layer - 1:
                        layer_output[layer_id] = conv + bias
                    else:
                        layer_output[layer_id] = tf.nn.relu(conv + bias)

            last_out = layer_output[self.total_layer - 1]

            # 换回来为常规的虚实分割形式，因为待会要和y做计算
            w_out = tf.reshape(last_out, [PACKETS_PER_BATCH, TRANSMIT_TIMES_PER_PACKET, NUM_ANT, 2])
            w_out = tf.transpose(w_out, perm=[0, 1, 3, 2])
            w_out = tf.reshape(w_out, [TRANSMIT_TIMES_PER_BATCH, 2 * NUM_ANT, 1])

            self.w_cnn = w_out
            # 计算损失函数
            if self.is_improved:
                self.tf_qpsk_candidates = tf.constant(QPSK_CANDIDATES, dtype=tf.float32)
                tmp_dst = (self.y - self.w_cnn) - tf.tensordot(self.h, self.tf_qpsk_candidates, axes=1)
                self.distance = tf.reduce_sum(tf.square(tmp_dst), axis=1)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.one_hot, logits=-self.distance))
            else:
                self.loss = tf.reduce_mean(tf.square(self.w_cnn - self.w))

            self.global_step = tf.Variable(tf.constant(0), trainable=False, name="global_step")
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
            self.init_variables = tf.global_variables_initializer()

    def load(self):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "savedModel/{}/".format(self.unique_name)
            saver.restore(self.sess, path)
            print("Model \"{}\" loaded".format(self.unique_name))

    def save(self):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            path = "savedModel/{}/".format(self.unique_name)
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            prefix = saver.save(self.sess, path)
            print("Model saved at \"{}\"".format(prefix))

    def close(self):
        self.sess.close()

    def train(self):
        print("Initializing model {}".format(self.unique_name))
        self.sess.run(self.init_variables)

        train_io = DataSet(flag=0, rho=self.rho, sir=self.sir_db)

        flip_count = 0
        best_ber = None
        epoch = 0
        while epoch < MAX_EPOCHS:            
            batch_idx = 0
            for y, h, s, one_hot, w, hat_s, hat_w in train_io.fetch():
                print("Training model \"{}\", epoch {}/{}, batch={}".format(self.unique_name, epoch + 1, MAX_EPOCHS, batch_idx+1), end='\r')
                self.sess.run(
                    self.optimizer,
                    feed_dict={
                        self.y: y,
                        self.h: h,
                        self.w: w,
                        self.hat_w: hat_w,
                        self.one_hot: one_hot
                    }
                )
                batch_idx += 1
            print()
            epoch += 1

            if epoch % VALID_MODEL_EVERY_EPOCHES == 0:
                new_ber = self.__valid_model()
                if best_ber is None or new_ber < best_ber:
                    best_ber = new_ber
                    self.save()
                else:
                    flip_count += 1
                    if flip_count >= MAX_FLIP:
                        break        
        print("Model \"{}\" train over".format(self.unique_name))

    def __valid_model(self):
        err_mld = 0
        err_cnn = 0
        bits_count = 0
        batch_idx = 0
        valid_io = DataSet(flag=1, rho=self.rho, sir=self.sir_db)
        for y, h, s, one_hot, w, hat_s, hat_w in valid_io.fetch():
            print("Validating model \"{}\", batch={}".format(self.unique_name, batch_idx+1), end="\r")
            bits = get_bits(s)
            bits_count += bits.size

            bits_mld = get_bits(hat_s)
            err_mld += len(np.argwhere(bits_mld != bits))
            ber_mld = err_mld / bits_count

            bits_cnn, _ = self.detect_bits_batch(y, h, hat_w, k=1)
            err_cnn += len(np.argwhere(bits_cnn != bits))
            ber_cnn = err_cnn / bits_count

            batch_idx+=1
        print()
        print("Model validated, MLD_BER={:e}, CNN_BER={:e}".format(ber_mld, ber_cnn))
        return ber_cnn

    def detect_bits_batch(self, y, h, hat_w_in, k=1):
        """检测一个batch中传输的比特"""
        hat_w = hat_w_in
        for _ in range(k):
            tilde_w = self.sess.run(self.w_cnn,feed_dict={self.hat_w: hat_w})
            tilde_s = mld_batch(y - tilde_w, h)
            if k != k - 1:
                hat_w = y - h @ tilde_s
        return get_bits(tilde_s), tilde_w
