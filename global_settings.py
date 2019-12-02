"""该模块包含了程序所需要的全局设置参数"""

# 发射端和接收端的天线数量
NUM_ANT = 4

# 归一化之后的多普勒频率
NORMALIZED_DOPPLER_FREQUENCY = 0.1

# 数据包长度（比特）
PACKET_SIZE = 576

# 每个数据包需要的传输次数
TRANSMIT_TIMES_PER_PACKET = int(PACKET_SIZE / (2 * NUM_ANT))

# 每批次数据所包含的数据包数量
PACKETS_PER_BATCH = 10

# 每批次数据需要的传输次数
TRANSMIT_TIMES_PER_BATCH = PACKETS_PER_BATCH * TRANSMIT_TIMES_PER_PACKET

# 训练集的总批次
TRAIN_TOTAL_BATCH = 20000

# 验证集的总批次
VALID_TOTAL_BATCH = 2000

# 测试集的总批次
TEST_TOTAL_BATCH = 1000

# 多进程并行生成数据的进程数量，一般小于等于你电脑的CPU核心数量，0代表自动识别
NUM_WORKERS = 0

# 每个子进程的任务数量（防止内存溢出）
MAX_TASKS_PER_CHILD = 100

# 最大训练代数
MAX_EPOCHS = 20

# 每间隔多少代训练后验证一次模型
VALID_MODEL_EVERY_EPOCHES = 1

# 最大震荡次数（没耐心的设置为1）
MAX_FLIP = 1
