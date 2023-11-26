import os
import time
from datetime import datetime
import multiprocessing #用于并行执行进程的库。
from argparse import ArgumentParser #用于解析命令行参数

import yaml
import torch

# 从模块或脚本导入的自定义函数
from tool import set_seed
from abfcm import abfcm_train_group, abfcm_output_group, abfcm_nms, abfcm_iou_process, \
    abfcm_final_result_per_subject, abfcm_final_result_best, abfcm_train_and_eval
# abfcm（自适应偏差模糊C均值）算法

# 根据输出和标签张量计算二元交叉熵损失
def bi_loss(output, label):
    weight = torch.empty_like(output)
    c_0 = 0.05  # (label > 0).sum / torch.numel(label)
    c_1 = 1 - c_0
    weight[label > 0] = c_1 # 给标签为 0 的样本赋予 c_1(即1-c_0) 的权重
    weight[label == 0] = c_0 # 给标签为 0 的样本赋予 c_0 的权重
    loss = torch.nn.functional.binary_cross_entropy(output, label, weight)
    return loss

# 根据提供的选项和主题列表创建必要的文件夹/目录
def create_folder(opt):
    # create folder
    # 创建文件夹的根路径
    output_path = os.path.join(opt['project_root'], opt['output_dir_name'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for subject in subject_list: # 遍历主题列表
        out_subject_path = os.path.join(output_path, subject) # 主题对应的文件夹路径
        if not os.path.exists(out_subject_path):
            os.mkdir(out_subject_path)
        subject_abfcm_out = os.path.join(out_subject_path, 'abfcm_out') # abcfm 输出文件夹路径
        if not os.path.exists(subject_abfcm_out):
            os.mkdir(subject_abfcm_out)
        subject_abfcm_nms_path = os.path.join(out_subject_path, 'abfcm_nms') # abcfm_nms 文件夹路径
        if not os.path.exists(subject_abfcm_nms_path):
            os.mkdir(subject_abfcm_nms_path)
        subject_abfcm_final_result_path = os.path.join(
            out_subject_path, 'sub_abfcm_final_result') # abcfm 最终结果文件夹路径
        if not os.path.exists(subject_abfcm_final_result_path):
            os.mkdir(subject_abfcm_final_result_path)

# 多个并行处理函数
# 调用了Process中的multiprocess.Process模块；四个定义的函数过程相同，不同的是target函数
#训练模型
def abfcm_train_mul_process(subject_group, opt):
    print("abfcm abfcm_train_mul_process ------ start: ")
    print("abfcm_training_lr: ", opt["abfcm_training_lr"])
    print("abfcm_weight_decay: ", opt["abfcm_weight_decay"])
    print("abfcm_lr_scheduler: ", opt["abfcm_lr_scheduler"])
    print("abfcm_apex_gamma: ", opt["abfcm_apex_gamma"])
    print("abfcm_apex_alpha: ", opt["abfcm_apex_alpha"])
    print("abfcm_action_gamma: ", opt["abfcm_action_gamma"])
    print("abfcm_action_alpha: ", opt["abfcm_action_alpha"])

    process = []
    start_time = datetime.now()
    for subject_list in subject_group:
        p = multiprocessing.Process(target=abfcm_train_group,
                                    args=(opt, subject_list)) #创建进程
        p.start() # 启动进程，调用进程中的run()方法
        # run()，进程启动时运行的方法，正是它去调用target指定的函数，我们自定义类的类中一定要实现该方法
        process.append(p) # 将新的 p 对象（一个进程）添加到 process 列表中
        time.sleep(1)
    for p in process:
        p.join()
        # join([timeout]) 主线程等待子线程终止。timeout为可选择超时时间；
        # 需要强调的是，p.join只能join住start开启的进程，而不能join住run开启的进程
    delta_time = datetime.now() - start_time
    print("abfcm abfcm_train_mul_process ------ sucessed: ")
    print("time: ", delta_time)

# 输出处理结果
def abfcm_output_mul_process(subject_group, opt):
    print("abfcm_output_mul_process ------ start: ")
    print("micro_apex_score_threshold: ", opt["micro_apex_score_threshold"])
    print("macro_apex_score_threshold: ", opt["macro_apex_score_threshold"])
    process = []
    start_time = datetime.now()
    for subject_list in subject_group:
        p = multiprocessing.Process(target=abfcm_output_group,
                                    args=(opt, subject_list))
        p.start()
        p.join()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("abfcm_output_mul_process ------ sucessed: ")
    print("time: ", delta_time)

# 执行 NMS（非极大值抑制）
def abfcm_nms_mul_process(subject_list, opt):
    print("abfcm_nms ------ start: ")
    print("nms_top_K: ", opt["nms_top_K"])
    process = []
    start_time = datetime.now()
    for subject in subject_list:
        p = multiprocessing.Process(target=abfcm_nms, args=(opt, subject))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("abfcm_nms ------ sucessed: ")
    print("time: ", delta_time)

# 执行IOU（交并比）处理
def abfcm_iou_mul_process(subject_list, opt):
    print("abfcm_iou_process ------ start: ")
    process = []
    start_time = datetime.now()
    for subject in subject_list:
        p = multiprocessing.Process(target=abfcm_iou_process,
                                    args=(opt, subject))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("abfcm_iou_process ------ sucessed: ")
    print("time: ", delta_time)


if __name__ == "__main__":
    set_seed(seed=42) #设置随机种子，用于随机数的可重现性

    # 解析命令行参数
    parser = ArgumentParser() # 创建一个参数解析器对象，用于管理命令行参数的解析
    # 定义了两个命令行参数，分别是 --dataset 和 --mode。可以在命令行中使用这两个参数来指定程序的行为或配置选项。
    parser.add_argument("--dataset")
    parser.add_argument("--mode")
    # 解析命令行参数并将其存储在 args 对象中。在程序运行时，可以通过 args.dataset 和 args.mode 来访问相应的命令行参数的值
    args = parser.parse_args()

    # 通过读取 config.yaml 文件加载配置信息
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        #如果命令行参数 --dataset 被指定，它会覆盖配置文件中的 dataset 值；否则，将使用配置文件中的 dataset 值作为默认值。
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        # 从 yaml_config 中选择特定的 dataset，并将其作为当前工作的数据集配置信息，存储在 opt 变量中。
        opt = yaml_config[dataset]
    subject_list = opt['subject_list']

    # 如果命令行参数 --mode 被指定，它会覆盖 opt 中的 mode 值
    if args.mode is not None:
        opt["mode"] = args.mode

    create_folder(opt) # 根据得到的配置信息创建所需的文件夹结构

    print(f"===================== Dataset is {dataset} =====================")

    # 如果数据集不是 "cross" 类型，将主题列表划分成多个子组，然后根据不同的模式选择不同的处理函数执行：
    if dataset != "cross":
        # 根据条件划分主题列表（subject_list）为多个子组（subject_group）
        tmp_work_numbers = 5 # 表示将主题列表划分为多少个子组
        subject_group = []
        # 如果恰好能划分成tmp_work_numbers个子组
        if len(subject_list) % tmp_work_numbers == 0:
            # 根据主题列表的长度和设定的子组数量，计算每个子组应该包含的主题数量
            len_per_group = int(len(subject_list) // tmp_work_numbers)
            # 划分子组
            for i in range(tmp_work_numbers):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
        # 如果不能恰好划分成tmp_work_numbers个子组
        else:
            len_per_group = int(len(subject_list) // tmp_work_numbers) + 1
            last_len = len(subject_list) - len_per_group * (tmp_work_numbers - 1)
            for i in range(tmp_work_numbers - 1):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
            subject_group.append(subject_list[-last_len:])

        if opt["mode"] == "abfcm_train_mul_process":
            abfcm_train_mul_process(subject_group, opt)
        elif opt["mode"] == "abfcm_output_mul_process":
            abfcm_output_mul_process(subject_group, opt)
        elif opt["mode"] == "abfcm_nms_mul_process":
            abfcm_nms_mul_process(subject_list, opt)
        elif opt["mode"] == "abfcm_iou_mul_process":
            abfcm_iou_mul_process(subject_list, opt)
        elif opt["mode"] == "abfcm_final_result":
            print("abfcm_final_result ------ start: ")
            # abfcm_final_result(opt, subject_list)
            # smic doesn't have macro label
            if dataset != "smic":
                abfcm_final_result_per_subject(opt, subject_list, type_idx=1)
            abfcm_final_result_per_subject(opt, subject_list, type_idx=2)
            abfcm_final_result_per_subject(opt, subject_list, type_idx=0)
            abfcm_final_result_best(opt, subject_list, type_idx=0)
            # abfcm_final_result_best(opt, subject_list, type_idx=1)
            # abfcm_final_result_best(opt, subject_list, type_idx=2)
            print("abfcm_final_result ------ successed")
    else:
        abfcm_train_and_eval(opt)
