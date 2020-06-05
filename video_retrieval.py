from os import listdir

from ImageNNExtract import *
from retrieval_util import *


# 帧级匹配
def retrieval(query_numpy_path, candidate_refer_numpy_path, query_frame_2_time=None, refer_frame_2_time=None,
              top_K=20):  # 这里用candicate第意义就是 当前还是对所有数据库中的视频计算相似度，这样会进行许多无用功
    # 后面可以考虑进行处理选取一部分候选集来计算相似度 缩小计算空间 而不是计算整个库
    # 粗筛 此处得到query与所有refer的欧氏距离 升序排序
    query_feature = np.load(query_numpy_path)
    refer_candicates_dict = {}
    r_score = []
    for path in listdir(candidate_refer_numpy_path):
        if not os.path.isdir(candidate_refer_numpy_path + path):
            refer_feature = np.load(candidate_refer_numpy_path + path)
            refer_candicates_dict[candidate_refer_numpy_path + path] = refer_feature
    for refer_video, refer_feature in refer_candicates_dict.items():
        idxs, unsorted_dists, sorted_dists = compute_dists(query_feature, refer_feature)
        score = np.sum(sorted_dists[:, 0])
        r_score.append((score, refer_video))
    r_score.sort(key=lambda x: x[0], reverse=False)
    # 细筛 帧级匹配
    top_K = 20
    q_ans = []
    for k, (_, r_vid) in enumerate(r_score):
        # q_baseaddr = train_query_vid2baseaddr[q_vid]

        if (k >= top_K):
            break
        refer_feature = refer_candicates_dict[r_vid]
        path_q, path_r, score = get_frame_alignment(query_feature, refer_feature, top_K=3, min_sim=0.85, max_step=10)
        if len(path_q) > 0:
            time_q = [query_frame_2_time[qf_id + 1] for qf_id in path_q]
            time_r = [refer_frame_2_time[r_vid.split('.')[0].split('/')[1]][rf_id + 1] for rf_id in path_r]
            q_ans.append((score, r_vid, time_q[0], time_q[-1], time_r[0], time_r[-1]))
            print(len(q_ans))
    q_ans.sort(key=lambda x: x[0], reverse=True)
    print(q_ans)


if __name__ == '__main__':
    # path ='data_base/'
    # net = Img2Vec()  # Img2Vec 是封装好网络的类
    # feature = net.get_vec("to_query/b62a8b88-b8cb-11e9-930e-fa163ee49799")  # video是关键帧所在的文件夹
    # print(feature.shape)
    # feature = normalize(feature)
    # np.save("62a8b88-b8cb-11e9-930e-fa163ee49799", feature)
    #
    # for dir in listdir('data_base'):
    #     frame_path = path + dir
    #     feature = net.get_vec(frame_path)
    #     feature = normalize(feature)
    #     np.save(dir,feature)

    # !!!!!!!!!!!!!!!!!! 请在按时间顺序将帧排好序后再进行特征提取
    database = 'data_base/'
    query_frame_path = 'to_query/b62a8b88-b8cb-11e9-930e-fa163ee49799/'
    top_K = 3

    query_frame_2_time = {}

    refer_frame_2_time = {}
    # get query_frame_2_time
    index = 1
    query_frame_path_list = listdir(query_frame_path)
    query_frame_path_list.sort()
    for frame_path in listdir(query_frame_path):
        query_frame_2_time[index] = index #这里因为是隔一秒抽帧 所以每一帧对应的是1,2,3,4,.....
        index += 1

    #get refer_frame_2_time  这里因为refer video是数据库里所有的video 所以需要二重循环
    for refer_video_path in listdir(database):
        if os.path.isdir(database+refer_video_path):
            refer_frame_path = listdir(database+refer_video_path)
            refer_frame_path.sort()
            refer_frame_2_time_for_one_video = {}
            index = 1
            for refer_frame in refer_frame_path:
                refer_frame_2_time_for_one_video[index] = index
                index += 1 #这里因为是隔一秒抽帧 所以每一帧对应的是1,2,3,4,.....
            refer_frame_2_time[refer_video_path] = refer_frame_2_time_for_one_video


    retrieval('to_query/62a8b88-b8cb-11e9-930e-fa163ee49799.npy', 'data_base/',query_frame_2_time,refer_frame_2_time)
    # print(listdir('data_base'))
