import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split



def calculate_f(user_like, user_count, movie_count):

    # 计算节点相似性矩阵
    u = user_like/user_count.reshape([-1, 1])
    v = user_like/movie_count.reshape([1, -1])
    D = np.dot(u.T, v)
    # f: 预测用户可能喜欢的电影 值越大 代表越感兴趣
    f = np.dot(D, user_like.T).T
    return f

def cross_validation(test, f, movie_id):

    # 降序排列
    predict_sort = np.argsort(-f, axis=1)

    R = []
    for user in test.itertuples():
        if user.rating < 3:
            continue
        predict = predict_sort[user.userId-1, :]
        # 找出当前行user观看的movie在预测中的位序
        mid = np.where(movie_id == user.movieId)[0][0]
        loc = np.where(predict == mid)[0][0]
        # r：距离因子
        r = loc / len(predict)
        R.append(r)
    # 计算指标，meanR
    meanR = np.array(R).mean()
    print('meanR={}'.format(meanR))
    print('meanR越小， 预测越精准')


def draw_ROC(f, user_like, user_dis):

    # 降序排列
    predict_sort = np.argsort(-f, axis=1)

    th = np.linspace(0, 1, 50)
    TPR = np.zeros(50)
    FPR = np.zeros(50)
    for i, thres in enumerate(th):
        predict_num = int(thres * user_like.shape[1])
        tpr = np.zeros(user_like.shape[0])
        fpr = np.zeros(user_like.shape[0])
        for user in range(user_like.shape[0]):
            predict = predict_sort[user, :predict_num]
            like = np.where(user_like[user, :] == 1)[0]  # 实际喜欢
            dis = np.where(user_dis[user, :] == 1)[0]  # 实际不喜欢
            predict_like = np.intersect1d(predict, like)  # 预测喜欢
            predict_dis = np.intersect1d(predict, dis)  # 预测不喜欢

            if len(like) == 0:
                tpr[user] = 0
            else:
                tpr[user] = len(predict_like)/len(like)
            if len(dis) == 0:
                fpr[user] = 0
            else:
                fpr[user] = len(predict_dis)/len(dis)

        TPR[i] = tpr.mean()
        FPR[i] = fpr.mean()
    roc_area = auc(FPR, TPR)
    print('ROC(面积为%0.2f)' % roc_area)
    plt.plot(FPR, TPR, color='b')
    plt.plot([0, 1], [0, 1], c='r', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def body(data):
    train, test, _, _ = train_test_split(data, data['userId'], test_size=0.1)

    users = data['userId']
    movies = data['movieId']

    user_count = np.array(users.value_counts())  # 生成用户信息表，存放用户观看电影次数
    movie_count = np.array(movies.value_counts())  # 生成电影信息表，存放电影被观看次数
    movie_id = np.array(movies.value_counts().index)  # 生成电影序列表，存放电影ID

    user_num = user_count.shape[0]  # 总用户数
    movie_num = movie_count.shape[0]  # 总电影数

    # 生成user对movie的评分表
    user_movie = np.zeros([user_num, movie_num])
    print('user_movie\'s shape is:{}'.format(user_movie.shape))

    for item in train.itertuples():
        user_movie[item.userId-1, np.where(movie_id == item.movieId)[0][0]] = item.rating

    user_like = (user_movie > 3) + 0
    user_dis = (user_movie <= 3) + 0

    f = calculate_f(user_like, user_count, movie_count)
    cross_validation(test, f, movie_id)
    draw_ROC(f, user_like, user_dis)

if __name__ =='__main__':
    data = pd.read_table('ratings.dat', header=None,
                         names=['userId', 'movieId', 'rating', 'timestamp'],
                         sep='::',
                         engine='python')
    body(data)