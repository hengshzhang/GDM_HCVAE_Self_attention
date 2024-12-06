import math
import torch.utils.data
from class_D_HCVAE import CVAE
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans

beta = 0.9
beta_consensus = 0.9
dist_consensus = 4.0
Length = 101
Order = 8
Cardinal = 8
lr = 1e-5  # learning rate
train_time = 2000

def linguistic_matrix_translation(matrix):

    matrix_order = len(matrix)

    linguistic_matrix = np.zeros((matrix_order, matrix_order))

    for i in range(matrix_order):
        for j in range(matrix_order):
            linguistic_matrix[i, j] = matrix_order/2 + 2*np.log(matrix[i, j])/np.log(2)
    return linguistic_matrix


class ExpertMatrixPreprocess(object):
    def __init__(self):
        self.expert_matrixs = torch.zeros(Length, Order, Order)

    """Read the original data from excel files"""

    def obtain_expert_matrix(self):
        matrix_size = Order
        data_length = Length + 1
        for i in range(1, data_length):
            h = i + (i - 1) * matrix_size
            expert_matrixs_dataframe = pd.read_excel(r"./datas/expert_matrix.xls", sheet_name="bettermatrix", header=h,
                                                     usecols="B:I",
                                                     nrows=matrix_size)
            expert_matrixs_dataframe.head()
            expert_matrixs_numpyArray = expert_matrixs_dataframe.to_numpy()
            # expert_matrixs[i] = torch.from_numpy(expert_matrixs_numpyArray)
            self.expert_matrixs[i - 1] = torch.tensor(expert_matrixs_numpyArray)
            # print(i-1, self.expert_matrixs[i-1])
        return self.expert_matrixs

    """Read the original data from the another type excel files"""

    def read_expert_matrix_from_excel(self):
        # 读取Excel文件的第四行到第11行的第E列到第L列
        temp_expert_matrixs = torch.zeros(Length, Order, Order)
        data_length = Length + 1
        for i in range(1, data_length):
            file_path = "./New data4/" + "sale (" + str(i) + ")" + ".xls"
            temp_matrix = pd.read_excel(file_path, header=None,
                                        skiprows=3, usecols="E:L", nrows=8)
            temp_expert_matrixs[i - 1] = torch.tensor(np.array(temp_matrix))
        for i in range(Length):
            for k in range(Order):
                for l in range(Order):
                    self.expert_matrixs[i, k, l] = pow(math.sqrt(2), temp_expert_matrixs[i, k, l] - Cardinal / 2)
        return self.expert_matrixs



class MatrixSetLoader(object):
    def __init__(self):
        self.expert_matrixs = ExpertMatrixPreprocess()
        # self.length = len(matrix_iter)

    # for i in range(0, Length):
    """Obtain the set of the matrices"""

    def matrix_dataset(self):
        # expert_matrixs = expert_matrix_process()
        # datasets_expert_matrix = self.expert_matrixs.obtain_expert_matrix()
        datasets_expert_matrix = ExpertMatrixPreprocess().read_expert_matrix_from_excel()
        # consistent_matrix = self.expert_matrixs.calculation_consistency_matrix()
        # datasets_expert_matrix = TensorDataset(expert_matrix)
        return datasets_expert_matrix

    """Calculation to the consistency matrix set of original matrix set which is existed at the theory"""

    def calculation_consistency_matrix_iter(self, matrix_iter):
        matrix_size = Order
        # data_length = Length + 1
        expert_matrix = matrix_iter
        data_length = len(matrix_iter)
        # print(expert_matrix[1])
        consistent_expertmatrix_iter = torch.zeros(data_length, matrix_size, matrix_size)
        geometric = np.zeros(matrix_size)
        consistency_matrix = np.zeros((matrix_size, matrix_size))
        # geometric_sum=0
        for i in range(1, data_length + 1):
            temp_matrix = np.array(expert_matrix[i - 1])
            for j in range(0, matrix_size):
                length = matrix_size
                geometric[j] = (temp_matrix[[j]].prod()) ** (1 / length)
                # print(geometric[j])
            for k in range(0, matrix_size):
                for l in range(0, matrix_size):
                    consistency_matrix[k, l] = geometric[k] / geometric[l]
                    # print(consistency_matrix[k,l])
            consistent_expertmatrix_iter[i - 1] = torch.tensor(consistency_matrix)
        return consistent_expertmatrix_iter

    """Random split the data set as the train data set and valid data set"""

    def dataloader(self):
        datasets_expert_matrix = self.matrix_dataset()
        # n_train = int(len(datasets_expert_matrix)*0.8)
        # n_valid = len(datasets_expert_matrix)-n_train
        # expert_matrix_train,expert_matrix_valid = random_split(datasets_expert_matrix,[n_train,n_valid])
        # #expert_matrix_train = torch.zeros(Length-1,Order,Order)
        expert_matrix_train = datasets_expert_matrix
        expert_matrix_valid = datasets_expert_matrix
        # print(len(datasets_expert_matrix))
        # print(len(expert_matrix_train))
        return expert_matrix_train, expert_matrix_valid



class MatrixInformationCalculation(object):
    """The calculation to the relation information of matrix set"""
    """The function is translated the original matrix into the normalize matrix 
    in which the values are in interval [0,1]"""

    def matrix_normalize(self, matrix, order):
        normalize_matrix = torch.zeros(order, order)
        for i in range(0, order):
            for j in range(0, order):
                temp = 0.5 * (1 + math.log(matrix[i, j], order / 2))
                if temp > 1: temp = 1
                if temp < 0: temp = 0
                normalize_matrix[i, j] = temp
        return normalize_matrix

    """Translate the normalize matrix to the original matrix"""

    @staticmethod
    def reverse_normalize_vector(vector, order):
        temp_vector = vector
        v_size = len(temp_vector)
        reverse_vector = torch.zeros(v_size)
        for i in range(v_size):
            base = order / 2
            index = 2 * temp_vector[i] - 1
            temp_value = math.pow(base, index)
            reverse_vector[i] = temp_value
        return reverse_vector

    """Translate the matrix to the upper triangle vector"""


    def matrix_to_uptringle_vector(self, matrix, order):
        temp_matrix = matrix
        r = order
        vector_size = int(0.5 * (r) * (r - 1))
        # print(vector_size)
        uptringle_vector = torch.zeros(vector_size)
        k = 0
        for i in range(0, r - 1):
            for j in range(i + 1, r):
                uptringle_vector[k] = temp_matrix[i][j]
                k += 1
        return uptringle_vector

    """Translate the upper triangle vector as the matrix """


    @staticmethod
    def uptriangle_vector_to_rematrix(vector, vector_size, order):
        uptriangle_vector = vector
        l = vector_size
        n = order
        re_matrix = torch.zeros(n, n)
        k = 0
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                if k >= l: break
                re_matrix[i][j] = uptriangle_vector[k]
                re_matrix[j][i] = 1 / re_matrix[i][j]
                k += 1
            re_matrix[i][i] = 1.0
        re_matrix[n - 1][n - 1] = 1.0
        return re_matrix

    """Computation of consistency index to the matrix"""

    def calculat_matrix_ci(self, original_matrix, original_matrix_star):
        ci_matrix = np.zeros((len(original_matrix), len(original_matrix)))
        matrix_order = len(original_matrix)
        for i in range(matrix_order):
            for j in range(matrix_order):
                ci_matrix[i][j] = original_matrix[i][j] / original_matrix_star[i][j]
        sum_list = ci_matrix.sum(1)
        temp_matrix_ci = sum(sum_list) / len(sum_list)
        matrix_ci = (temp_matrix_ci - matrix_order) / (matrix_order - 1)
        return matrix_ci

    """Calculation of the theory consistent matrix to each matrix"""

    def calculat_matrix_consistent(self, matrix, order):
        matrix_size = order
        temp_matrix = matrix
        geometric = np.zeros(matrix_size)
        matrix_consistent = np.zeros((matrix_size, matrix_size))
        # geometric_sum=0
        for j in range(0, matrix_size):
            length = matrix_size
            geometric[j] = (temp_matrix[[j]].prod()) ** (1 / length)
            # print(geometric[j])
        for k in range(0, matrix_size):
            for l in range(0, matrix_size):
                matrix_consistent[k, l] = geometric[k] / geometric[l]
                # print(consistency_matrix[k,l])
        return matrix_consistent

    """Comparing with the original matrix, analysis the changes of the order of the factors
     which is calculated via the generation expert matrix"""

    def order_analysis_to_expert_matrix_generation(self, original_matrix, generate_matrix):
        matrix_order = len(original_matrix)
        geometric_mean_of_original_matrix = np.zeros(matrix_order)
        geometric_mean_of_generate_matrix = np.zeros(matrix_order)
        sum_of_original = 0
        sum_of_generate = 0
        for i in range(matrix_order):
            geometric_mean_of_original_matrix[i] = np.prod(np.array(original_matrix[i])) ** (1 / matrix_order)
            geometric_mean_of_generate_matrix[i] = np.prod(np.array(generate_matrix[i])) ** (1 / matrix_order)
            sum_of_original += geometric_mean_of_original_matrix[i]
            sum_of_generate += geometric_mean_of_generate_matrix[i]
        for j in range(matrix_order):
            geometric_mean_of_original_matrix[j] = geometric_mean_of_original_matrix[j] / sum_of_original
            geometric_mean_of_generate_matrix[j] = geometric_mean_of_generate_matrix[j] / sum_of_generate
        original_matrix_order = np.argsort(geometric_mean_of_original_matrix)[::-1]
        generate_matrix_order = np.argsort(geometric_mean_of_generate_matrix)[::-1]
        print("original_matrix_order", original_matrix_order)
        print(f'ranking_values_of_original_matrix', geometric_mean_of_original_matrix)
        # print("geometric_mean_of_original_matrix",geometric_mean_of_original_matrix)
        print("generate_matrix_order", generate_matrix_order)
        print(f'ranking_values_of_generate_matrix', geometric_mean_of_generate_matrix)
        # print("geometric_mean_of_generate_matrix",geometric_mean_of_generate_matrix)


    def candidates_sort_with_geometric_mean(self, matrix):
        matrix_order = len(matrix)
        geometric_mean = torch.zeros(matrix_order)
        for i in range(matrix_order):
            geometric_mean[i] = torch.pow(torch.prod(matrix[i]), 1/matrix_order)
        candidates_sort, sort_index = torch.sort(geometric_mean)
        print(f'Rank of candidates:', sort_index)



class MatrixSetConsensusCalculation(object):
    """Calculation to the relation information of consensus to matrix set"""
    """The number calculation of the vector_set in which the element is less than threshold alpha"""

    def number_count_to_vector(self, vector, alpha):
        vector_number = 0
        length = len(vector)
        for i in range(length):
            if vector[i] <= alpha: vector_number += 1
        return vector_number

    """Calculation of the mean matrix to the matrix set"""

    def matrix_iter_mean(self, matrix_iter):
        matrix_size = Order
        mean_matrix = torch.zeros(matrix_size, matrix_size)
        data_length = len(matrix_iter)
        for i in range(data_length):
            mean_matrix += (1 / data_length) * matrix_iter[i]
        return mean_matrix

    """Calculation to the distance between the mean matrix and matrix of matrix set"""

    def process_matrix_iter_distance(self, matrix_iter):
        mean_matrix = self.matrix_iter_mean(matrix_iter)
        datalength = len(matrix_iter)
        distance_iter = torch.zeros(datalength)
        for i in range(datalength):
            # distance_iter[i] = np.sqrt(np.sum(np.square(mean_matrix - matrix_iter[i])))
            distance_iter[i] = torch.tensor(np.linalg.norm(mean_matrix - matrix_iter[i]))
        # median_distance = torch.median(distance_iter)
        # mean_distance = torch.mean(distance_iter)
        # count_ditance = number_count_to_vector(distance_iter, mean_distance + 2)
        savefile = 'res/'
        file_name = savefile + 'matrix_distance_to_mean.csv'
        df = pd.DataFrame(distance_iter)
        df.to_csv(file_name, index=False)
        return distance_iter

    """Calculation to the distance between the final_consensus_matrix matrix and matrix of matrix set"""

    def matrix_set_distance_to_final_matrix(self, matrix_set, final_consensus_matrix):
        final_matrix = final_consensus_matrix
        datalength = len(matrix_set)
        distance_iter = torch.zeros(datalength)
        for i in range(datalength):
            # distance_iter[i] = np.sqrt(np.sum(np.square(mean_matrix - matrix_iter[i])))
            distance_iter[i] = torch.tensor(np.linalg.norm(final_matrix - matrix_set[i]))
        # median_distance = torch.median(distance_iter)
        # mean_distance = torch.mean(distance_iter)
        # count_ditance = number_count_to_vector(distance_iter, mean_distance + 2)
        return distance_iter


    """Matrix_set clustering via K-means"""

    def consistent_matrix_clustering(self, consistent_matrix, matrix_order, cluster_number):
        matrix_number = len(consistent_matrix)
        r = matrix_order
        vector_size = int(0.5 * (r) * (r - 1))
        vector_to_matrix = np.zeros((matrix_number, vector_size))
        for i in range(matrix_number):
            temp_matrix = consistent_matrix[i]
            vector_to_matrix_temp = MatrixInformationCalculation(). \
                matrix_normalize(temp_matrix, matrix_order)
            vector_to_matrix[i] = MatrixInformationCalculation().matrix_to_uptringle_vector(vector_to_matrix_temp, matrix_order)
        # print(vector_to_matrix)
        # matrix_label, matrix_centroids = spectral_clustering().data_set_spectral_clustering(vector_to_matrix,4)
        result = KMeans(n_clusters=cluster_number, max_iter=100)
        result.fit(vector_to_matrix)
        matrix_label = result.labels_
        matrix_centroids = result.cluster_centers_
        print("matrix_label", matrix_label)
        # print("matrix_centroids",matrix_centroids)
        return matrix_label, matrix_centroids

    """Hierarchical calculation of the final consensus matrix via the clustering of matrix set"""

    def final_matrix_calculation_optimize_attention(self, matrix_data_set, matrix_order, cluster_number):
        m = cluster_number
        matrix_labels, matrix_centroids = self.consistent_matrix_clustering(matrix_data_set, matrix_order,
                                                                            cluster_number)
        matrix_number = len(matrix_data_set)
        cluster_temp_matrix_set = []
        total_temp_matrix_set = []
        generating_matrix_to_self_attention = torch.zeros(matrix_number, matrix_order, matrix_order)
        cluster_indices = []
        for k in range(m):
            count = 0
            cluster_temp_matrix_set.clear()
            cluster_indices.clear()
            for i in range(matrix_number):
                if matrix_labels[i] == k:
                    cluster_temp_matrix_set.append(matrix_data_set[i])
                    count += 1
                    print(f'cluster{k}:', i)
                    cluster_indices.append(i)
            temp_total_matrix_to_cluster, generating_matrix_set = \
                self.consensus_matrix_with_optimal_attention(cluster_temp_matrix_set, matrix_order)
            cluster_set_distance_to_fusion_matrix = \
                self.matrix_set_distance_to_final_matrix(generating_matrix_set, temp_total_matrix_to_cluster)
            print(f'cluster_set_distance_to_fusion_matrix', cluster_set_distance_to_fusion_matrix)
            total_temp_matrix_set.append(temp_total_matrix_to_cluster)
            middle_index = 0
            for j in cluster_indices:
                generating_matrix_to_self_attention[j] = generating_matrix_set[middle_index]
                middle_index += 1
        final_consensus_matrix, final_generating_matrix_set = \
            self.consensus_matrix_with_optimal_attention(total_temp_matrix_set, matrix_order)
        distance_to_final_generating_matrix_set = \
            self.matrix_set_distance_to_final_matrix(final_generating_matrix_set, final_consensus_matrix)
        print(f'distance_to_final_generating_matrix_set', distance_to_final_generating_matrix_set)
        return final_consensus_matrix, generating_matrix_to_self_attention

    """Consensus matrix calculation via the self attention to the matrix reaching consensus, 
    and the mean matrix calculation is optimized"""

    def consensus_matrix_with_optimal_attention(self, matrix_data_set, matrix_order):
        matrix_number = len(matrix_data_set)
        r = matrix_order
        vector_size = int(0.5 * (r) * (r - 1))
        vector_set = torch.zeros(matrix_number, vector_size)
        original_vector_set = torch.zeros(matrix_number, vector_size)
        final_generating_matrix = torch.zeros(matrix_number, matrix_order, matrix_order)
        for i in range(matrix_number):
            original_vector_set[i] = MatrixInformationCalculation().matrix_to_uptringle_vector(matrix_data_set[i], r)
        for i in range(matrix_number):
            vector_set[i] = original_vector_set[i]
        mean_vector, final_vector_set = SelfAttentionWithOptimize().self_attention_calculation(vector_set, matrix_number)
        mean_matrix = MatrixInformationCalculation(). \
            uptriangle_vector_to_rematrix(mean_vector, vector_size, matrix_order)
        for i in range(matrix_number):
            final_generating_matrix[i] = MatrixInformationCalculation()\
                .uptriangle_vector_to_rematrix(final_vector_set[i], vector_size, matrix_order)
        print("mean_matrix_with_optimal_attention:", mean_matrix)
        return mean_matrix, final_generating_matrix


class SelfAttentionWithOptimize(object):
    """The class used to calculate of the self attention of matrix with optimization"""
    """Given the data set and corresponding weight vector, make calculation of the average data"""

    def calculation_average_of_data_set(self, data_set, weight_vector):
        vector_length = len(data_set[0])
        vector_number = len(data_set)
        average_of_data_set = torch.zeros(vector_length)
        for i in range(vector_number):
            average_of_data_set += weight_vector[i] * data_set[i]
        return average_of_data_set

    """The differance calculation to the data"""

    @staticmethod
    def calculation_distance(y_true, y_pred):
        """compute distance between the matrices or vectors"""
        data_number = len(y_true)
        length = 0
        for i in range(data_number):
            if y_true[i] > 0.00001: length += 1
        temp_sum = 0
        for i in range(length):
            temp_sum += pow(y_true[i] - y_pred[i], 2)
        return math.sqrt(temp_sum)

    """The optimal update of the combination of data set """

    def calculation_combination_of_data_set(self, data_set, weight_vector):
        # vector_length = len(data_set[0])
        """Initialize the relation parameters:"""
        vector_number = len(data_set)
        calculated_weight_vector = torch.zeros(vector_number)
        update_weight_vector = torch.zeros(vector_number)
        r = int(0.5 * (Order-1) * Order)
        combination_of_data_tmp = torch.zeros(r)
        """Calculation of the last combination of data set:"""
        for i in range(vector_number):
            combination_of_data_tmp += data_set[i] * weight_vector[i]
        combination_of_data_last = combination_of_data_tmp

        """Update the weight vector of combination of data set via learning:"""
        learning_rate = 1e-3
        # train_times = 1
        # for epoch in range(train_times):
        for i in range(vector_number):
            update_weight_vector[i] = \
                1 / (1 + torch.linalg.norm(data_set[i] - combination_of_data_last))
            calculated_weight_vector[i] = weight_vector[i] + learning_rate * update_weight_vector[i]

        """calculated_weight_vector normalization:"""
        weight_sum = 0
        for k in range(vector_number):
            weight_sum += calculated_weight_vector[i]
        for i in range(vector_number):
            calculated_weight_vector[i] = calculated_weight_vector[i] / weight_sum

        """Calculation of the current combination of data set:"""
        combination_of_data_tmp = torch.zeros(r)
        for i in range(vector_number):
            combination_of_data_tmp += data_set[i] * calculated_weight_vector[i]
        combination_of_data_current = combination_of_data_tmp

        return combination_of_data_current, calculated_weight_vector

    """Calculation of data set, which is the average of the differences 
    between the data and the mean to date set"""

    def calculation_consensus_of_data_set(self, data_set):
        matrix_number = len(data_set)
        vector_length = len(data_set[0])
        combination_data_mean = torch.zeros(vector_length)
        for p in range(matrix_number):
            combination_data_mean += data_set[p] / matrix_number
        consensus_to_set = 0
        for q in range(matrix_number):
            consensus_of_data = self.calculation_distance(combination_data_mean, data_set[q])
            #consensus_to_set += consensus_of_data / matrix_number
            consensus_to_set += consensus_of_data
        return consensus_to_set

    """Each data set is changed via the cosine similarity between the data and differance of the 
       data and combination of data, and the cosine similarity is computed as weight of the data
       set via sigmoid function and normalization. Especially, the computation of combination of
       data is optimized"""

    def self_attention_calculation(self, data_set, combination_number, attention_number=10):
        #global final_combination_data
        vector_length = len(data_set[0])
        matrix_number = combination_number
        print(f'self_attention_calculation matrix number', matrix_number)
        projection_of_combination_data = torch.zeros(matrix_number, vector_length)
        differance_of_combination_data = torch.zeros(matrix_number, vector_length)
        mean_of_data_set = torch.zeros(vector_length)
        # final_combination_data_set = torch.zeros(matrix_number, vector_length)
        for i in range(matrix_number):
            projection_of_combination_data[i] = data_set[i]
        initial_consensus_of_set = self.calculation_consensus_of_data_set(projection_of_combination_data)
        print(f'initial_consensus_of_data_set', initial_consensus_of_set)
        train_time = attention_number
        weight_of_data_set = torch.ones(matrix_number) / matrix_number
        combination_of_data = self.calculation_average_of_data_set(projection_of_combination_data,
                                                                   weight_of_data_set)
        mean_of_data_set = combination_of_data
        for epoch in range(train_time):
            weight_of_combination_data = torch.ones(matrix_number, matrix_number) / matrix_number
            for j in range(matrix_number):
                # projection_of_data_set = (projection_of_data_set + projection_of_generate_data[j]) * 0.5
                differance_of_combination_data[j] = projection_of_combination_data[j] + combination_of_data
                # average_of_generate_data = calculation_average_of_data_set(attention_of_generate_data, weight_of_data_sum)
                # for j in range(original_model_number+1, model_number+1):
                weight_sum = 0
                for k in range(matrix_number):
                    vector_similarity = 2*(1+torch.cosine_similarity(differance_of_combination_data[j],
                                                                     data_set[k], dim=0))
                    weight_of_combination_data[j, k] = 1 / (1 + torch.exp(-vector_similarity))
                    #weight_of_combination_data[j, k] = vector_similarity
                    weight_sum += weight_of_combination_data[j, k]
                for k in range(matrix_number):
                    weight_of_combination_data[j, k] = weight_of_combination_data[j, k] / weight_sum
                average_projection_of_data = self.calculation_average_of_data_set(data_set,
                                                                                  weight_of_combination_data[j])
                projection_of_combination_data[j] = average_projection_of_data
                # projection_of_combination_data_rmse =self.rmse(combination_of_data, projection_of_combination_data[j])
                # print(f'projection_of_combination_data{j}_rmse', projection_of_combination_data_rmse)
            consensus_of_combination = self.calculation_consensus_of_data_set(projection_of_combination_data)
            print(f'combination_data_consensus{epoch}', consensus_of_combination)
            if epoch == train_time - 1:
                weight_of_data_set = torch.ones(matrix_number) / matrix_number
                final_combination_data = \
                    self.calculation_average_of_data_set(projection_of_combination_data, weight_of_data_set)
                break
                # test_combination_data = combination_of_data
            for j in range(matrix_number):
                projection_of_combination_data[j] = projection_of_combination_data[j]*0.5 + data_set[j] * 0.5

            combination_of_data, weight_of_data_set = self.calculation_combination_of_data_set(
                projection_of_combination_data, weight_of_data_set)

            aggregation_matrix = MatrixInformationCalculation().\
                uptriangle_vector_to_rematrix(combination_of_data, vector_length, Order)
            geometric_mean = np.zeros(Order)
            if epoch < 3:
                for j in range(Order):
                    geometric_mean[j] = np.prod(np.array(aggregation_matrix[j])) ** (1 / Order)
                matrix_order = np.argsort(geometric_mean)[::-1]
                print(f'{epoch} aggregation_matrix:', aggregation_matrix)
                print(f'{epoch} aggregation_matrix_order:', matrix_order)
        final_combination_data_distance = self.calculation_distance(mean_of_data_set, final_combination_data)
        print(f'final_combination_data_distance', final_combination_data_distance)
        return final_combination_data, projection_of_combination_data


"""Define the validation process of the Discrete HCVAE"""


def valid(model, valid_rawmatrix, matrix_order):
    model.eval()
    matrix_valid = valid_rawmatrix
    valid_loss = 0
    matrix_number = len(matrix_valid)
    re_expert_matrix_iter = torch.zeros(matrix_number, matrix_order, matrix_order)
    for i in range(0, matrix_number):
        temp_x = matrix_valid[i]
        temp_y = MatrixInformationCalculation().calculat_matrix_consistent(temp_x, matrix_order)
        temp_x_ci = MatrixInformationCalculation().calculat_matrix_ci(temp_x, temp_y)
        print("temp_x_ci:", temp_x_ci, "\n")
        normal_x = MatrixInformationCalculation().matrix_normalize(temp_x, matrix_order)
        normal_y = MatrixInformationCalculation().matrix_normalize(temp_y, matrix_order)
        linguistic_x = linguistic_matrix_translation(temp_x)
        if temp_x_ci > 0.115:
            file_name = f"./no_consistency_matrix/matrix{i}.txt"
            file_name1 = f"./no_consistency_normalize/matrix_norm{i}.txt"
            # print(f'file_name', file_name)
            with open(file_name, 'w', newline='', encoding='utf-8') as txtfile:
                for k in range(matrix_order):
                    for p in range(matrix_order):
                        formatted_value = f"{linguistic_x[k, p]:.0f}"
                        txtfile.write("{:<3}".format(formatted_value))
                    txtfile.write('\n')
            with open(file_name1, 'w', newline='', encoding='utf-8') as txtfile1:
                for k in range(matrix_order):
                    for p in range(matrix_order):
                        formatted_value1 = f"{normal_x[k, p]:.4f}"
                        txtfile1.write("{:<8}".format(formatted_value1))
                    txtfile1.write('\n')
        x = MatrixInformationCalculation().matrix_to_uptringle_vector(normal_x, matrix_order)
        y = MatrixInformationCalculation().matrix_to_uptringle_vector(normal_y, matrix_order)
        re_matrices_set = []
        for k in range(2):
            ge_matrices, mu_e, mu_d, sigma = model(x, y)
            re_matrices_set.append(ge_matrices)
        re_matrixs = average_of_vector_set(re_matrices_set)
        re_matrixs_len = len(re_matrixs)
        rv_matrixs = MatrixInformationCalculation().reverse_normalize_vector(re_matrixs, matrix_order)
        re_expert_matrix = MatrixInformationCalculation().uptriangle_vector_to_rematrix(rv_matrixs, re_matrixs_len,
                                                                                        matrix_order)
        re_expert_consistent_matrix = MatrixInformationCalculation().calculat_matrix_consistent(re_expert_matrix,
                                                                                                matrix_order)
        re_expert_matrix_iter[i] = re_expert_matrix
        if i < 20:
            print(f'original {i}th matrix', temp_x)
        print(f'Generated {i}th matrix:\n', re_expert_matrix)
        re_expert_matrix_ci = MatrixInformationCalculation().calculat_matrix_ci(re_expert_matrix,
                                                                                re_expert_consistent_matrix)
        print("re_expert_matrix_ci:", re_expert_matrix_ci, "\n")
        loss = model.loss_function(re_matrixs, x, mu_e, mu_d, sigma)
        valid_loss += loss.item()
        savefile_ci = 'res_CI/'
        file_name_1 = savefile_ci + 'original matrix CI.csv'
        with open(file_name_1, "a") as df_1:
            df_1.write(str(temp_x_ci))
            df_1.write("\n")
        file_name_2 = savefile_ci + 'Discrete HCVAE generating matrix CI.csv'
        with open(file_name_2, "a") as df_2:
            df_2.write(str(re_expert_matrix_ci))
            df_2.write("\n")
    # print(f'epoch:{i}|ValidLoss: ', valid_loss / (matrix_number))
    df_1.close()
    df_2.close()
    return re_expert_matrix_iter

def average_of_vector_set(vector_set):
    vector_set_average = vector_set[0]
    vector_set_length = len(vector_set)
    for i in range(1, vector_set_length):
        vector_set_average += vector_set[i]
    vector_set_average = vector_set_average/vector_set_length
    return vector_set_average

def main():
    """Initialize the parameters and define the model"""
    matrix_order = Order
    r = matrix_order
    vector_size = int(0.5 * (r) * (r - 1))
    latent_size = vector_size  # The latent size of encoder and decoder
    input_matrix_size = input_consistentmatrix_size = vector_size  # The size of input and output data
    model = CVAE(input_matrix_size, input_consistentmatrix_size, latent_size)
    """Load the training and validate data set"""
    dataset_train, dataset_valid = MatrixSetLoader().dataloader()
    #train_consistentmatrix = MatrixSetLoader().calculation_consistency_matrix_iter(dataset_train)
    #valid_consistentmatrix = MatrixSetLoader().calculation_consistency_matrix_iter(dataset_valid)
    length_valid = len(dataset_valid)
    model.load_state_dict(torch.load("./Discrete HCVAE_model_save/Discrete HCVAE_model.pth"))
    model.eval()
    re_consistent_matrix_iter = valid(model, dataset_valid, matrix_order)

    """Analysis the obtaining order changes between the generate matrix and original matrix"""
    for k in range(length_valid):
        MatrixInformationCalculation(). \
            order_analysis_to_expert_matrix_generation(dataset_valid[k], re_consistent_matrix_iter[k])

    """Calculation of the consensus matrix with optimal computation to combination of matrix without
     the clustering"""

    consensus_matrix_with_optimal_attention, generating_matrix_no_clustering = MatrixSetConsensusCalculation(). \
        consensus_matrix_with_optimal_attention(re_consistent_matrix_iter, matrix_order)
    distance_to_consensus_matrix_optimal_attention = MatrixSetConsensusCalculation(). \
        matrix_set_distance_to_final_matrix(generating_matrix_no_clustering, consensus_matrix_with_optimal_attention)
    MatrixInformationCalculation().candidates_sort_with_geometric_mean(consensus_matrix_with_optimal_attention)
    """Calculation of the consensus matrix with optimal computation to combination of matrix with
         the clustering"""
    cluster_number = 10
    final_consensus_matrix_optimal_attention, generating_matrix_with_clustering = MatrixSetConsensusCalculation(). \
        final_matrix_calculation_optimize_attention(re_consistent_matrix_iter, matrix_order, cluster_number)
    matrix_distance_to_optimal_attention = MatrixSetConsensusCalculation(). \
        matrix_set_distance_to_final_matrix(generating_matrix_with_clustering,
                                            final_consensus_matrix_optimal_attention)
    MatrixInformationCalculation().candidates_sort_with_geometric_mean(final_consensus_matrix_optimal_attention)

    # mean_matrix_to_ge_consistency_matrix = MatrixSetConsensusCalculation().matrix_iter_mean(re_consistent_matrix_iter)
    # distances_to_ge_consistency_matrix_mean = MatrixSetConsensusCalculation()\
    #     .matrix_set_distance_to_final_matrix(re_consistent_matrix_iter, mean_matrix_to_ge_consistency_matrix)
    """Print and save the relation information to consensus of matrix"""
    print("distance_to_consensus_matrix_optimal_attention_no_clustering",distance_to_consensus_matrix_optimal_attention)
    # print("matrix_distance_to_optimal_attention_with_clustering", matrix_distance_to_optimal_attention)
    # print(f'distances_to_ge_consistency_matrix_mean', distances_to_ge_consistency_matrix_mean)
    savefile = 'res/'
    file_name = savefile + 'distance_to_consensus_matrix_optimal_attention_no_clustering.csv'
    df = pd.DataFrame(distance_to_consensus_matrix_optimal_attention)
    df.to_csv(file_name, index=False)
    file_name = savefile + 'matrix_distance_to_optimal_attention_with_clustering.csv'
    df = pd.DataFrame(matrix_distance_to_optimal_attention)
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    main()
    print("the end")
