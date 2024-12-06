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
        n_train = int(len(datasets_expert_matrix)*0.8)
        n_valid = len(datasets_expert_matrix)-n_train
        # expert_matrix_train,expert_matrix_valid = random_split(datasets_expert_matrix,[n_train,n_valid])
        expert_matrix_train = torch.zeros(n_train, Order, Order)
        expert_matrix_valid = torch.zeros(n_valid, Order, Order)
        for i in range(int(3*n_valid), len(datasets_expert_matrix)):
            expert_matrix_train[i-int(2*n_valid)] = datasets_expert_matrix[i]
        for j in range(int(2*n_valid)):
            expert_matrix_train[j] = datasets_expert_matrix[j]
        for k in range(int(2*n_valid), int(3*n_valid)):
            expert_matrix_valid[k - int(2*n_valid)] = datasets_expert_matrix[k]
        print(len(datasets_expert_matrix))
        print(len(expert_matrix_train))
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
        # print(f'ranking_values_of_original_matrix', geometric_mean_of_original_matrix)
        # print("geometric_mean_of_original_matrix",geometric_mean_of_original_matrix)
        print("generate_matrix_order", generate_matrix_order)
        # print(f'ranking_values_of_generate_matrix', geometric_mean_of_generate_matrix)
        # print("geometric_mean_of_generate_matrix",geometric_mean_of_generate_matrix)


    def candidates_sort_with_geometric_mean(self, matrix):
        matrix_order = len(matrix)
        geometric_mean = torch.zeros(matrix_order)
        for i in range(matrix_order):
            geometric_mean[i] = torch.pow(torch.prod(matrix[i]), 1/matrix_order)
        candidates_sort, sort_index = torch.sort(geometric_mean)
        print(f'Rank of candidates:', sort_index)



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
        savefile_ci = 'res_CI/cross_validation/'
        file_name_1 = savefile_ci + 'original matrix CI_3.csv'
        with open(file_name_1, "a") as df_1:
            df_1.write(str(temp_x_ci))
            df_1.write("\n")
        file_name_2 = savefile_ci + 'Discrete HCVAE generating matrix CI_3.csv'
        with open(file_name_2, "a") as df_2:
            df_2.write(str(re_expert_matrix_ci))
            df_2.write("\n")
    # print(f'epoch:{i}|ValidLoss: ', valid_loss / (matrix_number))
    df_1.close()
    df_2.close()
    return re_expert_matrix_iter



class MatrixSetComparisonAnalysis(object):

    def order_difference_calculation(self, original_order_serial, order_serial):

        order_number = len(order_serial)

        order_difference = np.zeros(order_number)

        for i in range(order_number):
            if i == 0:
                if original_order_serial[i] != order_serial[i]:
                    order_difference[i] = 1 + 0.5
                else:
                    order_difference[i] = 0
            else:
                if original_order_serial[i] != order_serial[i]:
                    order_difference[i] = 1
                else:
                    order_difference[i] = 0
        total_order_difference = np.sum(order_difference)

        print(f'===============================================')
        print(f'original_order_serial', original_order_serial)
        print(f'order_serial', order_serial)
        print(f'order_difference', order_difference)
        print(f'total_order_difference', total_order_difference)
        return total_order_difference



    def matrix_metric_calculation(self, matrix):
        matrix = matrix.numpy()
        matrix_order = len(matrix)
        ranking_values = np.zeros(matrix_order)
        #ranking = np.zeros(matrix_order)
        ranking_values_sum = 0
        # print(f'The first element of matrix:', matrix[0, 0])
        if abs(matrix[0, 0] - 0.5) < 1e-5:
            for i in range(matrix_order):
                ranking_values[i] = np.sum(matrix[i])/matrix_order
                ranking_values_sum += ranking_values[i]
        else:
            if abs(matrix[0, 0] - 1.0) < 1e-5:
                for i in range(matrix_order):
                    ranking_values[i] = np.prod(matrix[i])**(1/matrix_order)
                    ranking_values_sum += ranking_values[i]
            else:
                print(f'The matrix is not satisfied!')
                return
        for j in range(matrix_order):
            ranking_values[j] = ranking_values[j] / ranking_values_sum
        ranking = np.argsort(ranking_values)[::-1]
        return ranking_values, ranking


    def calculation_comparison_differences(self, original_ranking_value, ranking_value, original_ranking, ranking,
                                           original_matrix, matrix):
        matrix_order = len(original_ranking_value)
        tmp_value_difference = 0
        tmp_matrix_difference = 0
        for i in range(matrix_order):
            tmp_value_difference += np.power(original_ranking_value[i] - ranking_value[i], 2)
        ranking_value_difference = np.sqrt(tmp_value_difference)
        for i in range(matrix_order):
            for j in range(matrix_order):
                tmp_matrix_difference += np.power(original_matrix[i, j] - matrix[i, j], 2)
        matrix_difference = np.sqrt(tmp_matrix_difference)
        ranking_difference = self.order_difference_calculation(original_ranking, ranking)
        #print(f"ranking_differences", i, ranking_value_differences[i], ranking_differences[i])
        return ranking_value_difference, ranking_difference, matrix_difference



    def comparison_results_obtained(self, matrix_set, consensus_matrix, name):
        #print(f' consensus_matrix', consensus_matrix)
        consensus_ranking_values, consensus_rankings = self.matrix_metric_calculation(consensus_matrix)
        print(f'consensus_ranking_values', consensus_ranking_values)
        print(f'consensus_rankings', consensus_rankings)
        matrix_number = len(matrix_set)
        file_name = f"./comparison_result/{name}_comparison_differences.csv"
        # print(f'file_name', file_name)
        with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write("No.,  value_difference,  order difference, matrix difference, \n")
            for i in range(matrix_number):
                matrix = matrix_set[i]
                ranking_values, rankings = self.matrix_metric_calculation(matrix)
                ranking_value_difference, ranking_difference, matrix_difference = \
                    self.calculation_comparison_differences(ranking_values, consensus_ranking_values, rankings,
                                                            consensus_rankings, matrix, consensus_matrix)
                # print(f"ranking_differences", i, ranking_value_differences[i], ranking_differences[i])
                str_No = str(i)
                str_value_difference = str(ranking_value_difference)
                str_ranking_difference = str(ranking_difference)
                str_matrix_difference = str(matrix_difference)
                write_row = str_No + ',' + str_value_difference + ',' + str_ranking_difference + ',' \
                            + str_matrix_difference
                csvfile.write(write_row)
                csvfile.write('\n')



    def consistent_comparison_results_obtained(self, matrix_set, original_matrix_set, name):
        #print(f' consensus_matrix', consensus_matrix)
        matrix_number = len(matrix_set)
        file_name = f"./comparison_result/{name}_CI_set_comparison_differences.csv"
        # print(f'file_name', file_name)
        with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write("No.,  value_difference,  order difference, matrix difference, \n")
            for i in range(matrix_number):
                matrix = matrix_set[i]
                org_matrix = original_matrix_set[i].clone().detach()
                ranking_values, rankings = self.matrix_metric_calculation(matrix)
                org_ranking_values, org_rankings = self.matrix_metric_calculation(org_matrix)
                ranking_value_difference, ranking_difference, matrix_difference = \
                    self.calculation_comparison_differences(ranking_values, org_ranking_values, rankings,
                                                            org_rankings, matrix, org_matrix)
                # print(f"ranking_differences", i, ranking_value_differences[i], ranking_differences[i])
                str_No = str(i)
                str_value_difference = str(ranking_value_difference)
                str_ranking_difference = str(ranking_difference)
                str_matrix_difference = str(matrix_difference)
                write_row = str_No + ',' + str_value_difference + ',' + str_ranking_difference + ',' \
                            + str_matrix_difference
                csvfile.write(write_row)
                csvfile.write('\n')



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

    cross_valid = MatrixSetComparisonAnalysis()
    cross_valid.consistent_comparison_results_obtained(re_consistent_matrix_iter, dataset_valid, "cross_3")




if __name__ == '__main__':
    main()
    print("the end")
