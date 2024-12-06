import torch.utils.data
from torch import optim
from class_D_HCVAE import CVAE
import torch
from Discrete_HCVAE_self_generation import MatrixInformationCalculation
from Discrete_HCVAE_self_generation import MatrixSetLoader


Length = 101
Order = 8
Cardinal = 8
lr = 1e-5  # learning rate
train_time = 2000


"""Define the training process of the discrete HCVAE"""


def train(model, train_rawmatrix, matrix_order):
    learning_rate = lr
    model_train = model
    model_train.train()
    model_train.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model_train.zero_grad()
    dataset_train = train_rawmatrix
    # dataset_consistentmatrix = train_consistentmatrix
    train_loss = 0
    matrix_number = len(dataset_train)
    for i in range(0, matrix_number):
        temp_x = dataset_train[i]
        temp_y = MatrixInformationCalculation().calculat_matrix_consistent(temp_x, matrix_order)
        normal_x = MatrixInformationCalculation().matrix_normalize(temp_x, matrix_order)
        normal_y = MatrixInformationCalculation().matrix_normalize(temp_y, matrix_order)
        x = MatrixInformationCalculation().matrix_to_uptringle_vector(normal_x, matrix_order)
        y = MatrixInformationCalculation().matrix_to_uptringle_vector(normal_y, matrix_order)
        re_matrixs, mu_e, mu_d, sigma = model_train(x, y)
        re_matrixs_len = len(re_matrixs)
        rv_matrixs = MatrixInformationCalculation().reverse_normalize_vector(re_matrixs, 8)
        re_expert_matrix = MatrixInformationCalculation().uptriangle_vector_to_rematrix(rv_matrixs, re_matrixs_len,
                                                                                        matrix_order)
        loss = model.loss_function(re_matrixs, x, mu_e, mu_d, sigma)
        loss.backward()
        model_train.optimizer.step()
        # 记录总损失
        model_train.zero_grad()
        train_loss += loss.item()
        # print(f'epoch:{i}|TrainLoss: ', train_loss / len(dataset_train))
    print(f'train_loss:', train_loss)
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
        re_matrixs, mu_e, mu_d, sigma = model(x, y)
        re_matrixs_len = len(re_matrixs)
        rv_matrixs = MatrixInformationCalculation().reverse_normalize_vector(re_matrixs, matrix_order)
        re_expert_matrix = MatrixInformationCalculation().uptriangle_vector_to_rematrix(rv_matrixs, re_matrixs_len,
                                                                                        matrix_order)
        re_expert_consistent_matrix = MatrixInformationCalculation().calculat_matrix_consistent(re_expert_matrix,
                                                                                                matrix_order)
        re_expert_matrix_iter[i] = re_expert_matrix
        print(f'Generated {i}th matrix:\n', re_expert_matrix)
        re_expert_matrix_ci = MatrixInformationCalculation().calculat_matrix_ci(re_expert_matrix,
                                                                                re_expert_consistent_matrix)
        print("re_expert_matrix_ci:", re_expert_matrix_ci, "\n")
        loss = model.loss_function(re_matrixs, x, mu_e, mu_d, sigma)
        valid_loss += loss.item()
    # print(f'epoch:{i}|ValidLoss: ', valid_loss / (matrix_number))
    return re_expert_matrix_iter


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
    train_consistentmatrix = MatrixSetLoader().calculation_consistency_matrix_iter(dataset_train)
    valid_consistentmatrix = MatrixSetLoader().calculation_consistency_matrix_iter(dataset_valid)
    """Training the model and utilizing the model"""
    epochs = train_time
    for epoch in range(epochs):
        train(model, dataset_train, matrix_order)
    torch.save(model.state_dict(), "./Discrete HCVAE_model_save/Discrete HCVAE_model_test_10.pth")
    re_consistent_matrix_iter = valid(model, dataset_valid, matrix_order)


if __name__ == '__main__':
    main()
    print("the end")
