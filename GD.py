import sys
import numpy as np


class GD:
    def __init__(self, xk, lr, term_cond, A, b):
        """
        initialize the gradient descent
        xk: starting point
        lr: learning rate
        term_cond: termination condition
        A: design matrix
        b: target vector
        """
        self.cur_poi = xk  # gradient descent current point
        self.lr = lr  # learning rate (step size)
        self.term_cond = term_cond  # termination condition
        self.design_mat = A  # design matrix
        self.tar_vec = b  # target vector

    def runGD(self, gamma):
        """
        run gradient descent
        gamma: parameter of ridge penalty term
        """
        idx = 0
        points = [(idx, self.cur_poi.flatten().round(4))]
        while self.calc_l2_norm(
            self.calc_grad(self.design_mat, self.tar_vec, self.cur_poi,
                           gamma)) >= 0.0001:
            idx += 1
            self.cur_poi = self.cur_poi - self.lr * self.calc_grad(
                self.design_mat, self.tar_vec, self.cur_poi, gamma)
            points.append((idx, self.cur_poi.flatten().round(4)))
        print(f'The first 5 rows with the starting point:')
        for idx, point in points[:6]:
                print(f'\tk = {idx}, xk = {point}')
        print()
        print(f'The last 5 rows:')
        for idx, point in points[-5:]:
            print(f'\tk = {idx}, xk = {point}')

    def calc_grad(self, A, b, xk, gamma):
        """
        calculate the gradient of the lost function
        A: design matrix
        b: target vector
        xk: current point
        gamma: parameter of ridge penalty term
        """
        gradient = self.mat_mul(((self.mat_mul(self.transpose(A), A)) +
                                 gamma * np.eye(A.shape[1])), xk) - \
                   self.mat_mul(self.transpose(A), b)
        return gradient

    def mat_mul(self, mat_lef, mat_rig):
        """ matrix multiplication """
        # check if the matrices can be multiplied
        if mat_lef.shape[1] != mat_rig.shape[0]:
            print('matrix multiplication error')
            sys.exit(1)  # exit the program and return 1
        # calculate the result
        mat_res = np.zeros((mat_lef.shape[0], mat_rig.shape[1]))
        for i in range(mat_lef.shape[0]):
            for j in range(mat_rig.shape[1]):
                for k in range(mat_lef.shape[1]):
                    mat_res[i][j] += mat_lef[i][k] * mat_rig[k][j]
        return mat_res

    def transpose(self, mat):
        """ transpose a matrix """
        mat_tran = np.zeros((mat.shape[1], mat.shape[0]))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat_tran[j][i] = mat[i][j]
        return mat_tran

    def calc_l2_norm(self, vec):
        """ calculate the l2 norm of a vector """
        l2_norm = 0
        for i in range(len(vec)):
            l2_norm += vec[i] ** 2
        return l2_norm ** 0.5


if __name__ == "__main__":
    # initialize the parameters
    A = np.array(
        [[1, 2, 1, -1],
         [-1, 1, 0, 2],
         [0, -1, -2, 1]])  # design matrix
    # note we reshape the vectors here to improve the numerical stability
    # (i.e. we treat vectors as matrices so that Python will not throw us errs)
    b = np.array([3, 2, -2]).reshape(-1, 1)  # target vector
    xk = np.array([1, 1, 1, 1]).reshape(-1, 1)  # current point
    lr = 0.1  # learning rate (aka step size)
    term_cond = 0.001  # termination condition
    gamma = 0.2  # parameter of ridge penalty term

    # run gradient descent
    gd = GD(xk, lr, term_cond, A, b)
    gd.runGD(gamma)


