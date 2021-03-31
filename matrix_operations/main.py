import copy

EPS = 1e-9


class Matrix:
    def __init__(self, n_rows, n_cols, elements):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.elements = elements

    def add(self, other):
        if other.n_rows != self.n_rows or other.n_cols != self.n_cols:
            raise IOError("Операция невозможна, размеры матриц не совпадают")
        res_elements = [[0] * self.n_cols for _ in range(self.n_rows)]
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                res_elements[i][j] = self.elements[i][j] + other.elements[i][j]
        return Matrix(self.n_rows, self.n_cols, res_elements)

    def subtract(self, other):
        if other.n_rows != self.n_rows or other.n_cols != self.n_cols:
            raise IOError("Операция невозможна, размеры матриц не совпадают")
        res_elements = [[0] * self.n_cols for _ in range(self.n_rows)]
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                res_elements[i][j] = self.elements[i][j] - other.elements[i][j]
        return Matrix(self.n_rows, self.n_cols, res_elements)

    def multiply(self, other):
        if self.n_cols != other.n_rows:
            raise IOError("Операция невозможна, размеры матриц не подходят для умножения")
        res_elements = [[0] * other.n_cols for _ in range(self.n_rows)]
        for i in range(self.n_rows):
            for j in range(other.n_cols):
                for k in range(self.n_cols):
                    res_elements[i][j] += self.elements[i][k] * other.elements[k][j]
        return Matrix(self.n_rows, other.n_cols, res_elements)

    def det(self):
        # вычисление определителя матрицы методом Гаусса
        if self.n_cols != self.n_rows:
            raise IOError("Операция невозможна, матрица не является квадратной")
        elems = copy.deepcopy(self.elements)
        m = Matrix(self.n_rows, self.n_cols, elems)
        res = 1
        n = self.n_cols
        for i in range(n):
            k = i
            for j in range(i + 1, n):
                if abs(m.elements[j][i]) > abs(m.elements[k][i]):
                    k = j
            if abs(m.elements[k][i]) < EPS:
                return 0
            m.elements[i], m.elements[k] = m.elements[k], m.elements[i]
            if i != k:
                res = - res
            res *= m.elements[i][i]
            for j in range(i + 1, n):
                m.elements[i][j] /= m.elements[i][i]
            for j in range(n):
                if j != i and abs(m.elements[j][i]) > EPS:
                    for k in range(i + 1, n):
                        m.elements[j][k] -= m.elements[i][k] * m.elements[j][i]
        return res

    def inv(self):
        if abs(self.det()) < EPS:
            raise IOError("Операция невозможна, матрица вырождена")
        n = self.n_cols
        elements_with_eye_matrix = [[0] * (2 * n) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                elements_with_eye_matrix[i][j] = self.elements[i][j]
            for j in range(n, 2 * n):
                elements_with_eye_matrix[i][j] = 1 * (j - n == i)
        m = Matrix(self.n_rows, 2 * self.n_cols, elements_with_eye_matrix)
        for i in range(n):
            k = i
            for j in range(i + 1, n):
                if abs(m.elements[j][i]) > abs(m.elements[k][i]):
                    k = j
            m.elements[i], m.elements[k] = m.elements[k], m.elements[i]
            for j in range(i + 1, 2 * n):
                m.elements[i][j] /= m.elements[i][i]
            for j in range(n):
                if j != i and abs(m.elements[j][i]) > EPS:
                    for k in range(i + 1, 2 * n):
                        m.elements[j][k] -= m.elements[i][k] * m.elements[j][i]
        elements_inv_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                elements_inv_matrix[i][j] = m.elements[i][j + n]
        return Matrix(n, n, elements_inv_matrix)

    def division(self, other):
        return self.multiply(other.inv())

    def transpose(self):
        new_elements = [[0] * self.n_rows for _ in range(self.n_cols)]
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                new_elements[j][i] = self.elements[i][j]
        return Matrix(self.n_cols, self.n_rows, new_elements)

    def print(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                print(self.elements[i][j], end=" ")
            print()


def input_matrix():
    n_rows = int(input('Введите количество строк: '))
    n_cols = int(input('Введите количество столбцов: '))
    elements_input = input('Введите значения элементов матрицы через пробел: ').split()
    elements = []
    for i in range(n_rows):
        lst = []
        for j in range(n_cols):
            lst.append(float(elements_input[i * n_cols + j]))
        elements.append(lst)
    return Matrix(n_rows, n_cols, elements)


def main():
    end = False
    while end is False:
        matrix_1 = input_matrix()
        print('Введенная матрица')
        matrix_1.print()
        operation = input('Введите операцию (+, -, *, /, transpose, det): ')
        if operation == 'transpose':
            result = matrix_1.transpose()
        elif operation == 'det':
            result = matrix_1.det()
        else:
            matrix_2 = input_matrix()
            print('Введенная матрица')
            matrix_2.print()
            if operation == '+':
                result = matrix_1.add(matrix_2)
            if operation == '-':
                result = matrix_1.subtract(matrix_2)
            if operation == '*':
                result = matrix_1.multiply(matrix_2)
            if operation == '/':
                result = matrix_1.division(matrix_2)
        print("Результат: ")
        if type(result) == Matrix:
            result.print()
        else:
            print(result)
        end = input('Продолжить?(y/n): ') == 'n'


if __name__ == '__main__':
    main()
