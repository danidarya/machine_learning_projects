from matrix import Matrix, input_matrix


def main():
    end = False
    while end is False:
        first_operand = input_matrix()
        print('Введенная матрица')
        first_operand.print()
        operation = input('Введите операцию (+, -, *, /, transpose, det): ')
        if operation == 'transpose':
            result = first_operand.transpose()
        elif operation == 'det':
            result = first_operand.det()
        else:
            second_operand = input_matrix()
            print('Введенная матрица')
            second_operand.print()
            if operation == '+':
                result = first_operand + second_operand
            if operation == '-':
                result = first_operand - second_operand
            if operation == '*':
                result = first_operand * (second_operand)
            if operation == '/':
                result = first_operand / second_operand
        print("Результат: ")
        if type(result) == Matrix:
            result.print()
        else:
            print(result)
        end = input('Продолжить?(y/n): ') == 'n'


if __name__ == '__main__':
    main()
