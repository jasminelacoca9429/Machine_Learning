class Matrix:

    def __init__(self, elements, dim):
        '''
        init the elements attribute to matrix_elements
        and set dim with the appropriate dimensions
        :param elements: a two-dimensional list of the elements of the matrix
        :param dim: a tuple with the dimensions of the matrix
        '''

        if len(elements) == dim[0] and len(elements[0]) == dim[1]:
            self.elements = elements
            self.dim = dim
        else:
            raise ValueError('The dimensions do not match the matrix.')

    def is_matrix(self):
        '''
        return whether the object A is of type Matrix
        :param A: matrix
        :return:
        '''
        m = len(self.elements[0])
        for i in range(self.dim[0]):
            if len(self.elements[i]) != m:
                print(self, 'is not a matrix, operations cannot be applied.')
                return False
        return True

    def det(self):
        '''
        compute the determinant of the matrix
        :return: The value of determinant
        '''
        if not self.is_matrix():
            return None
        elif self.dim[0] != self.dim[1]:
            return f'This matrix is not eligible for a corresponding determinant.'

        n = self.dim[0]
        mat = self.elements
        temp = [0] * n  # temporary array for storing row
        total = 1
        result = 1  # initialize result
        # loop for traversing the diagonal elements

        for i in range(0, n):
            index = i  # initialize the index
            # finding the index which has non zero value
            while index < n and mat[index][i] == 0:
                index += 1

            if index == n:  # if there is non zero element
                # the determinant of matrix as zero
                continue

            if index != i:
                # loop for swaping the diagonal element row and index row
                for j in range(0, n):
                    mat[index][j], mat[i][j] = mat[i][j], mat[index][j]

                # determinant sign changes when we shift rows
                # go through determinant properties
                result = result * int(pow(-1, index - i))

            # storing the values of diagonal row elements
            for j in range(0, n):
                temp[j] = mat[i][j]

            # traversing every row below the diagonal element
            for j in range(i + 1, n):
                num1 = temp[i]  # value of diagonal element
                num2 = mat[j][i]  # value of next row element

                # traversing every column of row
                # and multiplying to every row
                for k in range(0, n):
                    # multiplying to make the diagonal
                    # element and next row element equal

                    mat[j][k] = (num1 * mat[j][k]) - (num2 * temp[k])

                total = total * num1  # Det(kA)=kDet(A);

        # mulitplying the diagonal elements to get determinant
        for i in range(0, n):
            result = result * mat[i][i]

        return f'The determinant value is {int(result / total)}' # Det(kA)/k=Det(A);

    def transpose(self):
        '''
        :return: return a Matrix whose elements are transposed
        '''
        if not self.is_matrix():
            return [[]]
        trans_matrix = []
        for i in range(self.dim[1]):
            trans_line = []
            for j in range(self.dim[0]):
                trans_line.append(self.elements[j][i])
            trans_matrix.append(trans_line)
        return trans_matrix

    def sort(self, type):
        '''
        return a 1-D array with the matrix elements sorted
        in an increasing or decreasing order (according to type)
        :param type: increasing or decreasing order
        :return:
        '''
        if not self.is_matrix():
            return []
        oneD_array = []
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                oneD_array.append(self.elements[i][j])
        if type == 'in':
            oneD_array.sort()
        elif type == 'de':
            oneD_array.sort(reverse=True)
        else:
            print('Invalid type.')
            return []
        return f'sorted in {type}creasing order: {oneD_array}'

    def __add__(self, other):
        '''
        overload for the '+' operator
        :param other: matrix
        :return: sum result of matrices
        '''

        if not self.is_matrix():
            print('This is not a matrix.')
            return [[]], (0, 0)
        sum_result = []
        for i in range(self.dim[0]):
            line_result = []
            for j in range(self.dim[1]):
                unit = self.elements[i][j] + other.elements[i][j]
                line_result.append(unit)
            sum_result.append(line_result)

        return sum_result, self.dim

    def __sub__(self, other):
        '''
        overload for the '+' operator
        :param other: Matrix
        :return: subtraction result of matrices
        '''
        if not self.is_matrix():
            print('This is not a matrix.')
            return [[]], (0, 0)
        sum_result = []
        for i in range(self.dim[0]):
            line_result = []
            for j in range(self.dim[1]):
                unit = self.elements[i][j] - other.elements[i][j]
                line_result.append(unit)
            sum_result.append(line_result)

        return sum_result, self.dim

    def __mul__(self, other):
        '''
        overload for the '*' operator
        Two matrices could be multiplied only if the column amount of first matrix
        equals to the row amount of second matrix.
        :param other: matrix
        :return: result matrix of multiplication
        '''
        if not self.is_matrix():
            print('This is not a matrix.')
            return [[]], (0, 0)
        elif self.dim[1] != other.dim[0]:
            print('These matrices cannot be multiplied.')
            return [[]]
        result = []

        for m in range(self.dim[0]):
            row = []
            for n in range(other.dim[1]):
                row.append(0)
            result.append(row)
        print(result)
        # result matrix: self.dim[0] * other.dim[1]
        for i in range(self.dim[0]):
            for j in range(other.dim[1]):
                for k in range(other.dim[0]):
                    result[i][j] += self.elements[i][k] * other.elements[k][j]
        return result

    def __eq__(self, other):
        '''
        overload for the '==' operator
        Two matrices are equal when corresponding elements within each matrix are equal
        :param other: matrix
        :return: equavalence
        '''
        # return ([self.elements[i][j] == other.elements[i][j] for j in range(len(self.elements[i])] for i in range(len(self.elements)))
        n = False
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):

                if self.elements[i][j] == other.elements[i][j]:
                    n = True
                else:
                    n = False
                    break
            if not n:
                break
        return n

    def __len__(self):
        '''
        return the number of elements in the matrix
        :return: length
        '''
        if self.is_matrix():

            return self.dim[0] * self.dim[1]

        else:
            print('This object is not a matrix')
            return 0

    def __str__(self):
        '''
        return a string with the matrix's elements
        :return: string
        '''
        return f'{self.elements}'

    def save(self, filename):
        '''
        save the object to a binary file
        :param filename: the filename
        :return: file been saved
        '''
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            print(f'Saved as {filename}')

    def load(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            print(pickle.load(f))
