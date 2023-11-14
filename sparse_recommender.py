class SparseMatrix:
    def __init__(self, dic = {}, num_rows = 0, num_cols = 0):
        self.dictionary = dic
        self.num_rows = num_rows
        self.num_cols = num_cols
    
    def set(self, row, col, value):
        """ sets the value at (row, col) to value """
        if not isinstance(row, int) or not isinstance(col, int) or row < 0 or col < 0:
            raise ValueError("Row and column must be non-negative integers")
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be numeric")
        self.dictionary[(row, col)] = value

    def get(self, row, col):
        if row < 0 or col < 0 :
            raise ValueError("Row and column must be non-negative integers")
        return self.dictionary.get((row, col), 0)

    def recommender(self, vector):
        """ multiplies the sparse matrix with given vector to produce recommendations and return the result """
        if len(vector) != self.num_cols:
            raise ValueError("Vector size must match the number of columns in the matrix.")

        if not all(isinstance(element, (int, float)) for element in vector):
            raise ValueError("Vector elements must be numeric.")

        if not all(isinstance(element, int) for element in vector):
            raise ValueError("Vector elements must be integers.")

        result = [0] * self.num_rows
        for (row, col), value in self.dictionary.items():
            result[row] += value * vector[col]

        return result

    def add_movie(self, matrix,max_iterations=None):
        """ adds another sparse matrix, simulating the addition of a new movie to the service, and return the result """
        result = {}
        for loc, value in self.dictionary.items():
            result[loc] =  value


        for loc, value in matrix.items():
            row,col=loc
            if row>=0 or col>=0:
                if not isinstance(value, (int, float)):
                    raise ValueError("Movie ratings must be numeric")
                if value>10:
                    raise ValueError("Movie ratings cannot be greater than 10.")
                result[loc] = result.get(loc, 0) + value
        if max_iterations is not None and len(result) > max_iterations:
            raise ValueError(f"Exceeded maximum number of iterations: {max_iterations}")


        return {loc: value for loc, value in result.items() if all(i >= 0 for i in loc)}

    def to_dense(self, max_iterations=None):
        """Converts the sparse matrix to a dense matrix and returns it."""
        if max_iterations is not None:
            if not isinstance(max_iterations, int):
                raise ValueError("max_iterations must be an integer.")
            if max_iterations < 0:
                raise ValueError("max_iterations must be non-negative.")

        dense_matrix = [[0] * self.num_cols for _ in range(self.num_rows)]

        if max_iterations is None:
            for (row, col), value in self.dictionary.items():
                dense_matrix[row][col] = value
        else:
            iteration_count = 0
            for (row, col), value in self.dictionary.items():
                dense_matrix[row][col] = value
                iteration_count += 1
                if iteration_count >= max_iterations:
                    break

        return dense_matrix
