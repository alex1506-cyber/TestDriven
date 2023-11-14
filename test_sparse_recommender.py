import pytest
from sparse_recommender import SparseMatrix


def test_set():
    matrix = SparseMatrix()

    # Test setting and getting a valid value
    matrix.set(0, 1, 5)
    assert matrix.get(0, 1) == 5

    # Test setting and getting a large value
    matrix.set(1000, 1000, 999999)
    assert matrix.get(1000, 1000) == 999999

    # Test overwriting a previously set value
    matrix.set(0, 1, 10)
    assert matrix.get(0, 1) == 10

    # Test getting a default value when the cell has not been set
    assert matrix.get(1, 0) == 0
    assert matrix.get(0,0) == 0

    # Test raising ValueError for non-integer row
    with pytest.raises(ValueError):
        matrix.set("A", 0, 5)

    # Test raising ValueError for negative row
    with pytest.raises(ValueError):
        matrix.set(-1, 0, 5)

    # Test raising ValueError for negative column
    with pytest.raises(ValueError):
        matrix.set(0, -2, 5)

    with pytest.raises(ValueError):
        matrix.set(2, 2, "Invalid")

    # Test raising ValueError for non-numeric value
    with pytest.raises(ValueError):
        matrix.set(2, 2, None)


def test_get():
    sparse_matrix = SparseMatrix({(0, 2): 5, (2, 0): 3}, 3, 3)
    row = 0
    col = 2
    value = 5
    result = sparse_matrix.get(row, col)
    assert value == result
    assert sparse_matrix.get(1, 0) == 0

    with pytest.raises(ValueError):
        sparse_matrix.get(-1,0)
    with pytest.raises(ValueError):
        sparse_matrix.get(0,-3)
    #with pytest.raises(ValueError):
        #sparse_matrix.get(5, 0)

def test_recommender():
    sparse_matrix = SparseMatrix({(0, 2): 5, (2, 0): 3}, 3, 3)
    vector = [1, 2, 3]
    value = [15, 0, 3]
    # Test recommend function with incorrect vector size
    with pytest.raises(ValueError) as e:
        result = sparse_matrix.recommender([1, 2])
    assert str(e.value) == "Vector size must match the number of columns in the matrix."

    # Test recommend function with a non-numeric vector element
    with pytest.raises(ValueError) as e:
        result = sparse_matrix.recommender([1, 'a', 3])
    assert str(e.value) == "Vector elements must be numeric."

    # Test recommend function with a non-integer vector element
    with pytest.raises(ValueError) as e:
        result = sparse_matrix.recommender([1, 2.5, 3])
    assert str(e.value) == "Vector elements must be integers."

    result = sparse_matrix.recommender(vector)
    assert value == result

    sparse_matrix = SparseMatrix({(0, 0): 0, (1, 2): 0}, 2, 3)
    vector = [0, 0, 0]
    result = sparse_matrix.recommender(vector)
    assert result == [0, 0]




def test_add_movie():
    sparse_matrix = SparseMatrix({(0, 2): 5, (2, 0): 3}, 3, 3)
    matrix = {(2, 0): 1, (1, 1): 7, (2, 2): 10}
    value = {(0, 2): 5, (1, 1): 7, (2, 0): 4, (2, 2): 10}
    result = sparse_matrix.add_movie(matrix)
    assert value == result

    sparse_matrix = SparseMatrix({(0, 0): 1, (1, 2): 2}, 2, 3)
    movie = {(0, 1): 0, (1, 1): 0}
    result = sparse_matrix.add_movie(movie)
    assert result == {(0, 0): 1, (1, 2): 2,(0, 1): 0, (1, 1): 0}

    negative_movie = {(-1, 0): -2, (1, -1): -3}
    expected_result = {(0, 0): 1, (1, 2): 2}
    result = sparse_matrix.add_movie(negative_movie)
    assert result == expected_result
def test_add_movie_ratings_greater_than_10():
    sparse_matrix = SparseMatrix({(0, 2): 5, (2, 0): 3}, 3, 3)
    movie = {(1, 1): 12, (2, 2): 15}
    try:
        sparse_matrix.add_movie(movie)
        pytest.fail("Expected ValueError but no exception was raised.")
    except ValueError as e:
        assert str(e) == "Movie ratings cannot be greater than 10."

def test_add_movie_invalid_value():
    sparse_matrix = SparseMatrix({(0, 2): 5, (2, 0): 3}, 3, 3)
    movie = {(0, 1): "invalid", (1, 1): None}
    with pytest.raises(ValueError):
        sparse_matrix.add_movie(movie)


def test_add_movie_edge_iterations():
    # Test adding a movie with maximum values and iterations
    max_rows = 500
    max_cols = 500
    max_iterations = 500
    sparse_matrix = SparseMatrix({}, max_rows, max_cols)
    matrix = {(max_cols - 1, max_rows - 1): 10}

    result = sparse_matrix.add_movie(matrix, max_iterations=max_iterations)

    # Check that the result is correct
    assert result == matrix


def test_to_dense():
    sparse_matrix = SparseMatrix({(0, 2): 5, (1, 1): 4, (2, 0): 3}, 3, 3)
    value = [[0, 0, 5], [0, 4, 0], [3, 0, 0]]
    with pytest.raises(ValueError) as e:
        result = sparse_matrix.to_dense(max_iterations='invalid')
    assert str(e.value) == "max_iterations must be an integer."

    # Test to_dense function with a negative max_iterations
    with pytest.raises(ValueError) as e:
        result = sparse_matrix.to_dense(max_iterations=-1)
    assert str(e.value) == "max_iterations must be non-negative."

    result = sparse_matrix.to_dense()
    assert value == result

    sparse_matrix= SparseMatrix({(1,2):1},3,3)
    result = sparse_matrix.to_dense()
    assert result == [[0, 0, 0], [0, 0, 1], [0, 0, 0]]


def test_to_dense_edge_iterations():
    # Create a large sparse matrix for testing edge cases
    num_rows = 1000
    num_cols = 1000
    matrix_data = {(i, j): 1 for i in range(num_rows) for j in range(num_cols)}
    sparse_matrix = SparseMatrix(matrix_data, num_rows, num_cols)

    # Test converting the large sparse matrix to a dense matrix with a limited number of iterations
    max_iterations = 10  # Limit the iterations to 10
    dense_matrix = sparse_matrix.to_dense(max_iterations=max_iterations)

    # Ensure that the resulting dense matrix is of the correct size
    assert len(dense_matrix) == num_rows
    assert len(dense_matrix) == num_cols

    # Check some specific values to verify the correctness of the conversion
    assert dense_matrix[0][0] == 1
    assert dense_matrix[0][1] == 1
   # Testing for the max_iterations to 1000
    max_iterations = 1000
    dense_matrix=sparse_matrix.to_dense(max_iterations=max_iterations)
    assert len(dense_matrix) == num_rows
    assert len(dense_matrix) == num_cols
    assert dense_matrix[999][999] == 0