# Test: Sum of matrix elements
def sum_row(row):
    if not row:
        return 0
    return row[0] + sum_row(row[1:])

def sum_matrix(matrix):
    if not matrix:
        return 0
    return sum_row(matrix[0]) + sum_matrix(matrix[1:])

def main():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return sum_matrix(matrix)

print(main())
