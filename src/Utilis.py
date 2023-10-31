import numpy as np

def longest_common_subpath(array1, array2):
    m, n = len(array1), len(array2)

    # Initialize a matrix to store the lengths of the longest common subpaths
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Variables to store the length of the longest common subpath and its ending position
    max_length = 0
    ending_pos = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if array1[i - 1] == array2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    ending_pos = i
            else:
                dp[i][j] = 0

    # Retrieve the longest subpath by slicing the array using the ending position and length
    longest_subpath = array1[ending_pos - max_length:ending_pos]

    return longest_subpath

# # Example arrays
# array1 = np.array([1, 2, 9, 4, 5, 6, 7])
# array2 = np.array([1, 4, 9, 5, 5, 9, 10])

# result = longest_common_subpath(array1, array2)
# print("Longest Common Subpath:", result)
