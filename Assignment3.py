from Assignment2 import filtered_tokens
import numpy as np
import math as Math

print(filtered_tokens)

def editDistance(str1, str2):
    dp = [[0] * (len(str1)+1) for _ in range(len(str2) +1) ]
    for i in range(len(dp[0])):
        dp[0][i] = i
    for j in range(len(dp)):
        dp[j][0] = j

    for i in range(1,len(dp)):
        for j in range(1,len(dp[0])):
            if(str1[j-1] == str2[i-1]):
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(1+dp[i-1][j],1+dp[i][j-1],2+dp[i-1][j-1])
    return dp[len(str2)][len(str1)]


inputText = input("Enter any Word")

dict = dict()
for word in filtered_tokens:
    dict[word] = editDistance(word,inputText)
print(dict)