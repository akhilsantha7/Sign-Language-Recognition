# 1121
# for i in range(n):
#     if i == 0:
#         val = 1
#     elif i == 1:
#         val = 11
#     else:
#         temp = val
#         cnt = 1
#         for i in range(len(temp)-1):
#             if val[i] == val[i+1]
#                 cnt = cnt + 1
#             else:
#                 cont = 1 

# val = '111221'
# cnt = 1
# st1 = ''
# temp = ''
# for i in range(len(val)-1):
    
#     if val[i] == val[i+1]:
#         cnt = cnt + 1
#         print(cnt)
#         continue
        
#     else:
#         cnt = 1
        
#     print(cnt)
#     st1 = str(cnt)
#     temp = temp + st1 + val[i]

# print(temp)




from nltk.translate.bleu_score import sentence_bleu
reference = [['heftiger', 'wintereinbruch', 'gestern', 'in', 'nordirland', 'schottland']]
candidate = ['heftiger', 'wintereinbruch', 'gestern', 'in', 'nordirland', 'schottland']
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))