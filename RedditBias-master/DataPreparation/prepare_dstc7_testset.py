"""
This script removes the token __UNDISCLOSED__ from DSTC7 testset
"""
read_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/test_convos.txt'
write_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/test_convos_processed.txt'

with open(read_file_path, encoding="utf-8") as f:
    test_lines = [line.replace('__UNDISCLOSED__', '') for line in f.read().splitlines()]

for t_l in test_lines[:10]:
    print(t_l)

with open(write_file_path, "w", encoding="utf-8") as f:
    for t_l in test_lines:
        f.write(t_l + '\n')