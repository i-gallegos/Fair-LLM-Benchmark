

def load_csv(path):
	heads = []
	table = []
	with open(path, 'r') as f:
		ln_cnt = 0
		for l in f:
			ln_cnt += 1
			if ln_cnt == 1:
				heads.extend(l.rstrip().split(','))
				continue

			table.append(l.rstrip().split(','))
	return heads, table


head_p, table_p = load_csv('models/adjectives_countries_ordered/elmo.pred.txt')
head_f, table_f = load_csv('models/adjectives_countries_full/elmo.pred.txt')


cache = set()
p_idx = head_p.index('premise')
h_idx = head_p.index('hypothesis')
print(p_idx, h_idx)
for row in table_p:
	line = (row[p_idx] + row[h_idx]).lower()
	cache.add(line)

p_idx = head_f.index('premise')
h_idx = head_f.index('hypothesis')
print(p_idx, h_idx)
match_cnt = 0
total_cnt = 0
for row in table_f:
	line = (row[p_idx] + row[h_idx]).lower()
	if line in cache:
		match_cnt += 1
	total_cnt += 1

print(match_cnt, total_cnt)