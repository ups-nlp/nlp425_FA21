import string
print(string.punctuation)

# Cutoff -- this is the minimum number of occurances we want
threshhold = 10

# Files to be used 
f = open('wp_2gram.txt', 'r')
out = open('wiki.txt', 'a')

def makeText(): 
	for line in f:
		# Stops us from erroring out
		if line is None:
			break
		# Excludes the EOS markers that the n-grams came with, they dont have any valid words so we don't care
		if "#EOS" in line:
			continue
		else:
			l = line.strip().split('\t')
			if len(l) != 3 or l[1] in string.punctuation or l[2] in string.punctuation:
				continue
			freq = l[0]
			# Makes sure the n-grams are not of a word and punctuation or two punctuation characters 
			if int(freq) < threshhold:
				continue
			else:	
				out.write(l[0] + '\t' + l[1].lower() + '\t' + l[2].lower() + '\n')
	else:
		print("Reached eend of bigrams.")
		f.close()

def unigramizer():
	uni = open('uni.txt', 'w')

	with open('wiki.txt', 'r') as file:
		l  = file.readline().strip().split('\t')
		print(l)
		prev  = l[1]
		freq  = l[0]

		for line in file:
			line = line.strip().split('\t')

			if line[1] == prev:
				freq = int(freq) + int(line[0])
			else:
				uni.write(str(prev) + '\t' + str(freq) + '\n')
				prev = line[1]
				freq = line[0]


if __name__ == "__main__" :
	# Code you want to run
	#makeText()
	unigramizer()
	# FINAL LINES, coveirng my bases to make sure files are all shut
	f.close()
	out.close()
	# NO FURTHER CODE BELOW THIS POINT ==================