from conllu import parse, parse_tree
from sklearn.neural_network import MLPClassifier
from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf8')

num_pos = 17
num_dep = 37

model = KeyedVectors.load_word2vec_format('../glove.6B/glove.6B.50d.txt', binary=False)
# model = {}
model_size = 50

clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 5), random_state=1)
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 5), random_state=1)

STEP ={
	'la' : 0,
	'ra' : 1,
	'shift' : 2,
}

POS = {
	'ADJ' : 0,
	'ADV' : 1,
	'INTJ' : 2,
	'NOUN' : 3,
	'PROPN' : 4,
	'VERB' : 5,
	'ADP' : 6,
	'AUX' : 7,
	'CCONJ' : 8,
	'DET' : 9,
	'NUM' : 10,
	'PART' : 11,
	'PRON' : 12,
	'SCONJ' : 13,
	'PUNCT' : 14,
	'SYM' : 15,
	'X' : 16
}

DEP = {
	'nsubj' : 0,
	'obj' : 1,
	'iobj' : 2,
	'obl' : 3,
	'vocative' : 4,
	'expl' : 5,
	'dislocated' : 6,
	'nmod' : 7,
	'appos' : 8,
	'nummod' : 9,
	'conj' : 10,
	'cc' : 11,
	'fixed' : 12,
	'flat' : 13,
	'compound' : 14,
	'csubj' : 15,
	'ccomp' : 16,
	'xcomp' : 17,
	'advcl' : 18,
	'acl' : 19,
	'root' : 20,
	'advmod' : 21,
	'discourse' : 22,
	'amod' : 23,
	'aux' : 24,
	'cop' : 25,
	'mark' : 26,
	'det' : 27,
	'clf' : 28,
	'case' : 29,
	'dep' : 30,
	'list' : 31,
	'parataxis' : 32,
	'orphan' : 33,
	'goeswith' : 34,
	'reparandum' : 35,
	'punct' : 36,
}

def getAccuracy(vec1, vec2, correct, total):
	# vec2 = np.array(vec2)
	for i in range(vec1.shape[0]):
		total = total + 1
		if np.all(vec1[i] == vec2[i]):
			correct = correct + 1
	return (correct, total)

def neural_network(clf, ytrain, xtest, ytest, layer1, layer2):
	print("Performing Neural Network classification:")
	clf.fit(xtrain, ytrain)
	y_pred = clf.predict(xtest)
	accur = getAccuracy(ytest, y_pred)
	return accur

def read_data(filename):
	test = open(filename, 'r')
	text = test.read()
	test.close()
	# print text
	data = parse(text)		#parse(text)[i] represents the dependency relation of ith sentence
	tree = parse_tree(text)
	return(data, tree)

def get_next_step(stack_sigma, stack_buffer, dtree,  buffer):
	# print 1
	relation = None
	if (len(stack_sigma) == 0):
		y = stack_buffer.pop()
		stack_sigma.append(y)
		return (stack_sigma, stack_buffer, dtree, 'shift', relation)

	if (stack_sigma[-1] == 0):
		sigma = {
			'id' : 0,
			'head': None
		}
	else:
		sigma = buffer[stack_sigma[-1] - 1]

	beta = buffer[stack_buffer[-1] - 1]

	if (sigma['head'] == beta['id']):		
		dtree[sigma['id']] = {
			'head' : beta['id'],
			'relation' : sigma['deprel'],
			'side' : 'left'
		}
		x = stack_sigma.pop()
		relation = sigma['deprel']
		return (stack_sigma, stack_buffer, dtree, 'la', relation)
	
	elif (beta['head'] == sigma['id']):
		flag = False
		for word in buffer:
			if (word['head'] == beta['id']):
				if (word['id'] not in dtree):
					flag = True
		if not flag:
			dtree[beta['id']] = {
				'head' : sigma['id'],
				'relation' : beta['deprel'],
				'side' : 'right'
			}
			x = stack_sigma.pop()
			y = stack_buffer.pop()
			stack_buffer.append(x)
			relation = beta['deprel']
			return (stack_sigma, stack_buffer, dtree, 'ra', relation)
	
		else:
			y = stack_buffer.pop()
			stack_sigma.append(y)
			return (stack_sigma, stack_buffer, dtree, 'shift', relation)

	else:
		y = stack_buffer.pop()
		stack_sigma.append(y)
		return (stack_sigma, stack_buffer, dtree, 'shift', relation)

def give_features(iden, buffer):
	if (iden == -1 or iden == 1000 or iden == 0):
		return (np.zeros(num_pos), np.zeros(num_dep), np.zeros(model_size))
	word_dict = buffer[iden - 1]
	pos = np.zeros(num_pos)
	dep = np.zeros(num_dep)
	pos[POS[word_dict['upostag']]] = 1
	# print word_dict['deprel']
	dep[DEP[word_dict['deprel'].split(":")[0]]] = 1
	try:
		vector = model[word_dict['lemma']]
	except KeyError:
		vector = np.zeros(model_size)
	return (pos, dep, vector)

def construct_feature(stack_sigma, stack_buffer, dtree, buffer):
	if len(stack_sigma) != 0 and stack_sigma[-1] != 0:
		(pos_sigma_0, dep_sigma_0, word_sigma_0) = give_features(stack_sigma[-1], buffer)	
	else:
		sigma_0 = None
		pos_sigma_0 = np.zeros(num_pos)
		word_sigma_0 = np.zeros(model_size)
		dep_sigma_0 = np.zeros(num_dep)
	
	(pos_beta_0, dep_beta_0, word_beta_0) = give_features(stack_buffer[-1], buffer)

	if (len(stack_buffer) > 1):
		(pos_beta_1, dep_beta_1, word_beta_1) = give_features(stack_buffer[-2], buffer)
	else:
		beta_1 = None
		pos_beta_1 = np.zeros(num_pos)
		word_beta_1 = np.zeros(model_size)
		dep_beta_1 = np.zeros(num_dep)

	min_sigma = 1000
	min_beta = 1000
	max_sigma = -1
	max_beta = -1
	for node in dtree:
		if (len(stack_sigma) != 0 and dtree[node]['head'] == stack_sigma[-1]):
			if(min_sigma > node):
				min_sigma = node
			if(max_sigma < node):
				max_sigma = node
		if (dtree[node]['head'] == stack_buffer[-1]):
			if(min_beta > node):
				min_beta = node
			if(max_beta < node):
				max_beta = node

	(pos_sigma_left, dep_sigma_left, word_sigma_left) = give_features(min_sigma, buffer)
	(pos_sigma_right, dep_sigma_right, word_sigma_right) = give_features(max_sigma, buffer)
	(pos_beta_left, dep_beta_left, word_beta_left) = give_features(min_beta, buffer)
	(pos_beta_right, dep_beta_right, word_beta_right) = give_features(max_beta, buffer)
	return np.concatenate((word_sigma_0, pos_sigma_0, dep_sigma_0, word_beta_0, pos_beta_0, dep_beta_0, word_beta_1, 
				pos_beta_1, dep_beta_1, word_sigma_left, pos_sigma_left, dep_sigma_left, word_sigma_right, pos_sigma_right, 
				dep_sigma_right, word_beta_left, pos_beta_left, dep_beta_left, word_beta_right, pos_beta_right, dep_beta_right))

def construct_output(step, relation):
	out1 = np.zeros(3)
	out1[STEP[step]] = 1
	out2 = np.zeros(num_dep)
	if (relation):
		out2[DEP[relation]] = 1
	return (out1, out2)
	

def main():
	print "Training starts:"
	(data, tree) = read_data("data/train.conllu")
	for sentence in data:
		size = len(sentence)
		stack_sigma = [0]
		stack_buffer = []
		# corpus = []
		dtree = {}
		for i in range(size, 0, -1):
			stack_buffer.append(i)
			# corpus.append(sentence[size-i]['form'])
			# corpus.append(' ')
		# print corpus
		input_vector = []
		output1_vector = []
		output2_vector = []
		while (len(stack_buffer) != 0):
			(input_vec) = construct_feature(stack_sigma, stack_buffer, dtree, sentence)
			(stack_sigma, stack_buffer, dtree, step, relation) = get_next_step(stack_sigma, stack_buffer, dtree, sentence)
			if (relation):
				relation = relation.split(":")[0]
			(out1, out2) = construct_output(step, relation)
			input_vector.append(input_vec)
			output1_vector.append(out1)
			output2_vector.append(out2)
			# print step
			# print dtree
			# print stack_sigma
			# print stack_buffer
			# print vec
		clf1.fit(input_vector, output1_vector)
		clf2.fit(input_vector, output2_vector)

	
	print "Testing starts:"
	(data, tree) = read_data("data/test.conllu")
	total1 = 0
	total2 = 0
	correct1 = 0
	correct2 = 0
	for sentence in data:
		size = len(sentence)
		stack_sigma = [0]
		stack_buffer = []
		dtree = {}
		for i in range(size, 0, -1):
			stack_buffer.append(i)
		input_vector = []
		output1_vector = []
		output2_vector = []
		while (len(stack_buffer) != 0):
			(input_vec) = construct_feature(stack_sigma, stack_buffer, dtree, sentence)
			(stack_sigma, stack_buffer, dtree, step, relation) = get_next_step(stack_sigma, stack_buffer, dtree, sentence)
			# out1_pred = clf1.predict(input_vec)
			# out2_pred = clf2.predict(input_vec)
			if (relation):
				relation = relation.split(":")[0]
			(out1, out2) = construct_output(step, relation)
			# print out1_pred
			# print out1
			# print out2_pred
			# print out2
			# exit()
			input_vector.append(input_vec)
			output1_vector.append(out1)
			output2_vector.append(out2)
		result_1 = clf1.predict(input_vector)
		result_2 = clf2.predict(input_vector)
		# print result_1[0].shape
		(correct1, total1) = getAccuracy(result_1, output1_vector, correct1, total1)
		(correct2, total2) = getAccuracy(result_2, output2_vector, correct2, total2)

	print (correct1/(total1*1.0))
	print (correct2/(total2*1.0))

main()
# (data, tree) = read_data()
# stack_sigma = [0]

# (stack_sigma, stack_buffer, buf_pos, step) = get_next_step(stack, data[0], 0)
# print stack_sigma
# print buf_pos
# print step