import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import kmedoids

def solve_intersections(p1, p2, d13, d23):
	#return intersections for circles at p1, p2
	#with radius d13 and d23 respectedly
	#Naming convention because this is for solving
	#the position of p3 when distances to p1 and p2 are known
	v12 = p2 - p1
	d12 = np.linalg.norm(v12)

	p3_1 = 0.5*( p1 + p2 ) + (( d13**2 - d23**2 ) / (2*d12**2)) * (p2 - p1) \
			+ np.sqrt(2*((d13**2 + d23**2) / (d12**2) ) -  ( (d13**2 - d23**2) / (d12**4) ) - 1) \
			* np.asarray( [p2[1] - p1[1], p2[0] - p1[0]])

	p3_2 = 0.5*( p1 + p2 ) + (( d13**2 - d23**2 ) / (2*d12**2)) * (p2 - p1) \
			- np.sqrt(2*((d13**2 + d23**2) / (d12**2) ) -  ( (d13**2 - d23**2) / (d12**4) ) - 1) \
			* np.asarray( [p2[1] - p1[1], p2[0] - p1[0]])

	return(p3_1, p3_2)

def step_p(distance_matrix, points, indexes, index, step, beta=0.1):
	loss = 0.0
	for i in range(len(points)):
		loss += np.abs(distance_matrix[indexes[i], indexes[index]] - np.linalg.norm(points[i] - points[index]))

	loss_new = 0.0
	for i in range(len(points)):
		loss_new += np.abs(distance_matrix[indexes[i], indexes[index]] - np.linalg.norm(points[i] - (points[index] + step)))

	if (loss_new <= loss):
		p = 1.0
	else:
		p = np.exp(-beta*(loss_new - loss))
	return(p)


def optimize_positions(distance_matrix, points, indexes, iterations = 100000):
		max_step = 1.0
		T_0 = 1.0
		
		k = len(points)
		
		loss = 0.0
		for i in range(k):
			for j in range(i,k):
				loss += np.abs(distance_matrix[indexes[i], indexes[j]] - np.linalg.norm(points[i] - points[j]))
		
		for i in range(iterations):
			beta =  (1 + np.log(1 + i)) / T_0
			angle = 2*np.pi*np.random.random()
			distance = np.random.random()*max_step
			#Choose point:
			p = np.random.randint(k)
			step = np.asarray([distance*np.cos(angle), distance*np.sin(angle)])
			prob = step_p(distance_matrix, points, indexes, p, step, beta)
			r = np.random.random()
			
			if(r < prob):
				points[p] += step


		loss_new = 0.0
		for i in range(k):
			for j in range(i,k):
				loss_new += np.abs(distance_matrix[indexes[i], indexes[j]] - np.linalg.norm(points[i] - points[j]))

		print(loss)
		print(loss_new)

def plot_distances(distance_matrix, groups, spacing = 1000.0):
	#groups is kmedoids resulting object

	k = len(groups.medoids)
	
	#Random starting positions
	medoids = []
	pos = []
	for medoid in groups.medoids:
		#pos.append(np.asarray([100.0*np.random.random_sample(), 100.0*np.random.random_sample()]))
		pos.append(np.asarray([0.0, 0.0]))
		medoids.append(int(medoid))

	#Simulated annealing to get the points to minimize the errors in distances
	optimize_positions(distance_matrix, pos, medoids)
	
	points = []
	for i in range(k):
		points.append([])
	
	for i in range(len(groups.labels)):
		if(i in medoids):
			continue
		g = groups.labels[i]
		offset = pos[g]
		distance = distance_matrix[i, int(groups.medoids[g])]

		#second best distance
		second_best_d = np.inf
		second_best_ind = 0
		for j in range(k):
			if(j == g):
				continue
			else:
				d = distance_matrix[i, int(groups.medoids[j])]
				if(d < second_best_d):
					second_best_d = d
					second_best_ind = j

		r = np.random.random()
		#if(r>0.5):
		points[g].append(offset + solve_intersections(pos[g], pos[second_best_ind], distance, second_best_d)[0])
		#else:
		#	points[g].append(offset + solve_intersections(pos[g], pos[j], distance, d)[1])
		
	ax = plt.gca()
	i = 0
	for point_group in points:
		color = next(ax._get_lines.prop_cycler)['color']
		a = np.asarray(point_group)
		pos = np.asarray(pos)
		plt.plot(a[:,0], a[:,1], '.', color = color)
		plt.plot(pos[i,0], pos[i,1], '*', color = color)
		i += 1
	
	plt.show()

def manhattan_distance_mapping(excel):
	#Manhattan distance of survey results
	data = excel.to_numpy(dtype=np.float64)
	n_companies = len(data[:,0])
	d_matrix = np.zeros((n_companies, n_companies))

	for i in range(n_companies):
		j = 0
		for j in range(i, n_companies):
			d_matrix[i,j] = np.asarray([x for x in np.abs(data[i,:] - data[j,:]) if np.isnan(x) == False]).sum()
			d_matrix[j,i] = np.asarray([x for x in np.abs(data[i,:] - data[j,:]) if np.isnan(x) == False]).sum()

	return(d_matrix)

def euclidean_distance_mapping(excel):
	data = excel.to_numpy(dtype=np.float64)
	n_companies = len(data[:,0])
	d_matrix = np.zeros((n_companies, n_companies))

	for i in range(n_companies):
		j = 0
		for j in range(i, n_companies):
			d_matrix[i,j] = np.sqrt(np.asarray([x for x in (data[i,:] - data[j,:])**2 if np.isnan(x) == False]).sum())
			d_matrix[j,i] = np.sqrt(np.asarray([x for x in (data[i,:] - data[j,:])**2 if np.isnan(x) == False]).sum())

	return(d_matrix)
	
def parse_excel(excel):
	#In our case Column indexes 33-67 are survey questions with
	#1-7 scale for answers. We use thse for distance mapping
	return(excel.iloc[:,33:67])

def main():
	if(len(sys.argv)>=2):
		fname = sys.argv[1]
	else:
		print("No filename given", file=sys.stderr)

	if(len(sys.argv)>=3):
		k = int(sys.argv[2])
	else:
		#Based on k-loss curve when distances are manhattan distances
		k = 6


	excel = pd.read_excel(fname)
	survey_answers = parse_excel(excel)

	distance_matrix = manhattan_distance_mapping(survey_answers)
	#distance_matrix = euclidean_distance_mapping(survey_answers)

	#221 is all nulls in our data
	distance_matrix = np.delete(distance_matrix, 221, 0)
	distance_matrix = np.delete(distance_matrix, 221, 1)
	#k == 6 based on k-loss plot where the diminishing returns start
	groups = kmedoids.fasterpam(distance_matrix, k)

	group_list = []
	title_key = 5
	for medoid in groups.medoids:
		if(medoid>=221):
			group_list.append([excel.iat[int(medoid)+1,title_key]])
		else:
			group_list.append([excel.iat[int(medoid),title_key]])

	for i in range(len(groups.labels)):
		g_ind = groups.labels[i]
		if(i>=221):
			group_list[g_ind].append(excel.iat[i+1,title_key])
		else:
			group_list[g_ind].append(excel.iat[i,title_key])

	"""
	for group in group_list:
		for title in group:
			print(title)
		print("----------------")
	"""

	plot_distances(distance_matrix, groups, 50.0)
	

main()
