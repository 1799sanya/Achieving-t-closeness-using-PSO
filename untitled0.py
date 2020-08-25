
import numpy as np
import pandas as pd

from particle import Particle
from utils import standardize
from utils import normalize
import matplotlib.pyplot as plt 

class ParticleSwarmOptimizedClustering:
    def __init__(self,
                 n_cluster: int,
                 n_particles: int,
                 data: np.ndarray,
                 max_iter: int = 100,
                 print_debug: int = 10):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.print_debug = print_debug
        self.gbest_score = np.inf
        self.gbest_centroids = None
        self.gbest_sse = np.inf
        self._init_particles()

    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            particle = Particle(self.n_cluster, self.data)
            
            if particle.best_score < self.gbest_score:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_score = particle.best_score
            self.particles.append(particle)
            self.gbest_sse = min(particle.best_sse, self.gbest_sse)
            par = []
            par.append(self.gbest_sse)
        
            
        
        
            

    def run(self):
        print('Initial global best score', self.gbest_score)
        history = []
        myarray = []
        
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.gbest_centroids, self.data)


                
                #print(i, particle.best_score, self.gbest_score)
                if particle.best_score < self.gbest_score:
                    self.gbest_centroids = particle.centroids.copy()
                    self.gbest_score = particle.best_score
                if i == 99:
                    myarray.append(self.gbest_score) #gbest for every particle
            
                    
            history.append(self.gbest_score) #gbest for every itteration

            
            if i % self.print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                    i + 1, self.max_iter, self.gbest_score))
        
        print('Finish with gbest score {:.18f}'.format(self.gbest_score))
        clus = particle.update(self.gbest_centroids, self.data)
        return clus
        

dataset = pd.read_csv("PPDPfinal.csv")
x = dataset.iloc[:100 , [1,3]].values



normalize(x)
# print(x.head())
pso = ParticleSwarmOptimizedClustering(n_cluster=5  , n_particles=100, data=x)  #, max_iter=2000, print_debug=50)

clus = pso.run()
newdf =dataset[0:100]
newdf['CLUSTER'] = clus 
count_row = newdf.shape[0]

column_names = ["ID","AGE","SEX","ZIP","DISEASE"]
empty_df = pd.DataFrame(columns = column_names) 

  
entry0 = newdf.loc[newdf['CLUSTER'] == 0]
df_0 = pd.concat([empty_df,entry0])
        
entry1 = newdf.loc[newdf['CLUSTER'] == 1]
df_1 = pd.concat([empty_df,entry1])
        
entry2 = newdf.loc[newdf['CLUSTER'] == 2]
df_2 = pd.concat([empty_df,entry2])
        
entry3 = newdf.loc[newdf['CLUSTER'] == 3]
df_3 = pd.concat([empty_df,entry3])
        
entry4 = newdf.loc[newdf['CLUSTER'] == 4]
df_4 = pd.concat([empty_df,entry4])


row_0 = df_0.shape[0]
row_1 = df_1.shape[0]
row_2 = df_2.shape[0]
row_3 = df_3.shape[0]
row_4 = df_4.shape[0]
count_0=count_1=count_2=count_3=count_4=0
count = []
j=2

count_0 = len(df_0[df_0['DISEASE'] == j])
if row_0 == 0:
    count.append(0.0)
else:
    count.append(count_0/row_0)     

count_1 = len(df_1[df_1['DISEASE'] == j])
if row_1 == 0:
    count.append(0.0)
else:
    count.append(count_1/row_1) 
 
count_2 = len(df_2[df_2['DISEASE'] == j])
if row_2 == 0:
    count.append(0.0)
else:
    count.append(count_2/row_2) 

count_3 = len(df_3[df_3['DISEASE'] == j])
if row_3 == 0:
    count.append(0.0)
else:
    count.append(count_3/row_3) 
 
count_4 = len(df_4[df_4['DISEASE'] == j])
if row_4 == 0:
    count.append(0.0)
else:
    count.append(count_4/row_4) 

threshold = len(newdf[newdf['DISEASE'] == j])/count_row
upper=threshold+0.03

while ((count[0]>upper) or (count[1]>upper) or (count[2]>upper) or (count[3]>upper) or (count[4]>upper)):
    for i in range (count_0):
        df_00 = df_0.loc[df_0['DISEASE'] == j]
        if threshold>count[0] or len(df_00.index) == 0:
            break
        else:
            row = df_00.sample()
            id = row.iloc[0 ,3]
            df_1 = df_1.append(row)
            df_0 = df_0.drop(df_0[df_0['ID']== id].index)
            count_0 = len(df_0[df_0['DISEASE'] == j])
            row_0 = df_0.shape[0]
            if row_0 ==0:
                count[0]=0
            else:
                count[0] = count_0/row_0
            count_1 = len(df_1[df_1['DISEASE'] == j])
            row_1 = df_1.shape[0]
            if row_1 ==0:
                count[1]=0
            else:
                count[1] = count_1/row_1
                
    for i in range (count_1):
        df_11 = df_1.loc[df_1['DISEASE'] == j]
        if threshold>count[1] or len(df_11.index) == 0:
            break
        else:
            row = df_11.sample()
            id = row.iloc[0 ,3]
            df_2=df_2.append(row)
            df_1=df_1.drop(df_1[df_1['ID']== id].index)
            count_1 = len(df_1[df_1['DISEASE'] == j])
            row_1 = df_1.shape[0]
            if row_1 ==0:
                count[1]=0
            else:
                count[1] = count_1/row_1
            count_2 = len(df_2[df_2['DISEASE'] == j])
            row_2 = df_2.shape[0]
            if row_2 ==0:
                count[2]=0
            else:
                count[2] = count_2/row_2
            
    for i in range (count_2):
        df_22 = df_2.loc[df_2['DISEASE'] == j]
        if threshold>count[2] or len(df_22.index) == 0:
            break
        else:
            row = df_22.sample()
            id = row.iloc[0 ,3]
            df_3 = df_3.append(row)
            df_2 = df_2.drop(df_2[df_2['ID']== id].index)
            count_2 = len(df_2[df_2['DISEASE'] == j])
            row_2 = df_2.shape[0]
            if row_2 ==0:
                count[2]=0
            else:
                count[2] = count_2/row_2
            count_3 = len(df_3[df_3['DISEASE'] == j])
            row_3 = df_3.shape[0]
            if row_3 ==0:
                count[3]=0
            else:
                count[3] = count_3/row_3
            
    for i in range (count_3):
        df_33 = df_3.loc[df_3['DISEASE'] == j]
        if threshold>count[3] or len(df_33.index) == 0 :
            break
        else:
            row = df_33.sample()
            id = row.iloc[0 ,3]
            df_4=df_4.append(row)
            df_3 = df_3.drop(df_3[df_3['ID']== id].index)
            count_3 = len(df_3[df_3['DISEASE'] == j])
            row_3 = df_3.shape[0]
            if row_3 ==0:
                count[3]=0
            else:
                count[3] = count_3/row_3
            count_4 = len(df_4[df_4['DISEASE'] == j])
            row_4 = df_4.shape[0]
            if row_4 ==0:
                count[4]=0
            else:
                count[4] = count_4/row_4
            
            
    for i in range (count_4):
        df_44 = df_4.loc[df_4['DISEASE'] == j]
        if threshold>count[4] or len(df_44.index) == 0:
            break
        else:
            row = df_44.sample()
            id = row.iloc[0 ,3]
            df_0 = df_0.append(row)
            df_4 = df_4.drop(df_4[df_4['ID']== id].index)
            count_4 = len(df_4[df_4['DISEASE'] == j])
            row_4 = df_4.shape[0]
            if row_4 ==0:
                count[4]=0
            else:
                count[4] = count_4/row_4
            count_0 = len(df_0[df_0['DISEASE'] == j])
            row_0 = df_0.shape[0]
            if row_0 ==0:
                count[0]=0
            else:
                count[0] = count_0/row_0
        
print("end")   
    