import umap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

def umap_plot(data,color,components=2):
    
    n_neighbors = [15]
    
    for i in n_neighbors:
        reducer = umap.UMAP(n_components=components,n_neighbors=i)
        embedding = reducer.fit_transform(data)
        embedding_df = pd.DataFrame(embedding)
        embedding_df['Subgroups']= color
        embedding_df = embedding_df.rename(columns={0: "Dim 1", 1: "Dim 2"})
        if components == 3:
            embedding_df = embedding_df.rename(columns={2: "Dim 3"})
        return embedding_df
            
def load_data():
    data = pd.read_csv('Medulloblastoma Files\Medulloblastoma_Cavalli_VAE_data_Less.csv', sep=',', na_values=".")
    data = data.rename(columns={'Unnamed: 0': 'Patient'})

    subgroups = pd.read_csv('Medulloblastoma Files\GSE85218_subgroups.csv', sep=' ',header=None)
    colors_train = subgroups[1].values

    data_test = pd.read_csv('Medulloblastoma Files\Medulloblastoma_Northcott_VAE_data_Less.csv', sep=',', na_values=".")
    data_test = data_test.rename(columns={'Unnamed: 0': 'Patient'})

    subgroups_test = pd.read_csv('Medulloblastoma Files\GSE37382_subgroups.csv', sep=' ',header=None)
    colors = subgroups_test[1].values

    data = data.drop(['Patient'],axis=1)
    data_test = data_test.drop(['Patient'],axis=1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaler.fit(data)
    data = scaler.transform(data) #(x - mu / s) almost all values between -1,1

    scaler.fit(data_test)
    data_test = scaler.transform(data_test)
    
    data = pd.DataFrame(data)
    train_dataset = torch.tensor(data.values).float()

    data_test = pd.DataFrame(data_test)
    test_dataset = torch.tensor(data_test.values).float()


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    return train_dataset, test_dataset, colors, colors_train

def data_generation(N,data,test_dataset,colors, model):

    data = int(data)
    
    if data > len(test_dataset):
        return test_dataset, colors
    
    sample = np.zeros(shape=(1,features))

    data = test_dataset[data]

    with torch.no_grad():
        data = data.to(device)
        reconstruction, mean, logvar, coded = model(data)          

    for i in range(0,N):
        std = torch.exp(0.5*logvar) 
        eps = torch.randn_like(std) 
        resultado = mean + (eps*std)
        sample = np.concatenate((sample, resultado), axis=0)

    sample = sample.reshape(N+1,features)
    z = sample[1:]

    z = torch.from_numpy(z)
    z = z.float()

    with torch.no_grad():  
        z = z.to(device)
        samples = model.decoder(z)   #decode the data
    generated = torch.cat([test_dataset, samples], dim=0) #concat the test data and the generate data to visualize it
    new_colors = np.array(['Generated']*len(samples)) #create the reference to paint black the generate examples
    colors_generated = np.concatenate((colors,new_colors),axis=0) #concat the colors of the test data and the generate data

    return generated, colors_generated

features = 32

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=kwargs["mid_dim"]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["mid_dim"], out_features=features*2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=features, out_features=kwargs["mid_dim"]),
            nn.ReLU(),
            nn.Linear(in_features=kwargs["mid_dim"], out_features=kwargs["input_shape"]),
            nn.Tanh()
            #nn.Sigmoid()
        )

    def reparametrize(self, mu, log_var):

        if self.training:
            std = torch.exp(0.5*log_var) 
            eps = torch.randn_like(std) 
            sample = mu + (eps*std) 
        else:
            sample = mu
        return sample

    def forward(self, x):

        mu_logvar = self.encoder(x).view(-1,2,features)
        mu = mu_logvar[:, 0, :] 
        log_var = mu_logvar[:, 1, :] 

        z = self.reparametrize(mu,log_var) 
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var, z

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    PATH = './vaeAnnealing32MSE.pth'
    model = VAE(input_shape=5668, mid_dim=2048).to(device)
    model.load_state_dict(torch.load(PATH))
    return model

def data_interpolation(N,patient1,patient2, colors, test_dataset, model):

    z1 = test_dataset[patient1]
    z2 = test_dataset[patient2]

    with torch.no_grad(): 
        z1 = z1.to(device)
        z2 = z2.to(device)
        reference1, mean1, logvar1, _ = model(z1)                
        reference2, mean2, logvar2, _ = model(z2)  

    sample = np.zeros(shape=(1,features))
    for i in range(0,N):
        mean = i / (N - 1) * mean2 + (1 - i / (N - 1) ) * mean1 #interpolation mean
        std1 = torch.exp(0.5*logvar1)
        std2 = torch.exp(0.5*logvar2)
        std = i / (N - 1) * std2 + (1 - i / (N - 1) ) * std1 #interpolation logvar
        eps = torch.randn_like(std) 
        resultado = mean + (eps*std)
        sample = np.concatenate((sample, resultado), axis=0)
    sample = sample.reshape(N+1,features)
    z = sample[1:]

    z = torch.from_numpy(z) #preprocessing to introduce samples in NN
    z = z.float()

    
    #GENERATE INTERPOLATION DATA
    with torch.no_grad():        
        z = z.to(device)
        samples = model.decoder(z)   #decode the data
    generated = torch.cat([test_dataset, samples], dim=0) #concat the test data and the generate data to visualize it
    new_colors = np.array(['Generated']*len(samples)) #create the reference to paint black the generate examples
    colors_generated = np.concatenate((colors,new_colors),axis=0) #concat the colors of the test data and the generate data

    # ADD REFERENCES
    generated = torch.cat((generated,reference1),axis=0) # add the centroids to the data to plot them
    colors_reference = np.array(['Reference1']) #add label generated
    colors_generated = np.concatenate((colors_generated,colors_reference),axis=0) #generate new colors

    generated = torch.cat((generated,reference2),axis=0) # add the centroids to the data to plot them
    colors_reference = np.array(['Reference2']) #add label generated
    colors_generated = np.concatenate((colors_generated,colors_reference),axis=0) #generate new colors

    return generated, colors_generated