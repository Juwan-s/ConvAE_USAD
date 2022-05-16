import torch
import torch.nn as nn
import time

from utils import *
device = get_default_device()

class Conv1dEncoder(nn.Module):
  def __init__(self, w_size, latent_size, kernel_size, first_layer, feature_dim):
    super().__init__()

    self.w_size = w_size
    self.kernel_size = kernel_size
    self.feature_dim = feature_dim
    self.latent_size = latent_size
    self.first_layer = first_layer
    self.out_dim = (self.first_layer * 4) * (self.w_size - 3 * (self.kernel_size - 3))
    # self.out_dim = (self.first_layer//2) * (self.w_size - (self.kernel_size - 3))

    # self.conv_encoder = nn.Sequential(

    #   nn.Conv1d(self.feature_dim, self.first_layer, self.kernel_size  ,padding=1),
    #   # nn.BatchNorm1d(self.first_layer),
    #   nn.GELU(),
    #   nn.Conv1d(self.first_layer, self.first_layer*2, self.kernel_size, padding=1),
    #   # nn.BatchNorm1d(self.first_layer*2),
    #   nn.GELU(),
    #   nn.Conv1d(self.first_layer*2, self.first_layer*4, self.kernel_size, padding=1),
    #   # nn.BatchNorm1d(self.first_layer*4),
    #   nn.GELU()
    #   )

    self.conv_encoder = nn.Sequential(

      nn.Conv1d(self.feature_dim, self.feature_dim//2, self.kernel_size  ,padding=1),
      # nn.BatchNorm1d(self.first_layer),
      nn.GELU(),
      # nn.Dropout(0.2),
      nn.Conv1d(self.feature_dim//2, self.feature_dim//4, self.kernel_size, padding=1),
      # nn.BatchNorm1d(self.first_layer*2),
      nn.GELU(),
      # nn.Dropout(0.2),
      nn.Conv1d(self.feature_dim//4, self.feature_dim//8, self.kernel_size, padding=1),
      # nn.BatchNorm1d(self.first_layer*4),
      nn.GELU()
      
      )
    # self.flatten = nn.Flatten()
    
    # self.linear = nn.Linear(self.out_dim, self.latent_size)
        
  def forward(self, w):

    z = self.conv_encoder(w)

    # z = self.flatten(z)

    # z = self.linear(z)
    return z

class Conv1dDecoder(nn.Module):
  def __init__(self, latent_size, w_size, kernel_size, first_layer, feature_dim):
    super().__init__()

    self.first_layer = first_layer
    self.kernel_size = kernel_size
    self.w_size = w_size
    self.latent_size = latent_size
    self.out_dim = (self.first_layer * 4) * (self.w_size - 3 * (self.kernel_size - 3))
    # self.out_dim = (self.first_layer//2) * (self.w_size - (self.kernel_size - 3))

    
    self.feature_dim = feature_dim
    # self.linear = nn.Linear(self.latent_size, self.out_dim)

    # self.unflatten = nn.Unflatten(1, (self.first_layer//2, (self.w_size - (self.kernel_size - 3))))

    # self.conv_decoder = nn.Sequential(

    #   nn.ConvTranspose1d(self.first_layer*4, self.first_layer * 2, self.kernel_size, padding=1),
    #   # nn.BatchNorm1d(self.first_layer*2),
    #   nn.GELU(),
    #   nn.Dropout(0.1),
    #   nn.ConvTranspose1d(self.first_layer * 2, self.first_layer, self.kernel_size, padding=1),
    #   # nn.BatchNorm1d(self.first_layer),
    #   nn.GELU(),
    #   nn.Dropout(0.1),
    #   nn.ConvTranspose1d(self.first_layer, self.feature_dim, self.kernel_size, padding=1),
    #   # nn.BatchNorm1d(self.feature_dim),
    #   nn.Sigmoid()

    # )
    self.conv_decoder = nn.Sequential(

      nn.ConvTranspose1d(self.feature_dim//8, self.feature_dim//4, self.kernel_size, padding=1),
      # nn.BatchNorm1d(self.first_layer*2),
      nn.GELU(),
      nn.ConvTranspose1d(self.feature_dim//4, self.feature_dim//2, self.kernel_size, padding=1),
      # nn.BatchNorm1d(self.first_layer),
      nn.GELU(),
      nn.ConvTranspose1d(self.feature_dim//2, self.feature_dim, self.kernel_size, padding=1),
      # nn.BatchNorm1d(self.feature_dim),
      nn.Sigmoid()

    )
        
  def forward(self, z):
    
    # z = self.linear(z)
    # z = self.unflatten(z)

    w = self.conv_decoder(z)
    
    
    return w

class Conv1dModel(nn.Module):
  def __init__(self, w_size, z_size, k_size, config):
    super().__init__()
    
    first_layer = config['first_layer']
    feature_dim = config['feature_dim']

    self.encoder = Conv1dEncoder(w_size, z_size, k_size, first_layer, feature_dim)
    self.decoder1 = Conv1dDecoder(z_size, w_size, k_size, first_layer, feature_dim)
    self.decoder2 = Conv1dDecoder(z_size, w_size, k_size, first_layer, feature_dim)
  
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))

def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def Conv1dtraining(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
# def Conv1dtraining(epochs, model, train_loader, val_loader):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    # optimizer1 = torch.optim.SGD(params = (list(model.encoder.parameters())+list(model.decoder1.parameters())), lr = 0.0001)
    # optimizer2 = torch.optim.SGD(params = (list(model.encoder.parameters())+list(model.decoder2.parameters())), lr = 0.0001)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def Usadtraining(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def Conv1dtesting(model, test_loader, alpha=.5, beta=.5):
    results=[]
    for [batch] in test_loader:
        batch=to_device(batch,device)
        w1=model.decoder1(model.encoder(batch))
        w2=model.decoder2(model.encoder(w1))
        results.append(alpha* torch.mean(torch.mean((batch-w1)**2,axis=1), axis=1).detach().cpu()  + beta* torch.mean(torch.mean((batch-w2)**2,axis=1), axis = 1).detach().cpu() )
    return results


class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))

def Usadtesting(model, test_loader, alpha=.5, beta=.5):
    results=[]
    for [batch] in test_loader:
        batch=to_device(batch,device)
        w1=model.decoder1(model.encoder(batch))
        w2=model.decoder2(model.encoder(w1))
        results.append((alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1)).detach().cpu())
    return results