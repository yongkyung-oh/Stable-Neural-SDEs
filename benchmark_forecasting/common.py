import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm
import os
import time 
import models
OUR_MODEL =["learnable_forecasting"]

here = pathlib.Path(__file__).resolve().parent


def _add_weight_regularisation(loss_fn, regularise_parameters, scaling=0.03):
    def new_loss_fn(pred_y, true_y):
        total_loss = loss_fn(pred_y, true_y)
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
        return total_loss
    return new_loss_fn
class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        
        return self.model(*args, **kwargs).squeeze(-1)


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]



def _evaluate_metrics(dataloader, model, times, loss_fn, num_classes, device, kwargs, model_name):
    with torch.no_grad():
        total_accuracy = 0
        total_confusion = torch.zeros(num_classes, num_classes).numpy()  
        total_dataset_size = 0
        total_loss = 0
        mse_loss = 0
        logpz_loss = 0
        true_y_cpus = []
        pred_y_cpus = []

        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)

            if 'learnable' in model_name:
                pred_y, loss1, loss2 = model(times, coeffs, lengths, **kwargs)
            else:
                if model_name == 'ncde':
                    pred_y = model(times, coeffs, lengths, **kwargs)
                else:
                    pred_y = model(times, coeffs, lengths)
                loss1, loss2 = 0, 0

            if len(pred_y.shape) ==2:
                if len(pred_y.shape)==len(true_y.shape):
                    pred_y = pred_y
                else:
                    pred_y = pred_y.squeeze(-1)
            if num_classes == 2:
                thresholded_y = (pred_y > 0).to(true_y.dtype)
            else:
                thresholded_y = torch.argmax(pred_y, dim=1)
            true_y_cpu = true_y.detach().cpu()
            pred_y_cpu = pred_y.detach().cpu()
            if num_classes == 2:
                # Assume that our datasets aren't so large that this breaks
                true_y_cpus.append(true_y_cpu)
                pred_y_cpus.append(pred_y_cpu)
            thresholded_y_cpu = thresholded_y.detach().cpu()

            total_accuracy += (thresholded_y == true_y).sum().to(pred_y.dtype)
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y) * batch_size

            mse_loss += loss1 *batch_size
            logpz_loss += loss2 * batch_size
            
        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        mse_loss /= total_dataset_size
        logpz_loss /= total_dataset_size
        total_accuracy /= total_dataset_size
        # total_loss : downstream task loss (MSE)
        # mse_loss : MSE loss 
        # logpz_loss : logpz_t 
        if 'learnable' in model_name:
            metrics = _AttrDict(accuracy=total_accuracy.item(), dataset_size=total_dataset_size,
                                loss=total_loss.item(),mse_loss = mse_loss.item(),logpz = logpz_loss.item())
        else:
            metrics = _AttrDict(accuracy=total_accuracy.item(), dataset_size=total_dataset_size,
                                loss=total_loss.item(),mse_loss = 0.0,logpz = 0.0)
        
        if num_classes == 2:
            true_y_cpus = torch.cat(true_y_cpus, dim=0)
            pred_y_cpus = torch.cat(pred_y_cpus, dim=0)
            metrics.auroc = sklearn.metrics.roc_auc_score(true_y_cpus, pred_y_cpus)
            metrics.average_precision = sklearn.metrics.average_precision_score(true_y_cpus, pred_y_cpus)
        return metrics




def _evaluate_metrics_forecasting(model_name, dataloader, model, times, loss_fn, device, kwargs):
    with torch.no_grad():
        total_dataset_size = 0
        total_loss = 0
        mse_loss = 0
        logpz_loss = 0
        
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)

            if model_name in OUR_MODEL :    
                pred_y, loss1, loss2 = model(times, coeffs, lengths, **kwargs)
            
            else:
                pred_y = model(times, coeffs, lengths, **kwargs)
                loss1 = 0
                loss2 = 0
                
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y) * batch_size
            mse_loss += loss1 *batch_size
            logpz_loss += loss2 * batch_size

        # total_loss : downstream task loss (MSE)
        # mse_loss : MSE loss 
        # logpz_loss : logpz_t 
        total_loss /= total_dataset_size  
        mse_loss /= total_dataset_size
        logpz_loss /= total_dataset_size
        
        if 'learnable' in model_name:
            metrics = _AttrDict( dataset_size=total_dataset_size,
                                loss=total_loss.item(),mse_loss = mse_loss.item(),logpz = logpz_loss.item())
        else:
            metrics = _AttrDict( dataset_size=total_dataset_size,
                                loss=total_loss.item(),mse_loss = 0.0,logpz = 0.0)
    
        return metrics

class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True

def _train_loop_forecasting(model_name, train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, max_epochs,
                           writer, device, c1, c2, kwargs, step_mode) :
                           
    model.train()
    best_model = model
    best_train_loss = math.inf
    
    best_train_loss_epoch = 0
    best_val_loss = 0
    
    history = []
    breaking = False
    # scheduler : Reduce learning rate when a metric has stopped improving.
    if step_mode == 'trainloss':
        
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
    elif step_mode=='valloss':
        
        epoch_per_metric = 1
        plateau_terminate = 100
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
    elif step_mode == 'valaccuracy':
        
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        

    elif step_mode=='valauc':
        
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        
    elif step_mode=='none':
        epoch_per_metric=1 
        plateau_terminate=100

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    for epoch in tqdm_range:
        if breaking:
            break
        
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                
                *train_coeffs, train_y, lengths = batch
                
                if  model_name in OUR_MODEL:
                    pred_y, loss1, loss2 = model(times,train_coeffs,lengths,**kwargs)
                
                else:
                    #baseline    
                    pred_y = model(times, train_coeffs, lengths, **kwargs)
                    loss1 , loss2 =0 , 0
                # loss_task : downstream task loss. (MSE)
                loss_task = loss_fn(pred_y, train_y)
                
                # c1 : alpha in eq(11) in paper
                # c2 : beta in eq(11) in paper
                # loss1 : MSE loss  
                # loss2 : logpz_t 
                loss_y = c1*loss1 +c2*loss2
                loss =  loss_y + loss_task 
                
                # loss : final loss which is the sum of Ly and a task loss 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            # evaluate train,val,test. 
            train_metrics = _evaluate_metrics_forecasting(model_name, train_dataloader, model, times, loss_fn,  device, kwargs)
            val_metrics = _evaluate_metrics_forecasting(model_name, val_dataloader, model, times, loss_fn,  device, kwargs)
            test_metrics = _evaluate_metrics_forecasting(model_name, test_dataloader, model, times, loss_fn,  device,kwargs)
            
            writer.add_scalar('train/loss', train_metrics.loss, epoch)
            writer.add_scalar('train/mseloss', train_metrics.mse_loss, epoch)
            writer.add_scalar('train/LogPz', train_metrics.logpz, epoch)
            
            
            writer.add_scalar('validation/loss', val_metrics.loss, epoch)
            writer.add_scalar('validation/mseloss',
                              val_metrics.mse_loss, epoch)
            writer.add_scalar('validation/LogPz', val_metrics.logpz, epoch)
            
            writer.add_scalar('test/loss', test_metrics.loss, epoch)
            writer.add_scalar('test/mseloss', test_metrics.mse_loss, epoch)
            writer.add_scalar('test/LogPz', test_metrics.logpz, epoch)

            
            model.train()
            
            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch
                        
            if val_metrics.loss * 1.0001 < best_val_loss:
                best_val_loss = val_metrics.loss
                best_val_loss_epoch = epoch
         
            
            tqdm_range.write('Epoch: {}  Train loss: {:.3} MSE loss : {:.3} Logpz : {:.3}  Val loss: {:.3}  '
                                'MSE loss : {:.6} Logpz : {:.3} Test loss : {:.3} Test MSE loss : {:.6} Logpz :{:.3}'
                                ''.format(epoch, train_metrics.loss, train_metrics.mse_loss, train_metrics.logpz,  val_metrics.loss,
                                        val_metrics.mse_loss, val_metrics.logpz, test_metrics.loss,test_metrics.mse_loss,test_metrics.logpz))
                
            
            if step_mode == 'trainloss':
                scheduler.step(train_metrics.loss)
            elif step_mode=='valloss':
                scheduler.step(val_metrics.loss)
            elif step_mode == 'valaccuracy':
                scheduler.step(val_metrics.accuracy)
            elif step_mode=='valauc':
                scheduler.step(val_metrics.auroc)

                
            history.append(_AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))
            # Early stop
            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                    ''.format(plateau_terminate))
                breaking = True
            
           

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    return history,epoch




def _train_loop(train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, max_epochs, num_classes,device,c1,c2,
                kwargs, step_mode, model_name):
    model.train()
    best_model = model
    best_train_loss = math.inf
    best_train_accuracy = 0
    best_val_accuracy = 0
    best_val_auc= 0
    best_train_accuracy_epoch = 0
    best_train_loss_epoch = 0
    history = []
    breaking = False
    best_train_auc = 0
    # scheduler : Reduce learning rate when a metric has stopped improving.
    
    if step_mode == 'trainloss':
        print("trainloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        

    elif step_mode=='valloss':
        print("valloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        

    elif step_mode == 'valaccuracy':
        print("valaccuracy")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        

    elif step_mode=='valauc':
        print("valauc")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        
    elif step_mode=='none':
        epoch_per_metric=1 
        plateau_terminate=50

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    for epoch in tqdm_range:
        if breaking:
            break
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                
                *train_coeffs, train_y, lengths = batch
                
                if 'learnable' in model_name:
                    pred_y, loss1, loss2 = model(times, train_coeffs, lengths, **kwargs)
                # baselines
                elif model_name == 'ncde':
                    pred_y = model(times, train_coeffs, lengths, **kwargs)
                
                else:
                    pred_y = model(times, train_coeffs, lengths)
                if len(pred_y.shape) ==2:
                    if len(pred_y.shape)==len(train_y.shape):
                        pred_y = pred_y
                    else:
                        pred_y = pred_y.squeeze(-1)
                loss_task = loss_fn(pred_y, train_y)
                # c1 : alpha in eq(11) in paper
                # c2 : beta in eq(11) in paper
                # loss1 : MSE loss  
                # loss2 : logpz_t 
                if 'learnable' in model_name:
                    loss = loss_task + (c1*loss1) + (c2*loss2) 
                else:
                    loss = loss_task
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            # evaluate train,val,test. 
            train_metrics = _evaluate_metrics(train_dataloader, model, times, loss_fn, num_classes, device, kwargs, model_name)
            val_metrics = _evaluate_metrics(val_dataloader, model, times, loss_fn, num_classes, device, kwargs, model_name)
            test_metrics = _evaluate_metrics(test_dataloader, model, times, loss_fn, num_classes, device, kwargs, model_name)
            model.train()

            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch       
            
            if num_classes == 2 : 
                if train_metrics.auroc > best_train_auc * 1.001:
                    best_train_auc = train_metrics.auroc
                    best_train_auc_epoch = epoch
                tqdm_range.write('Epoch: {}  Train loss: {:.3} Train MSE {:.3} Train logpz {:.3} Train AUC : {:.3}  Train accuracy: {:.3}\n'
                                                'Val loss: {:.3} Val MSE {:.3} Val logpz {:.3} Val AUC : {:.3}  Val accuracy: {:.3}\n'
                                                'Test loss : {:.3} Test MSE {:.3} Test logpz {:.3} Test auc : {:.3} Test accuracy: {:.3}\n'
                                                ''.format(epoch, train_metrics.loss, train_metrics.mse_loss,train_metrics.logpz,train_metrics.auroc,
                                                train_metrics.accuracy, val_metrics.loss,val_metrics.mse_loss,val_metrics.logpz, val_metrics.auroc, val_metrics.accuracy,
                                                test_metrics.loss,test_metrics.mse_loss,test_metrics.logpz,test_metrics.auroc,test_metrics.accuracy))
                
                if val_metrics.auroc > best_val_auc:
                    best_val_auc = val_metrics.auroc
                    del best_model 
                    best_model = copy.deepcopy(model)

                    print(f"\n[ Epoch {epoch} ] Test Loss : {test_metrics.loss}, Test MSE loss : {test_metrics.mse_loss}, Test logpz : {test_metrics.logpz}, Test auc: {test_metrics.auroc}")

            else:
                if train_metrics.accuracy > best_train_accuracy * 1.001:
                    best_train_accuracy = train_metrics.accuracy
                    best_train_accuracy_epoch = epoch

                tqdm_range.write('Epoch: {}  Train loss: {:.3} MSE loss : {:.3} Logpz : {:.3} Train accuracy: {:.3}  Val loss: {:.3}  '
                                'MSE loss : {:.3} Logpz : {:.3} Val accuracy: {:.3}'
                                ''.format(epoch, train_metrics.loss, train_metrics.mse_loss, train_metrics.logpz, train_metrics.accuracy, val_metrics.loss,
                                        val_metrics.mse_loss, val_metrics.logpz, val_metrics.accuracy))
                
                if val_metrics.accuracy > best_val_accuracy:
                    best_val_accuracy = val_metrics.accuracy
                    del best_model  # so that we don't have three copies of a model simultaneously
                    best_model = copy.deepcopy(model)
                    print(f"\n[ Epoch {epoch} ] Test Loss : {test_metrics.loss}, Test MSE loss : {test_metrics.mse_loss}, Test logpz : {test_metrics.logpz}, Test accuracy: {test_metrics.accuracy}")

                
            
            if step_mode == 'trainloss':
                scheduler.step(train_metrics.loss)
            elif step_mode=='valloss':
                scheduler.step(val_metrics.loss)
            elif step_mode == 'valaccuracy':
                scheduler.step(val_metrics.accuracy)
            elif step_mode=='valauc':
                scheduler.step(val_metrics.auroc)
            
            history.append(_AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))

            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                 ''.format(plateau_terminate))
                breaking = True
            if num_classes ==2 :
                if epoch > best_train_auc_epoch + plateau_terminate:
                    tqdm_range.write('Breaking because of no improvement in training auc for {} epochs.'
                                    ''.format(plateau_terminate))
                    breaking = True

            else:
                if epoch > best_train_accuracy_epoch + plateau_terminate:
                    tqdm_range.write('Breaking because of no improvement in training accuracy for {} epochs.'
                                    ''.format(plateau_terminate))
                    breaking = True

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
     
    return history





class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result):
    loc = here / 'results' / name
    if not os.path.exists(loc):
        os.mkdir(loc)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])

    num += 1
    with open(loc / str(num), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)

def main_forecasting(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model, max_epochs,
         lr,weight_decay,writer,file, kwargs, step_mode,c1,c2, pos_weight=torch.tensor(1)):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None   

    model, regularise_parameters = make_model()  
    # mse loss function
    loss_fn = torch.nn.functional.mse_loss
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters)
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    # train function
    history = _train_loop_forecasting(model_name, train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, max_epochs,
                           writer, device, c1, c2, kwargs, step_mode)
    if os.path.isfile(file):
        os.remove(file)  
    
    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
        print(f"memory_usage:{memory_usage}")
    else:
        memory_usage = None
    result = _AttrDict(times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       parameters=_count_parameters(model),
                       history=history)
                    
    if name is not None:
        _save_results(name, result)
    return result



def main(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model, num_classes, max_epochs,
         lr,weight_decay,file, kwargs, step_mode,c1=0,c2=0, pos_weight=torch.tensor(1)):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None   
    
      
    model, regularise_parameters = make_model()  

    if num_classes == 2:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.functional.cross_entropy

    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    
    history = _train_loop(train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, max_epochs,
                          num_classes,  device, c1, c2, kwargs, step_mode, model_name)
    
    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
        print(f"memory_usage:{memory_usage}")
    else:
        memory_usage = None
    memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
    print(f"memory_usage:{memory_usage}")
    result = _AttrDict(times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       num_classes=num_classes,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       parameters=_count_parameters(model),
                       history=history,
                       )
    if name is not None:
        _save_results(name, result)
    return result





def make_model(name, input_channels, output_channels, hidden_channels, hidden_hidden_channels, 
    ode_hidden_hidden_channels,num_hidden_layers, file, 
    use_intensity,initial, output_time = 0):
    if name == 'learnable_forecasting':
        def make_model():
            func_k = models.FinalTanh( input_channels=input_channels, hidden_channels=hidden_channels,
                                       hidden_hidden_channels=hidden_hidden_channels,
                                        num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh2(input_channels=input_channels, hidden_channels=hidden_channels,
                                      hidden_hidden_channels=hidden_hidden_channels,
                                      num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f2(input_channels=input_channels,hhidden_channels=ode_hidden_hidden_channels, hidden_channels=hidden_channels) 
            mapping = models.Mapping_f(input_channels=input_channels, hidden_channels=hidden_channels)                          
            model = models.NeuralCDE_Learnable_forecasting(func_k=func_k,func_g = func_g, func_f=func_f, mapping=mapping, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels,file=file, initial=initial)
            return model, func_k 

    elif name == 'ncde':
        def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde_forecasting':
         def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'gruode':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels, output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name =='gruode_forecasting':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'dt':
        def make_model():
            model = models.GRU_dt(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'dt_forecasting':
        def make_model():
            model = models.GRU_dt_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'decay':
        def make_model():
            model = models.GRU_D(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'decay_forecasting':
        def make_model():
            model = models.GRU_D_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'odernn':
        def make_model():
            model = models.ODERNN(input_channels=input_channels, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'odernn_forecasting':
        def make_model():
            
            model = models.ODERNN_forecasting(input_channels=input_channels,output_time = output_time, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'learnable':
        def make_model():
            func_k = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_g = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            func_f = models.ODEFunc_f(input_channels=input_channels, hidden_channels=hidden_channels) 
            mapping = models.Mapping_f(input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.NeuralCDE_Learnable(func_k=func_k, func_g=func_g, func_f=func_f, mapping=mapping, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, file=file, initial=initial)
            return model, func_k
    else:
        raise ValueError("Unrecognised model name {}. Valid names are 'ncde', 'gruode', 'dt', 'decay' and 'odernn'."
                         "".format(name))
    return make_model
