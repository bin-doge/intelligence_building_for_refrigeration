from intelligence_building import *
import torch
from common.utils.analysis.a_distance import calculate
from common.utils.analysis.tsne import visualize
from copy import deepcopy
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy

def main():   
    
    data = dataset.DatasetLoader()
    split_string = "***************************************************************************************************************"
    for i in range(11):
        alpha = 0.1*i
        logger = log.Log(f'alpha_{alpha}')
        X_train,X_test,y_train,y_test = data.get_data(alpha)
        
        #svm result
        svm = traditional.SVM()
        svm.fit(X_train,y_train)
        svm_result = svm.svm.score(X_test,y_test)
        logger.info(f"SVM: {svm_result}")
        logger.info(split_string)
        #rf result
        rf = traditional.RandomForest()
        rf.fit(X_train,y_train)
        rf_result = rf.rf.score(X_test,y_test)
        logger.info(f"rf: {rf_result}")
        logger.info(split_string)

        #pytorch area
        train_set = dataset.TorchDataset(X_train,y_train)
        test_set = dataset.TorchDataset(X_test,y_test)

        #ANN_all result
        ANN_all_train_loader = dataset.torch_dataset_to_dataloader(train_set,batch_size=len(train_set))
        model = deeplearning.BasicModel(8,29)
        ANN_all_optimzer = torch.optim.Adam(model.parameters())
        for i in range(301):
            deeplearning.train(model,ANN_all_train_loader,ANN_all_optimzer,torch.nn.functional.cross_entropy,i)
            if i>=250 & i%10==0:
                model.reset_batchnorm(train_set)
                train_acc = deeplearning.evaluate(model,X_train,y_train)
                test_acc = deeplearning.evaluate(model,X_test,y_test)
                logger.info(f"ANN_all: {i} times, train_acc: {train_acc}, test_acc: {test_acc}")

        f_train = model.get_numpy_feature(X_train)
        f_test = model.get_numpy_feature(X_test)

        visualize(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),filename=f'alpha_{alpha}_ann_all.svg')
        a_dis = calculate(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),device='cuda')
        a_dis = a_dis.item()
        logger.info(f"ANN_all: A_distance: {a_dis}")
        logger.info(split_string)

        #ANN_32 result
        model.init_parameters()
        source_loader = dataset.torch_dataset_to_dataloader(train_set)
        target_loader = dataset.torch_dataset_to_dataloader(test_set)
        ANN_32_optimizer = torch.optim.Adam(model.parameters())
        for i in range(301):
            deeplearning.train(model,source_loader,ANN_32_optimizer,torch.nn.functional.cross_entropy,i)
            if i>=250 & i%10==0:
                model.reset_batchnorm(train_set)
                train_acc = deeplearning.evaluate(model,X_train,y_train)
                test_acc = deeplearning.evaluate(model,X_test,y_test)
                logger.info(f"ANN_32: {i} times, train_acc: {train_acc}, test_acc: {test_acc}")

        f_train = model.get_numpy_feature(X_train)
        f_test = model.get_numpy_feature(X_test)

        visualize(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),filename=f'alpha_{alpha}_ann_32.svg')
        a_dis = calculate(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),device='cuda')
        a_dis = a_dis.item()
        logger.info(f"ANN_32: A_distance: {a_dis}")
        logger.info(split_string)

        model.set_device('cpu')
        #DANN result
        dann_model = deepcopy(model)
        dann_model.set_device('cuda')
        dis = DomainDiscriminator(512,512)
        dann_optimizer = torch.optim.Adam([{'params':dann_model.parameters()}]+dis.get_parameters())
        dann_loss = DomainAdversarialLoss(dis)
        for i in range(301):
            deeplearning.train_dann(dann_model,dann_loss,source_loader,target_loader,dann_optimizer,i)
            if i>=250 % i%10==0:
                dann_model.reset_batchnorm(X_train)
                train_acc = deeplearning.evaluate(dann_model,X_train,y_train)
                test_acc = deeplearning.evaluate(dann_model,X_test,y_test)
                logger.info(f"DANN: {i} times, train_acc: {train_acc}, test_acc: {test_acc}")

        f_train = dann_model.get_numpy_feature(X_train)
        f_test = dann_model.get_numpy_feature(X_test)

        visualize(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),filename=f'alpha_{alpha}_dann.svg')
        a_dis = calculate(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),device='cuda')
        a_dis = a_dis.item()
        logger.info(f"DANN: A_distance: {a_dis}")
        logger.info(split_string)



        dann_model.set_device('cpu')
        dann_loss.to('cpu')
        #DAN result
        dan_model = deepcopy(model)
        dan_model.set_device('cuda')
        kernels = (GaussianKernel(0.5),GaussianKernel(1.0),GaussianKernel(1.5))
        dan_loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        dan_optimizer = torch.optim.Adam(dan_model.parameters())
        for i in range(301):
            deeplearning.train_dan(dan_model,dan_loss,source_loader,target_loader,dan_optimizer,i)
            if i>=250 % i%10==0:
                dan_model.reset_batchnorm(X_train)
                train_acc = deeplearning.evaluate(dan_model,X_train,y_train)
                test_acc = deeplearning.evaluate(dan_model,X_test,y_test)
                logger.info(f"DAN: {i} times, train_acc: {train_acc}, test_acc: {test_acc}")

        f_train = dan_model.get_numpy_feature(X_train)
        f_test = dan_model.get_numpy_feature(X_test)

        visualize(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),filename=f'alpha_{alpha}_dan.svg')
        a_dis = calculate(torch.tensor(f_train,dtype=torch.float32),torch.tensor(f_test,dtype=torch.float32),device='cuda')
        a_dis = a_dis.item()
        logger.info(f"DAN: A_distance: {a_dis}")
        logger.info(split_string)
        dan_model.set_device('cpu')
        dan_loss.to('cpu')






        

            



        




if __name__ == '__main__':
    main()
