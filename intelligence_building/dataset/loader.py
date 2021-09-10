import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

__all__ = ['DatasetLoader']
class DatasetLoader:


    """
    Read and process the csv_file using corresponding handler, the default file is RP_1043.csv

    Available Attribute:
    data: original data of csv_file, type: pd.DataFrame
    X: feature array, type: np.Array 
    y: label array, type: np.array
    oper_condition: opertion condition array, type: np.array
    feature_name: feature name of X, type: list
    oper_condition_name: feature name of oper_condition, type: np.array
    mapping_dict: the relationship of label and its corresponding fault level, type: dict


    Available Function:
    get_data: return a copy of training data and testing data

    """


    def __init__(self,csv_name='RP_1043.csv'):
        self.data = pd.read_csv(__file__.replace('loader.py',csv_name))
        if csv_name == 'RP_1043.csv':
            self.__RP_1043_Handler()
        
        
    def __RP_1043_Handler(self):
        # get the feature array
        self.feature_name = ['TEI','TCI','TCO','TCA','TEA','TRC_sub','TO_sump','TR_dis']
        self.oper_condition_name = ['TCI','TEO','Evap.Tons']
        self.X = self.data.loc[:,self.feature_name].copy().values
        self.oper_condition = self.data.loc[:,self.oper_condition_name].copy().values

        #get the label array
        self.data['fault_level'] = self.data.fault + self.data.level.apply(str)
        self.mapping_dict = {'normal0':0}
        i = 1
        for index in self.data.fault_level.value_counts().sort_index().index:
            if index == 'normal0':
                pass
            else:
                self.mapping_dict[index] = i
                i += 1
        self.data['label'] = self.data.fault_level.map(self.mapping_dict)
        self.y = self.data.label.copy().values
        

    def get_data(self,alpha=0.5,train_size=0.7,split_column=-1,requires_mix=True):
        """
        split the data into training data and testing data, see split_data function for details
        
        Parameters:
        alpha: the percentage of out-of-distribution testing data, float, default 0.5
        train_size: the size of training data, float, default 0.7
        split_column: the column index of oper_condition, which used for splitting data, int, default -1 (last column)
        requires_mix: whether mix in-distribution testing data and out-of-distribution testing data, boolean, default True

        return a tuple of X_train,X_test,y_train,y_test (mix=True)
               or X_train,X_itest,X_otest,y_train,y_itest,y_otest (mix=False)
        """

        return self.split_data(self.X,self.y,self.oper_condition,alpha=alpha,train_size=train_size,split_column=split_column,requires_mix=requires_mix)


    @staticmethod
    def split_data(X,y,oper_condition,alpha,train_size,split_column,requires_mix):
        """
        split the X,y into in-distribution and out-of-distribution data
        for each unique label in y, we random cut a continuous range of oper_condition split_column as out-of-distribution data
        and then random select other data as in-distribution data

        Parameters:
        X: feature_array, np.array
        y: label_array, np.array
        oper_condition: oper_condition array use for splitting data, np.array
        alpha: the percentage of out_of_distribution testing data in testing data, float
        train_size: size of training data, float
        split_column: the column index of oper_condition, int
        requires_mix: whether mix in-distribution testing data and out-of-distribution testing data, boolean 
                
        return a tuple of X_train,X_test,y_train,y_test (mix=True)
               or X_train,X_itest,X_otest,y_train,y_itest,y_otest (mix=False)
        """

        itest_size = (1-train_size)*(1-alpha)
        otest_size = (1-train_size)*alpha
        X_train,y_train = [],[]
        X_itest,y_itest = [],[]
        X_otest,y_otest = [],[]

        for label in np.unique(y):
            #get the sub-array of each unique label
            mask = (y==label)
            X_middle = X[mask]
            y_middle = y[mask]
            oper_middle = oper_condition[mask]

            #sort the sub-array using the split_column
            sort_index = np.argsort(oper_middle[:,split_column])
            X_middle = X_middle[sort_index]
            y_middle = y_middle[sort_index]
            oper_middle = oper_middle[sort_index]

            #compute the size of each sub-train, sub-itest, sub-otest array
            total = len(X_middle)
            train_total = max(int(total*train_size),1)
            itest_total = max(int(total*itest_size),1)
            otest_total = max(int(total*otest_size),1)

            #get the sub-otest array, use randint to generate the random continuous range of testing data 
            otest_start = np.random.randint(total-otest_total)
            otest_index = np.arange(start=otest_start,stop=otest_start+otest_total,dtype=np.int)
            ti_index = np.delete(np.arange(total),otest_index)
            X_otest.append(X_middle[otest_index])
            y_otest.append(y_middle[otest_index])

            #get the sub-itest array and sub-train array

            sub_X_train,sub_X_itest,sub_y_train,sub_y_itest = train_test_split(X_middle[ti_index],y_middle[ti_index],test_size=itest_total,shuffle=True)
            X_train.append(sub_X_train)
            y_train.append(sub_y_train)
            X_itest.append(sub_X_itest)
            y_itest.append(sub_y_itest)
        
        #concatenate each sub-array
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_otest = np.concatenate(X_otest)
        y_otest = np.concatenate(y_otest)
        X_itest = np.concatenate(X_itest)
        y_itest = np.concatenate(y_itest)

        #return the result
        if requires_mix:
            X_test = np.concatenate((X_itest,X_otest))
            y_test = np.concatenate((y_itest,y_otest))
            return X_train,X_test,y_train,y_test
        else:
            return X_train,X_itest,X_otest,y_train,y_itest,y_otest


        
