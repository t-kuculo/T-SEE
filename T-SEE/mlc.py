from re import X
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from transformers import DistilBertTokenizer, DistilBertModel
import ast
from torch.nn import Module
import networkx as nx
import csv
from torch import nn
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from tqdm import tqdm


mode = "wde" #change to "dbpe" to train on the dbpedia dataset
print("start")

def initialize_model_and_device(model_path, ev_schema_path):
    '''
    Initializes the model and sets the device for training.

    Parameters:
    - model_path (str): Path to the pre-trained BERT model.
    - ev_schema_path (str): Path to the event schema file.

    Returns:
    - model (BERTClass): Initialized model.
    - device (torch.device): Configured device for training.
    '''
    model = BERTClass(ev_schema_path=ev_schema_path)	
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        device_ids = list(range(num_gpus))  
        model = nn.DataParallel(model, device_ids=device_ids)
        device = torch.device("cuda:0")  # Use the first GPU as the primary device
    else:
        device = torch.device("cpu")  # Use CPU if no GPU is available
    model.to(device)
    return model, device

def prepare_data(train_path, test_path, dev_path, tokenizer, max_len, batch_sizes):
    '''
    Prepares the datasets and dataloaders for training, testing, and validation.

    Parameters:
    - train_path (str): Path to the training data file.
    - test_path (str): Path to the testing data file.
    - dev_path (str): Path to the validation data file.
    - tokenizer (BertTokenizer): Tokenizer for BERT model.
    - max_len (int): Maximum length of tokenized sequences.
    - batch_sizes (dict): A dictionary with keys 'train', 'test', and 'dev' for respective batch sizes.

    Returns:
    - tuple: Tuple containing DataLoaders for training, testing, and validation datasets.
    '''

    # Function to create dataset from dataframe
    def create_dataset(dataframe):
        dataframe['list'] = dataframe[dataframe.columns[2:]].values.tolist()
        return dataframe[['text', 'list']].copy()

    # Load the datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    dev_df = pd.read_csv(dev_path)

    # Create datasets
    train_dataset = create_dataset(train_df)
    test_dataset = create_dataset(test_df)
    dev_dataset = create_dataset(dev_df)

    # Create the instance of CustomDataset
    training_set = CustomDataset(train_dataset, tokenizer, max_len)
    testing_set = CustomDataset(test_dataset, tokenizer, max_len)
    dev_set = CustomDataset(dev_dataset, tokenizer, max_len)

    # Set parameters for DataLoader
    train_params = {'batch_size': batch_sizes['train'],
                    'shuffle': True,
                    'num_workers': 0}

    test_params = {'batch_size': batch_sizes['test'],
                   'shuffle': False,
                   'num_workers': 0}

    dev_params = {'batch_size': batch_sizes['dev'],
                  'shuffle': False,
                  'num_workers': 0}

    # Create DataLoaders
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    dev_loader = DataLoader(dev_set, **dev_params)

    return training_loader, testing_loader, dev_loader


def optimize_thresholds(y_true, y_pred):
    ''' 
    Optimizes thresholds for each label in multilabel classification to maximize F1 score.

    Parameters:
    - y_true (np.array): The ground truth binary labels (2D array).
    - y_pred (np.array): The predicted probabilities (2D array).

    Returns:
    - list: A list of optimal thresholds for each label.

    This function individually optimizes the threshold for each label to maximize the F1 score.
    '''
    thresholds = []
    num_labels = y_true.shape[1]
    
    for i in range(num_labels):
        result = minimize_scalar(lambda x: -f1_score(y_true[:, i], y_pred[:, i] >= x),
                                 bounds=(0, 1),
                                 method='bounded')
        optimal_threshold = result.x
        thresholds.append(optimal_threshold)
    return thresholds

        
class CustomDataset(Dataset):
    ''' 
    Custom Dataset class for multilabel classification task using BERT.

    Attributes:
    - dataframe (pd.DataFrame): DataFrame containing the text and labels.
    - tokenizer (BertTokenizer): Tokenizer for our BERT model.
    - max_len (int): Maximum length of the tokenized input sequence.

    The class inherits from torch.utils.data.Dataset and overrides __len__ and __getitem__ methods.
    '''
    def __init__(self, dataframe, tokenizer, max_len):
        ''' Initialization of the CustomDataset object. '''
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        ''' Returns the length of the dataset. '''
        return len(self.text)

    def __getitem__(self, index):
        ''' 
        Retrieves an item by its index.

        Parameters:
        - index (int): Index of the item to retrieve.

        Returns:
        - dict: A dictionary containing input IDs, attention masks, token type IDs, and targets.
        '''
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BERTClass(torch.nn.Module):
    ''' 
    BERT model for multilabel classification.

    Attributes:
    - ev_classes (int): Number of event classes.
    - l1 (BertModel): Pretrained BERT model.
    - l2 (nn.Dropout): Dropout layer.
    - l3 (nn.Linear): Linear layer for classification.

    Inherits from torch.nn.Module and overrides the forward method.
    '''
    def __init__(self, ev_schema_path):
        ''' Initializes the BERTClass object with the specified schema path, such that we adjust the output layer to the number of event classes to be predicted.'''
        super(BERTClass, self).__init__()
        with open(ev_schema_path,"r") as f:
            lines = [line.rstrip() for line in f]
        self.ev_classes = len(ast.literal_eval(lines[0]))
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, self.ev_classes)
    
    def forward(self, ids, mask, token_type_ids):
        ''' 
        Forward pass of the model.

        Parameters:
        - ids (torch.tensor): Input IDs.
        - mask (torch.tensor): Attention mask.
        - token_type_ids (torch.tensor): Token type IDs.

        Returns:
        - torch.tensor: The output logits from the model.
        '''
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class FocalLoss(torch.nn.Module):
    ''' 
    Focal loss function for addressing class imbalance in multilabel classification.

    Attributes:
    - gamma (float): Focusing parameter.
    - alpha (float, optional): Balancing parameter.

    Inherits from torch.nn.Module and overrides the forward method.
    '''
    def __init__(self, gamma=2, alpha=None):
        ''' Initializes the FocalLoss object with specified gamma and alpha values. '''
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ''' 
        Calculates the focal loss.

        Parameters:
        - inputs (torch.tensor): Predicted logits.
        - targets (torch.tensor): Ground truth labels.

        Returns:
        - torch.tensor: Computed focal loss.
        '''
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return torch.mean(focal_loss)

def loss_fn(outputs, targets, gamma=2, alpha=None):
    ''' 
    Computes the focal loss for multilabel classification.

    Parameters:
    - outputs (torch.tensor): Predicted logits.
    - targets (torch.tensor): Ground truth labels.
    - gamma (float): Focusing parameter for FocalLoss.
    - alpha (float, optional): Balancing parameter for FocalLoss.

    Returns:
    - torch.tensor: Computed loss.
    '''
    return FocalLoss(gamma=gamma, alpha=alpha)(outputs, targets)


def train(epoch):
    ''' 
    Training function for one epoch.

    Parameters:
    - epoch (int): Current epoch number.
    - model (torch.nn.Module): The model to train.
    - training_loader (DataLoader): DataLoader for the training data.
    - optimizer (torch.optim.Optimizer): Optimizer for model training.
    - device (torch.device): Device on which to train the model.

    Performs training over one epoch and updates the model parameters.
    '''
    model.train()
    _ = 0
    for data in tqdm(training_loader):
        _+=1
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        sigm = torch.sigmoid(outputs)
        sigm = sigm.tolist()    
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)#/max(min(sigm))

        if _%100==0:
            print("Epoch: %d" %(epoch)+" , Loss: %.8f" %(loss))
        
        loss.backward()
        optimizer.step()


def validation(epoch, mode="dev"):
    ''' 
    Validation function for the model.

    Parameters:
    - epoch (int): Current epoch number.
    - model (torch.nn.Module): The model to validate.
    - loader (DataLoader): DataLoader for the validation or test data.
    - device (torch.device): Device on which to validate the model.

    Evaluates the model's performance on the validation or test dataset.

    Returns:
    - list: List of predicted outputs.
    - list: List of target values.
    '''
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    if mode=="dev":
        loader = dev_loader
    else:
        loader = testing_loader
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

if __name__=="__main__":
    
    # Set paths and parameters
    model_path = 'bert-base-uncased'
    ev_schema_path = f"../data/training/t2e/{mode}_unlabelled_event.schema"
    train_path = f'../data/training/mlc_data/{mode}_multilabel_train.csv'
    test_path = f'../data/training/mlc_data/{mode}_multilabel_test.csv'
    dev_path = f'../data/training/mlc_data/{mode}_multilabel_dev.csv'
    MAX_LEN = 256
    EPOCHS = 30
    LEARNING_RATE = 1e-05
    # Define batch sizes for the datasets
    batch_sizes = {
        'train': 12,
        'test': 6,
        'dev': 6
    }

    # Read the header from the test data CSV for later use in CSV writing
    with open(test_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)

    # Load the test dataset for later use in CSV writing
    test_df = pd.read_csv(f'../data/training/mlc_data/{mode}_multilabel_test.csv')
    test_df['list'] = test_df[test_df.columns[2:]].values.tolist()
    test_dataset = test_df[['text', 'list']].copy()

    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Initialize model and device
    model, device = initialize_model_and_device(model_path, ev_schema_path)

    # Prepare data
    training_loader, testing_loader, dev_loader = prepare_data(train_path, test_path, dev_path, tokenizer, MAX_LEN, batch_sizes)

    # Set optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


    # Initialize early stopping parameters
    max_score = 0.0  # Maximum F1 score
    patience = 5  # Number of epochs to wait before stopping training
    epochs_no_improve = 0  # Counter for epochs without improvement
    early_stop = False  # Flag to indicate whether to stop training

    for epoch in range(EPOCHS):
        train(epoch)
        outputs, targets = validation(epoch)
        outputs = np.array(outputs)
        indices = np.argmax(outputs, axis = 1)

        targets = np.array(targets)
        outputs = outputs >= 0.5

        #optimal_thresholds = optimize_thresholds(targets, outputs)
        #outputs = outputs >= optimal_thresholds

        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        f1_score_weighted= metrics.f1_score(targets, outputs, average='weighted')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        print(f"F1 Score (weighted) = {f1_score_weighted}")
        if f1_score_micro > max_score:
            max_score = f1_score_micro
            epochs_no_improve = 0
            

            torch.save(model.state_dict(), f"{mode}_mlc_model")
            print("Test results:")
            outputs, targets = validation(epoch, mode="test")
            outputs = np.array(outputs)
            indices = np.argmax(outputs, axis = 1)

            outputs = outputs >= 0.5
            #outputs = outputs >= optimal_thresholds
            targets = np.array(targets)
            classes = list(pd.read_csv(f"../data/training/mlc_data/{mode}_multilabel_test.csv").columns)[2:]
            tp = 0
            fp = 0
            fn = 0
            for sentence_target, sentence_output in zip(targets,outputs):
                ground_classes = [cl for cl, target in zip(classes, sentence_target) if target!=False] 
                predicted_classes = [cl for cl, target in zip(classes, sentence_output) if target!=False] 
                for cl in ground_classes:
                    if cl not in predicted_classes:
                        fn+=1
                    else:
                        tp+=1
                        predicted_classes.remove(cl)
                for cl in predicted_classes:
                    if cl not in ground_classes:
                        fp+=1
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            f1 = 2*p*r/(p+r)
            

            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            f1_score_weighted= metrics.f1_score(targets, outputs, average='weighted')
            print(f"Accuracy Score = {accuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")
            print(f"F1 Score (weighted) = {f1_score_weighted}")
            #best_wd2 with loss division
            with open(f"../evaluation/output/minority_classes/mlc_output/{mode}_with_focal_loss.csv", "w") as f, open(f'../data/training/mlc_data/{mode}_multilabel_test.csv',"r") as f2:
                writer = csv.DictWriter(f, fieldnames=header, delimiter = '\t',  quoting=csv.QUOTE_NONE, quotechar='')
                reader = csv.reader(f2)
                next(reader, None)
                for output, input, target, row  in zip(outputs, test_dataset["text"], targets, reader):
                    d = {}
                    for o, h in zip(output,header[2:]):
                        d[h] = o
                    d["id"] = row[0]
                    d["text"] = row[1]
                    writer.writerow(d)
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=patience:
                print("Early stopping!")
                break


