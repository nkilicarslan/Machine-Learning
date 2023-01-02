import torch
import torch.nn as nn
import numpy as np
import pickle

iteration_num = 250
class MLPModel(nn.Module):
    # I defined the hyperparameters. These are in the parameters of the init function.
    def __init__(self,number_of_hidden_layers,number_of_neurons,learning_rate,activation_function,epoch):
        # I initialized the parameters.
        super(MLPModel,self).__init__()
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.learning_rate = learning_rate
        self.number_of_hidden_layers = number_of_hidden_layers
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function
        self.epoch = epoch
        self.hidden_layer_output = None
        self.hidden_layer_output_2 = None
        self.output_layer = None
        # Since there are two options of the hidden layer, we have to handle this situation in the two different block.
        if number_of_hidden_layers == 1:
            self.layer1 = nn.Linear(784,number_of_neurons)
            self.layer2 = nn.Linear(number_of_neurons,10)
        if number_of_hidden_layers == 2:
            self.layer1 = nn.Linear(784,number_of_neurons)
            self.layer2 = nn.Linear(number_of_neurons,number_of_neurons)
            self.layer3 = nn.Linear(number_of_neurons,10)

        # nn.LeakyReLU()
        # nn.PReLU()
        self.activation_function = activation_function()
        self.softmax_function = nn.Softmax(dim=1)

    def forward(self,x):
        # In forward propogation since there are two options in the hidden layer,
        # I divided into to two if part.
        if self.number_of_hidden_layers == 1:
            hidden_layer_output = self.activation_function(self.layer1(x))
            output_layer = self.layer2(hidden_layer_output)
        if self.number_of_hidden_layers == 2:
            hidden_layer_output = self.activation_function(self.layer1(x))
            hidden_layer_output_2 = self.activation_function(self.layer2(hidden_layer_output))
            output_layer = self.layer3(hidden_layer_output_2)
        return output_layer

# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)

# In this part I created all the different model and then I created all different optimizer.
# Since I want to analyze all hyperparameter effect one by one, I changed the hyperparameter values one by one
# then I combined them.
model1 = MLPModel(1,8,0.001,nn.LeakyReLU,250)
optimizer1 = torch.optim.Adam(model1.parameters(),model1.learning_rate)
model2 = MLPModel(1,16,0.001,nn.LeakyReLU,250)
optimizer2 = torch.optim.Adam(model2.parameters(),model2.learning_rate)
model3 = MLPModel(1,8,0.002,nn.LeakyReLU,250)
optimizer3 = torch.optim.Adam(model3.parameters(),model3.learning_rate)
model4 = MLPModel(1,8,0.001,nn.PReLU,250)
optimizer4 = torch.optim.Adam(model4.parameters(),model4.learning_rate)
model5 = MLPModel(1,8,0.001,nn.LeakyReLU,500)
optimizer5 = torch.optim.Adam(model5.parameters(),model5.learning_rate)
model6 = MLPModel(2,8,0.001,nn.LeakyReLU,250)
optimizer6 = torch.optim.Adam(model6.parameters(),model6.learning_rate)
model7 = MLPModel(2,16,0.001,nn.LeakyReLU,250)
optimizer7 = torch.optim.Adam(model7.parameters(),model7.learning_rate)
model8 = MLPModel(2,16,0.002,nn.LeakyReLU,250)
optimizer8 = torch.optim.Adam(model8.parameters(),model8.learning_rate)
model9 = MLPModel(2,16,0.002,nn.PReLU,500)
optimizer9 = torch.optim.Adam(model9.parameters(),model9.learning_rate)
model10 = MLPModel(2,16,0.01,nn.PReLU,500)
optimizer10 = torch.optim.Adam(model10.parameters(),model10.learning_rate)

# In this following code part, I added the model and optimizer to the list.
keep_all_model_and_optimizer = []
keep_all_model_and_optimizer.append((model1,optimizer1))
keep_all_model_and_optimizer.append((model2,optimizer2))
keep_all_model_and_optimizer.append((model3,optimizer3))
keep_all_model_and_optimizer.append((model4,optimizer4))
keep_all_model_and_optimizer.append((model5,optimizer5))
keep_all_model_and_optimizer.append((model6,optimizer6))
keep_all_model_and_optimizer.append((model7,optimizer7))
keep_all_model_and_optimizer.append((model8,optimizer8))
keep_all_model_and_optimizer.append((model9,optimizer9))
keep_all_model_and_optimizer.append((model10,optimizer10))

#keep_all_mean_accuracy = []
loss_function = torch.nn.CrossEntropyLoss()
mean_iterations = []

# Based on the recitation slides, since I have a 10 model the first loop range in the(0,100)
for i in range(0,10):
    # According to the assignment.pdf, we have to run each model 10 times, so this for loop range in between (1,11)
    for j in range(1,11):
        list_validation_accuracy = []
        for iteration in range(1, keep_all_model_and_optimizer[i][0].epoch + 1):
            # I did the loss_value, prediction calculations and backpropagation in this part.
            keep_all_model_and_optimizer[i][1].zero_grad()
            predictions = keep_all_model_and_optimizer[i][0](x_train)
            loss_value = loss_function(predictions, y_train)
            loss_value.backward()
            keep_all_model_and_optimizer[i][1].step()

            with torch.no_grad():
                #I calculate the validation prediction, loss and accuracy in the following code piece.
                validation_predictions = keep_all_model_and_optimizer[i][0](x_validation)
                validation_loss = loss_function(validation_predictions, y_validation)
                validation_prediction_arr = torch.argmax(validation_predictions, dim=1)
                validation_accuracy = validation_prediction_arr[validation_prediction_arr == y_validation].size()[0] / \
                                      y_validation.size()[0] * 100
                list_validation_accuracy.append(validation_accuracy)
                #print("iteration: %d - Validation Loss: %f - Validation accuracy %f" % (iteration, validation_loss, validation_accuracy))
        mean_iterations.append(sum(list_validation_accuracy) / len(list_validation_accuracy))

each_model_accuracy_mean = []
count = 0
# In the following for loop, I calculate the overall mean of the each model
for i in range(1,101):
    count += mean_iterations[i-1]
    if i % 10 == 0:
        each_model_accuracy_mean.append(count/10)
        count = 0

each_model_standard_deviation = []
tmp_list = []
# In the following code piece, I calculate the each standard deviation of the each model
for i in range(1,101):
    tmp_list.append(mean_iterations[i-1])
    if i % 10 == 0:
        each_model_standard_deviation.append(np.std(tmp_list))
        tmp_list = []
# I calculate the all confidence interval of the all model.
each_model_confidence_interval = []
for i in range(0,10):
    conf_left = each_model_accuracy_mean[i] - 1.96 * each_model_standard_deviation[i] / 3.16227
    conf_right = each_model_accuracy_mean[i] + 1.96 * each_model_standard_deviation[i] / 3.16227
    each_model_confidence_interval.append((conf_left,conf_right))

max_index = 0
max_val = each_model_accuracy_mean[0]
# In the following for loop I calculate the maximum value and maximum index.
for i in range(1,10):
    if each_model_accuracy_mean[i] > max_val:
        max_val = each_model_accuracy_mean[i]
        max_index = i

# In order the create last dataset, I join the validation and train dataset.
x_new_dataset = torch.cat((x_validation,x_train),dim=0)
y_new_dataset = torch.cat((y_validation,y_train),dim=0)

# I do the same calculations above, actually the code block is the repetaiton of the above code block,just
# some parameters is changed.
test_means = []
for j in range(1, 11):
    list_test_accuracy = []
    for iteration in range(1, keep_all_model_and_optimizer[max_index][0].epoch + 1):
        keep_all_model_and_optimizer[max_index][1].zero_grad()
        predictions = keep_all_model_and_optimizer[max_index][0](x_new_dataset)
        loss_value = loss_function(predictions, y_new_dataset)
        loss_value.backward()
        keep_all_model_and_optimizer[max_index][1].step()

        with torch.no_grad():
            test_predictions = keep_all_model_and_optimizer[max_index][0](x_test)
            test_loss = loss_function(test_predictions, y_test)
            test_prediction_arr = torch.argmax(test_predictions, dim=1)
            test_accuracy = test_prediction_arr[test_prediction_arr == y_test].size()[0] / \
                                  y_test.size()[0] * 100
            list_test_accuracy.append(test_accuracy)
            #print("iteration: %d - Validation Loss: %f - Validation accuracy %f" % (iteration, test_loss, test_accuracy))
    test_means.append(sum(list_test_accuracy) / len(list_test_accuracy))

# calculate standard deviation for the last train
last_mean = 0
# I calculated the last train mean and also the standard deviation, confidence interval
for i in range(1,11):
    last_mean += test_means[i-1]
    if i % 10 == 0:
        last_mean = last_mean/10
last_standard_dev = np.std(test_means)
last_conf_left = last_mean - 1.96 * last_standard_dev / 3.16227
last_conf_right = last_mean + 1.96 * last_standard_dev / 3.16227

# In the following code session, I printed the all the accuracy_mean, standard deviation and the confidence interval
# in order to use the part3 report
for i in range(1,11):
    print("Model: %d - Mean Value: %f " % (i, each_model_accuracy_mean[i-1]))
for i in range(1,11):
    print("Model: %d - Standard Deviation: %f " % (i, each_model_standard_deviation[i-1]))
for i in range(1,11):
    print("Model: %d - confidence interval left: %f - confidence interval left: %f " % (i, each_model_confidence_interval[i - 1][0],each_model_confidence_interval[i - 1][1]))

# I also print the maximum mean, maximum value, and the last train mean, standard deviation, and the confidence interval
print("Max mean index",max_index+1)
print("Max mean value",max_val)

print("Last train mean: ",last_mean)
print("Last train standard deviation", last_standard_dev)
print("Last conf_left: ",last_conf_left)
print("Last conf_right: ",last_conf_right)
