import torch
import matplotlib.pyplot as plt
import sys

## Utils functions ##
def train_optim(model, trainloader, testloader, epochs, log_frequency, device, learning_rate=1e-4):
    model.to(device)  # we make sure the model is on the proper device

    # Multiclass classification setting, we use cross-entropy
    # note that this implementation requires the logits as input
    # logits: values prior softmax transformation
    loss_fn = torch.nn.MSELoss() # Fonction de loss pour de la régression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        model.train()  # we specify that we are training the model

        # At each epoch, the training set will be processed as a set of batches

        for batch_id, batch in enumerate(trainloader):
            images, labels = batch
            # we put the data on the same device
            images, labels = images.to(device), labels.to(device)
            y_pred = model(images)  # forward pass output=logits
            y_pred = y_pred.reshape(images.shape[0], 3, 96, 96)
            loss = loss_fn(y_pred, labels)

            if batch_id % log_frequency == 0:
                print("epoch: {:03d}, batch: {:03d}, loss: {:.3f} ".format(t + 1, batch_id + 1, loss.item()))

            optimizer.zero_grad()  # clear the gradient before backward
            loss.backward()  # update the gradient
            optimizer.step()  # update the model parameters using the gradient
            #break

        # Model evaluation after each step computing the accuracy
        evaluate_model(model, testloader, device, t)


def evaluate_model(model, testloader, device, epoch=0):
    print('Starting evaluation')
    model.eval()
    total = 0
    index = 0
    delta_max = 0
    delta_min = sys.maxsize
    for batch_id, batch in enumerate(testloader):
        images , labels = batch
        images , labels = images.to(device), labels.to(device)
        y_pred = model.forward(images) # forward computes the logits
        y_pred = y_pred.reshape(images.shape[0], 3, 96, 96)
        
        total += labels.size(0)
        y_pred = y_pred.detach()

        # Metrique d'évaluation du model
        evaluation = eval_metric(labels, y_pred)
        total += evaluation
        print("Delta : "+str(evaluation))

        if (evaluation > delta_max) :
            delta_max = evaluation
        if (evaluation < delta_min) :
            delta_min = evaluation

        # Sauvegarde de l'image déterioré
        #plt.figure(1)
        #plt.imshow(images[0].permute(1, 2, 0))
        #plt.show()
        #plt.savefig('images/base_'+str(index))

        # Sauvegarde de l'image attendu
        #plt.figure(1)
        #plt.imshow(labels[0].permute(1, 2, 0))
        #plt.show()
        #plt.savefig('images/attendu_'+str(index))

        # Sauvegarde de notre prediction
        #plt.figure(1)
        #plt.imshow(y_pred[0].permute(1, 2, 0))
        #plt.show()
        #plt.savefig('images/prediction_'+str(index))
        index += 1
    
        #break
    
    file = open('trained_model/log_model.txt', 'a')
    file.write("--------------- EPOCH "+str(epoch+1)+" ---------------\n")
    file.write("Average delta : "+str(int(total/index)) + "\n")
    file.write("Delta max : "+str(int(delta_max)) + "\n")
    file.write("Delta min : "+str(int(delta_min)) + "\n\n")
    file.close()


def eval_metric(img, pred):
    return torch.abs(img - pred).sum().item()