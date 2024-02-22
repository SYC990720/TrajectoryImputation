import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from Generator import Encoder, Decoder, Seq2Seq, TrajectoryGenerator
from Discriminator import TrajectoryDiscriminator, TrajectoryClassifier
from utils import pretrain_G, evaluate_G, pretrain_D, evaluate_D, pretrain_C, evaluate_C
from Dataset import TrajectoryDataset, collate_fn
import argparse

parser = argparse.ArgumentParser(description='TIGAN')
parser.add_argument('--walk_file_path', dest='walk_file_path', default='./Data/walk.csv')
parser.add_argument('--bike_file_path', dest='bike_file_path', default='./Data/bike.csv')
parser.add_argument('--car_file_path', dest='car_file_path', default='./Data/car.csv')
parser.add_argument('--bus_file_path', dest='bus_file_path', default='./Data/bus.csv')
parser.add_argument('--pretrain_G_epoch', dest='pretrain_G_epoch', default=100)
parser.add_argument('--pretrain_D_epoch', dest='pretrain_D_epoch', default=100)
parser.add_argument('--pretrain_C_epoch', dest='pretrain_C_epoch', default=100)
parser.add_argument('--train_epoch', dest='train_epoch', default=500)
parser.add_argument('--missing_rate', dest='missing_rate', default=0.2)
parser.add_argument('--input_dim', dest='input_dim', default=2)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=64)
parser.add_argument('--num_layers', dest='num_layers', default=1)
parser.add_argument('--output_dim', dest='output_dim', default=2)
parser.add_argument('--flag', dest='flag', default=0)
parser.add_argument('--pretrain_G', dest='pretrain_G', type=bool, default=False)
parser.add_argument('--pretrain_D', dest='pretrain_D', type=bool, default=False)
parser.add_argument('--pretrain_C', dest='pretrain_C', type=bool, default=False)
parser.add_argument('--train', dest='train', type=bool, default=False)
parser.add_argument('--test', dest='test', type=bool, default=True)
parser.add_argument('--model_path', dest='model_path', default='model/best_seq2seq_model.pth')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

walk_file_path = args.walk_file_path
bike_file_path = args.bike_file_path
car_file_path = args.car_file_path
bus_file_path = args.bus_file_path

missing_rate = args.missing_rate

walk_dataset = TrajectoryDataset(walk_file_path, 0, missing_rate)
bike_dataset = TrajectoryDataset(bike_file_path, 1, missing_rate)
car_dataset = TrajectoryDataset(car_file_path, 2, missing_rate)
bus_dataset = TrajectoryDataset(bus_file_path, 3, missing_rate)

all_dataset = ConcatDataset([walk_dataset, bike_dataset, car_dataset, bus_dataset])

num_samples = len(all_dataset)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_size = int(train_ratio * num_samples)
val_size = int(val_ratio * num_samples)
test_size = num_samples - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

input_dim = args.input_dim
hidden_dim = args.hidden_dim
num_layers = args.num_layers
output_dim = args.output_dim
flag = args.flag

model = TrajectoryGenerator(input_dim, hidden_dim, output_dim, flag, device).to(device)
G_optimizer = Adam(model.parameters(), lr=0.01)
G_criterion = nn.MSELoss().to(device)

if args.pretrain_G:
    for epoch in range(args.pretrain_G_epoch):
        train_loss = pretrain_G(model, train_dataloader, G_optimizer, G_criterion, device)
        val_loss = evaluate_G(model, val_dataloader, G_criterion, device)

        if epoch == 0:
            best_val_loss = val_loss
        print(f"Epoch [{epoch + 1}/{args.pretrain_G_epoch}], Train Loss: {train_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{args.pretrain_G_epoch}], Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model/best_model.pth"
            torch.save(model.state_dict(), model_path)
            print("Best model saved!")


input_size = 2
hidden_size = 64
num_classes = 2
Discriminator = TrajectoryDiscriminator(input_size, hidden_size, num_classes).to(device)
Dis_optimizer = Adam(Discriminator.parameters(), lr=0.001)
Dis_criterion = nn.CrossEntropyLoss().to(device)

if args.pretrain_D:
    for epoch in range(args.pretrain_D_epoch):
        train_loss = pretrain_D(Discriminator, train_dataloader, Dis_optimizer, Dis_criterion, device)
        val_loss = evaluate_D(Discriminator, val_dataloader, Dis_criterion, device)

        if epoch == 0:
            best_val_loss = val_loss

        print(f"Epoch [{epoch + 1}/{args.pretrain_D_epoch}], Train Loss: {train_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{args.pretrain_D_epoch}], Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model/best_discriminator.pth"
            torch.save(Discriminator.state_dict(), model_path)
            print("Best discriminator saved!")


input_size = 2
hidden_size = 64
num_classes = 4
Classifier = TrajectoryClassifier(input_size, hidden_size, num_classes).to(device)
Cla_optimizer = Adam(Classifier.parameters(), lr=0.001)
Cla_criterion = nn.CrossEntropyLoss().to(device)

if args.pretrain_C:
    for epoch in range(args.pretrain_C_epoch):
        train_loss = pretrain_D(Classifier, train_dataloader, Cla_optimizer, Cla_criterion, device)
        val_loss = evaluate_D(Classifier, val_dataloader, Cla_criterion, device)
        if epoch == 0:
            best_val_loss = val_loss

        print(f"Epoch [{epoch + 1}/{args.pretrain_C_epoch}], Train Loss: {train_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{args.pretrain_C_epoch}], Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model/best_Classifier.pth"
            torch.save(Classifier.state_dict(), model_path)
            print("Best Classifier saved!")

if args.train:
    for epoch in range(args.train_epoch):
        gloss = 0.0
        dloss = 0.0
        closs = 0.0

        for inputs, targets, modes in train_dataloader:
            model.train()
            Discriminator.train()
            Classifier.train()
            inputs = inputs.to(device)
            targets = targets.to(device)
            modes = modes.to(device)

            G_optimizer.zero_grad()
            fake_data = model(inputs, targets, modes).to(device)
            generator_loss = Dis_criterion(Discriminator(fake_data).to(device), torch.ones(fake_data.shape[0]).long().to(device))
            generator_loss.backward(retain_graph=True)
            G_optimizer.step()

            real_loss = Dis_criterion(Discriminator(targets).to(device), torch.ones(targets.shape[0]).long().to(device))
            fake_loss = Dis_criterion(Discriminator(fake_data).to(device), torch.zeros(fake_data.shape[0]).long().to(device))
            discriminator_loss = real_loss + fake_loss
            Dis_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            Dis_optimizer.step()

            Cla_optimizer.zero_grad()
            classifier_loss = Cla_criterion(Classifier(targets), modes)
            classifier_loss.backward()
            Cla_optimizer.step()

            gloss += generator_loss
            dloss += discriminator_loss
            closs += classifier_loss

        val_loss = evaluate_G(model, val_dataloader, G_criterion, device)

        if epoch == 0:
            best_val_loss = val_loss

        print(f"Epoch [{epoch + 1}/{args.train_epoch}], Train Loss: {train_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{args.train_epoch}], Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model/best_model.pth"
            torch.save(model.state_dict(), model_path)
            print("Best model saved!")

        print(f"Epoch [{epoch + 1}/{args.train_epoch}], Generator Loss: {gloss/len(train_dataset)}, Discriminator Loss: {dloss/len(train_dataset)}, Classifier Loss: {closs/len(train_dataset)}")

if args.test:
    test_criterion = nn.MSELoss().to(device)
    model.load_state_dict(torch.load(args.model_path))

    with torch.no_grad():
        test_loss = evaluate_G(model, test_dataloader, test_criterion, device)
        print("test_loss: ", test_loss)
