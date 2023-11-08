import torch
import numpy as np

def pretrain_G(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, modes in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        modes = modes.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, targets, modes).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches

    return average_loss


def evaluate_G(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, modes in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            modes = modes.to(device)
            outputs = model(inputs, targets, modes).to(device)
            loss = criterion(outputs, targets)
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)

    average_loss = total_loss / total_samples

    return average_loss

def pretrain_D(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, modes in dataloader:
        targets = targets.to(device)
        real_label = torch.ones(targets.shape[0]).to(device)
        fake_label = torch.zeros(targets.shape[0]).to(device)
        noise_target = add_noise_to_trajectory(targets, 1.5)

        outputs = model(targets).to(device)
        loss_real = criterion(outputs, real_label.long())

        outputs = model(noise_target).to(device)
        loss_fake = criterion(outputs, fake_label.long())

        loss = loss_fake + loss_real
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches

    return average_loss


def evaluate_D(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, modes in dataloader:
            targets = targets.to(device)
            outputs = model(targets).to(device)
            real_label = torch.ones(targets.shape[0]).to(device)
            loss = criterion(outputs, real_label.long())
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)

    average_loss = total_loss / total_samples

    return average_loss


def pretrain_C(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, modes in dataloader:
        targets = targets.to(device)
        modes = modes.to(device)

        optimizer.zero_grad()
        outputs = model(targets).to(device)
        loss = criterion(outputs, modes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches

    return average_loss


def evaluate_C(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, modes in dataloader:
            targets = targets.to(device)
            modes = modes.to(device)
            outputs = model(targets, modes).to(device)
            loss = criterion(outputs, modes)
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)

    average_loss = total_loss / total_samples

    return average_loss


def add_noise_to_trajectory(trajectory, noise_level):
    noise = torch.randn_like(trajectory) * noise_level
    noisy_trajectory = trajectory + noise
    return noisy_trajectory
