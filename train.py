import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import *
from layer import SSCLP
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, f1_score, auc, confusion_matrix


def train_model(data_s, data_f, train_loader, test_loader, args, fold):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model = SSCLP(in_dim=args.dimensions, hid_dim=args.hidden1, out_dim=args.hidden2, decoder1=args.decoder1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCEWithLogitsLoss()
    loss_node = torch.nn.BCELoss()

    model = model.to(device)
    data_s = data_s.to(device)
    data_f = data_f.to(device)

    save_dir = f"../plots-LDA/fold{fold}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(save_dir)}")

    print('Start Training...')
    for epoch in range(args.epochs):
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        lbl_1 = torch.ones(997 * 2).to(device)
        lbl_2 = torch.zeros(997 * 2).to(device)
        lbl = torch.cat((lbl_1, lbl_2)).to(device)

        for i, (label, inp) in enumerate(train_loader):
            label = label.to(device)

            model.train()
            optimizer.zero_grad()

            output, log, z_clean, z_perturbed = model(data_s, data_f, inp)

            log = torch.squeeze(m(log))
            loss_class = loss_node(log, label.float())
            loss_constra = loss_fct(output, lbl)
            consis_loss = consistency_loss(z_clean, z_perturbed)

            loss_train = (
                loss_class
                + args.loss_ratio1 * loss_constra
                + args.loss_ratio2 * consis_loss
            )

            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train += label_ids.flatten().tolist()
            y_pred_train += log.flatten().tolist()

            if i % 100 == 0:
                print('epoch: {}/ iteration: {}/ loss_train: {:.4f}'.format(
                    epoch + 1, i + 1, loss_train.cpu().detach().numpy()
                ))

        roc_train = roc_auc_score(y_label_train, y_pred_train)
        print('epoch: {:04d} loss_train: {:.4f} auroc_train: {:.4f}'.format(
            epoch + 1, loss_train.item(), roc_train
        ))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    print("Optimization Finished!")

    # Final evaluation
    model.eval()
    auroc_test, prc_test, f1_test, loss_test, y_label, y_pred = test(model, test_loader, data_s, data_f, args, device)
    outputs = np.asarray([1 if i >= 0.5 else 0 for i in y_pred])  # <<< FIXED

    print('loss_test: {:.4f} auroc_test: {:.4f} auprc_test: {:.4f} f1_test: {:.4f}'.format(
        loss_test.item(), auroc_test, prc_test, f1_test
    ))

    # Printing 10 random prediction results
    print("\n10 Random Prediction Results (Predicted | True):")
    indices = np.random.choice(len(outputs), size=10, replace=False)
    for idx in indices:
        print(f"{outputs[idx]} | {y_label[idx]}")

    # Plotting
    fpr, tpr, _ = roc_curve(y_label, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'SSCLP (AUC = {auroc_test:.4f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve for LDA Prediction on dataset 1 (Fold {fold})')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f'roc_curve_LDA_fold{fold}.png'))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_label, y_pred)
    plt.figure()
    plt.plot(recall, precision, label=f'SSCP (AUPRC = {prc_test:.4f})', color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve for LDA Prediction on dataset 1 (Fold {fold})')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(save_dir, f'pr_curve_LDA_fold{fold}.png'))
    plt.close()

    cm = confusion_matrix(y_label, outputs)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f"{value}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm_normalized.flatten()]
    labels = [f"{name}\n{count}\n{percent}" for name, count, percent in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=labels, fmt='', cmap='Blues')
    plt.title(f'Confusion Matrix for LDA Prediction on dataset 1 (Fold {fold})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_LDA_fold{fold}.png'))
    plt.close()

    return model


def test(model, loader, data_s, data_f, args, device):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCEWithLogitsLoss()
    loss_node = torch.nn.BCELoss()

    lbl_1 = torch.ones(997 * 2).to(device)
    lbl_2 = torch.zeros(997 * 2).to(device)
    lbl = torch.cat((lbl_1, lbl_2)).to(device)

    model.eval()
    y_pred = []
    y_label = []

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            label = label.to(device)

            output, log, _, _ = model(data_s, data_f, inp)
            log = torch.squeeze(m(log))
            loss_class = loss_node(log, label.float())
            loss_constra = loss_fct(output, lbl)

            loss = loss_class + args.loss_ratio1 * loss_constra

            label_ids = label.to('cpu').numpy()
            y_label += label_ids.flatten().tolist()
            y_pred += log.flatten().tolist()

    return (
        roc_auc_score(y_label, y_pred),
        average_precision_score(y_label, y_pred),
        f1_score(y_label, np.asarray([1 if i >= 0.5 else 0 for i in y_pred])),
        loss,
        y_label,
        y_pred
    )
